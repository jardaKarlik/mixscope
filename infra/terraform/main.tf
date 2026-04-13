terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  # Store state in GCS so the team can share it
  backend "gcs" {
    bucket = "mixscope-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ─── Enable required APIs ─────────────────────────────────────────────────────
resource "google_project_service" "apis" {
  for_each = toset([
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
    "run.googleapis.com",
    "storage.googleapis.com",
    "cloudscheduler.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
  ])
  service            = each.key
  disable_on_destroy = false
}

# ─── Service Account ──────────────────────────────────────────────────────────
resource "google_service_account" "scraper" {
  account_id   = "mixscope-scraper"
  display_name = "Mixscope Scraper Service Account"
  description  = "Used by Cloud Run Jobs to access Secret Manager, Cloud SQL, GCS"
}

# Least-privilege IAM bindings
resource "google_project_iam_member" "scraper_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.scraper.email}"
}

resource "google_project_iam_member" "scraper_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.scraper.email}"
}

resource "google_project_iam_member" "scraper_storage_writer" {
  project = var.project_id
  role    = "roles/storage.objectCreator"
  member  = "serviceAccount:${google_service_account.scraper.email}"
}

resource "google_project_iam_member" "scraper_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.scraper.email}"
}

# ─── Secret Manager — all API keys ───────────────────────────────────────────
# Secrets are created empty here; you populate them via gcloud or the console.
# The scraper reads them at runtime — never stored in code or config files.
resource "google_secret_manager_secret" "secrets" {
  for_each  = toset([
    "spotify-client-id",
    "spotify-client-secret",
    "youtube-api-key",
    "soundcloud-client-id",
    "soundcloud-client-secret",
    "mixcloud-client-id",
    "mixcloud-client-secret",
    "db-password",
  ])
  secret_id = "mixscope-${each.key}"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

# ─── Cloud Storage — raw scrape backups ───────────────────────────────────────
resource "google_storage_bucket" "raw_backups" {
  name          = "${var.project_id}-mixscope-raw"
  location      = var.region
  force_destroy = false

  # Lifecycle: keep raw JSON for 90 days, then delete
  lifecycle_rule {
    condition { age = 90 }
    action    { type = "Delete" }
  }

  versioning { enabled = false }
}

# Terraform state bucket (create this manually BEFORE running terraform init)
# resource "google_storage_bucket" "tf_state" {
#   name     = "mixscope-terraform-state"
#   location = var.region
#   versioning { enabled = true }
# }

# ─── Cloud SQL — PostgreSQL 15 ────────────────────────────────────────────────
resource "google_sql_database_instance" "main" {
  name             = "mixscope-db"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier              = var.db_tier   # db-g1-small for dev, db-custom-2-4096 for prod
    availability_type = "ZONAL"       # upgrade to REGIONAL for HA prod

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"  # 3am backup
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 14
      }
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }

    insights_config {
      query_insights_enabled = true
    }

    ip_configuration {
      # Private IP only — scrapers connect via Cloud SQL Auth Proxy
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
    }
  }

  deletion_protection = true
  depends_on          = [google_project_service.apis, google_service_networking_connection.private_vpc]
}

resource "google_sql_database" "mixscope" {
  name     = "mixscope"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "scraper" {
  name     = "scraper"
  instance = google_sql_database_instance.main.name
  # Password stored in Secret Manager as mixscope-db-password
  password = data.google_secret_manager_secret_version.db_password.secret_data
}

data "google_secret_manager_secret_version" "db_password" {
  secret = google_secret_manager_secret.secrets["db-password"].secret_id
}

# ─── VPC for private Cloud SQL ────────────────────────────────────────────────
resource "google_compute_network" "vpc" {
  name                    = "mixscope-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "mixscope-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
}

resource "google_compute_global_address" "private_ip" {
  name          = "mixscope-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "private_vpc" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip.name]
}

# ─── Artifact Registry — Docker images for scrapers ──────────────────────────
resource "google_artifact_registry_repository" "scrapers" {
  location      = var.region
  repository_id = "mixscope-scrapers"
  format        = "DOCKER"
  description   = "Docker images for Mixscope scraper jobs"
}

# ─── Cloud Run Jobs — one per source ─────────────────────────────────────────
locals {
  scraper_image = "${var.region}-docker.pkg.dev/${var.project_id}/mixscope-scrapers/scraper:latest"

  scrapers = {
    "spotify"         = { schedule = "0 2 * * 1", description = "Spotify playlist co-presence" }
    "youtube"         = { schedule = "0 3 * * 1", description = "YouTube DJ set tracklists" }
    "mixcloud"        = { schedule = "0 4 * * 1", description = "Mixcloud set tracklists" }
    "soundcloud"      = { schedule = "0 5 * * 1", description = "SoundCloud set descriptions" }
    "onzerotracklists"= { schedule = "0 6 * * 1", description = "1001Tracklists DJ sets" }
    "residentadvisor" = { schedule = "0 7 * * 1", description = "Resident Advisor podcasts" }
  }
}

resource "google_cloud_run_v2_job" "scrapers" {
  for_each = local.scrapers
  name     = "mixscope-scraper-${each.key}"
  location = var.region

  template {
    template {
      service_account = google_service_account.scraper.email
      max_retries     = 2
      timeout         = "3600s"  # 1 hour max per run

      containers {
        image = local.scraper_image

        env {
          name  = "SCRAPER_SOURCE"
          value = each.key
        }
        env {
          name  = "GCP_PROJECT"
          value = var.project_id
        }
        env {
          name  = "GCS_BUCKET"
          value = google_storage_bucket.raw_backups.name
        }
        env {
          name  = "DB_INSTANCE"
          value = google_sql_database_instance.main.connection_name
        }
        env {
          name  = "DB_NAME"
          value = google_sql_database.mixscope.name
        }
        env {
          name  = "DB_USER"
          value = google_sql_user.scraper.name
        }

        resources {
          limits = {
            cpu    = "1"
            memory = "1Gi"
          }
        }

        # Cloud SQL Auth Proxy sidecar
        # Connects to Cloud SQL via private IP without exposing credentials
      }

      # Cloud SQL Auth Proxy as sidecar
      containers {
        image = "gcr.io/cloud-sql-connectors/cloud-sql-proxy:2.8.0"
        args  = [
          "--structured-logs",
          "--port=5432",
          google_sql_database_instance.main.connection_name,
        ]
        resources {
          limits = {
            cpu    = "0.5"
            memory = "256Mi"
          }
        }
      }

      vpc_access {
        network_interfaces {
          network    = google_compute_network.vpc.id
          subnetwork = google_compute_subnetwork.subnet.id
        }
        egress = "PRIVATE_RANGES_ONLY"
      }
    }
  }

  depends_on = [google_project_service.apis]
}

# ─── Cloud Scheduler — weekly trigger per scraper ────────────────────────────
resource "google_cloud_scheduler_job" "scraper_schedules" {
  for_each = local.scrapers
  name     = "trigger-mixscope-scraper-${each.key}"
  schedule = each.value.schedule
  region   = var.region
  timezone = "Europe/Prague"

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/mixscope-scraper-${each.key}:run"

    oauth_token {
      service_account_email = google_service_account.scraper.email
    }
  }
}

# ─── Outputs ──────────────────────────────────────────────────────────────────
output "db_connection_name" {
  value       = google_sql_database_instance.main.connection_name
  description = "Cloud SQL connection name for Auth Proxy"
}

output "scraper_sa_email" {
  value       = google_service_account.scraper.email
  description = "Service account email for scraper jobs"
}

output "artifact_registry_url" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/mixscope-scrapers"
  description = "Docker registry URL for scraper images"
}

output "raw_bucket" {
  value       = google_storage_bucket.raw_backups.name
  description = "GCS bucket for raw scrape backups"
}
