variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region — europe-west1 (Frankfurt) is closest to CZ"
  type        = string
  default     = "europe-west1"
}

variable "db_tier" {
  description = "Cloud SQL machine tier. db-g1-small for dev (~$25/mo), db-custom-2-4096 for prod"
  type        = string
  default     = "db-g1-small"
}
