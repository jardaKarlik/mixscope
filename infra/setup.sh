#!/bin/bash
# infra/setup.sh
# ==============
# One-shot GCP bootstrap. Run this ONCE to set up the project.
# After this, Terraform manages everything else.
#
# Usage:
#   chmod +x infra/setup.sh
#   ./infra/setup.sh YOUR_GCP_PROJECT_ID

set -euo pipefail

PROJECT_ID="${1:-}"
REGION="europe-west1"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Usage: ./infra/setup.sh YOUR_GCP_PROJECT_ID"
  exit 1
fi

echo "========================================"
echo "Mixscope GCP Bootstrap"
echo "Project: $PROJECT_ID"
echo "Region:  $REGION"
echo "========================================"

# 1. Set project
gcloud config set project "$PROJECT_ID"

# 2. Enable billing check
echo ""
echo "[1/7] Checking billing..."
BILLING=$(gcloud billing projects describe "$PROJECT_ID" --format="value(billingEnabled)" 2>/dev/null || echo "false")
if [[ "$BILLING" != "True" ]]; then
  echo "⚠  Billing not enabled on project $PROJECT_ID"
  echo "   Enable at: https://console.cloud.google.com/billing"
  exit 1
fi
echo "  ✓ Billing enabled"

# 3. Create Terraform state bucket (must exist before terraform init)
echo ""
echo "[2/7] Creating Terraform state bucket..."
TF_BUCKET="mixscope-terraform-state-${PROJECT_ID}"
gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://${TF_BUCKET}" 2>/dev/null || echo "  (bucket already exists)"
gsutil versioning set on "gs://${TF_BUCKET}"
# Update backend config
sed -i "s/mixscope-terraform-state/${TF_BUCKET}/g" infra/terraform/main.tf
echo "  ✓ State bucket: gs://${TF_BUCKET}"

# 4. Create initial secrets (empty — you fill values below)
echo ""
echo "[3/7] Creating Secret Manager secrets (empty)..."
for secret in \
  spotify-client-id \
  spotify-client-secret \
  youtube-api-key \
  soundcloud-client-id \
  soundcloud-client-secret \
  mixcloud-client-id \
  mixcloud-client-secret \
  db-password; do
  gcloud secrets create "mixscope-${secret}" \
    --project="$PROJECT_ID" \
    --replication-policy="automatic" 2>/dev/null || echo "  (mixscope-${secret} already exists)"
done
echo "  ✓ Secrets created"

# 5. Populate Spotify credentials (you have these ready)
echo ""
echo "[4/7] Populating Spotify credentials..."
echo -n "Enter Spotify Client ID: "
read -s SPOTIFY_CLIENT_ID
echo ""
echo -n "Enter Spotify Client Secret: "
read -s SPOTIFY_CLIENT_SECRET
echo ""

echo -n "$SPOTIFY_CLIENT_ID"  | gcloud secrets versions add "mixscope-spotify-client-id"  --data-file=- --project="$PROJECT_ID"
echo -n "$SPOTIFY_CLIENT_SECRET" | gcloud secrets versions add "mixscope-spotify-client-secret" --data-file=- --project="$PROJECT_ID"
echo "  ✓ Spotify credentials stored in Secret Manager"

# 6. Generate and store DB password
echo ""
echo "[5/7] Generating database password..."
DB_PASS=$(openssl rand -base64 32 | tr -d '=/+' | head -c 32)
echo -n "$DB_PASS" | gcloud secrets versions add "mixscope-db-password" --data-file=- --project="$PROJECT_ID"
echo "  ✓ DB password generated and stored"

# 7. Set project in terraform.tfvars
echo ""
echo "[6/7] Updating Terraform config..."
sed -i "s/YOUR_GCP_PROJECT_ID/${PROJECT_ID}/g" infra/terraform/terraform.tfvars
echo "  ✓ terraform.tfvars updated"

# 8. Init and apply Terraform
echo ""
echo "[7/7] Running Terraform..."
cd infra/terraform
terraform init \
  -backend-config="bucket=${TF_BUCKET}"
terraform plan -out=tfplan
echo ""
echo "Review the plan above. Press Enter to apply, Ctrl+C to abort."
read -r
terraform apply tfplan
cd ../..

echo ""
echo "========================================"
echo "✅ Bootstrap complete!"
echo ""
echo "Next steps:"
echo "  1. Add remaining API keys when ready:"
echo "     echo -n 'YOUR_KEY' | gcloud secrets versions add mixscope-youtube-api-key --data-file=-"
echo "     echo -n 'YOUR_KEY' | gcloud secrets versions add mixscope-soundcloud-client-id --data-file=-"
echo ""
echo "  2. Build and push the scraper Docker image:"
echo "     docker build -f infra/docker/Dockerfile.scraper -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/mixscope-scrapers/scraper:latest ."
echo "     docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/mixscope-scrapers/scraper:latest"
echo ""
echo "  3. Run your first scrape (Spotify):"
echo "     gcloud run jobs execute mixscope-scraper-spotify --region=${REGION}"
echo ""
echo "  4. Check results:"
echo "     gcloud run jobs describe mixscope-scraper-spotify --region=${REGION}"
echo "     gcloud logging read 'resource.type=cloud_run_job' --limit=50"
echo "========================================"
