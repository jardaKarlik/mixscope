"""
scrapers/utils/secrets.py
=========================
Thin wrapper around GCP Secret Manager.
All API keys are fetched from here at runtime — never from env vars or files.

Usage:
    from utils.secrets import get_secret
    client_id = get_secret("spotify-client-id")
"""

import os
from functools import lru_cache
from google.cloud import secretmanager


PROJECT_ID = os.environ.get("GCP_PROJECT")


@lru_cache(maxsize=32)
def get_secret(name: str, version: str = "latest") -> str:
    """
    Fetch a secret value from GCP Secret Manager.
    Secret names are prefixed with 'mixscope-' automatically.

    Args:
        name:    short name e.g. "spotify-client-id"
        version: secret version, default "latest"

    Returns:
        Secret value as string.

    Raises:
        ValueError if GCP_PROJECT env var not set.
        google.api_core.exceptions.NotFound if secret doesn't exist.
    """
    if not PROJECT_ID:
        raise ValueError(
            "GCP_PROJECT environment variable not set. "
            "Are you running inside a Cloud Run Job?"
        )

    client      = secretmanager.SecretManagerServiceClient()
    secret_path = f"projects/{PROJECT_ID}/secrets/mixscope-{name}/versions/{version}"

    response = client.access_secret_version(request={"name": secret_path})
    return response.payload.data.decode("utf-8").strip()


def get_db_password() -> str:
    return get_secret("db-password")

def get_spotify_credentials() -> tuple[str, str]:
    return get_secret("spotify-client-id"), get_secret("spotify-client-secret")

def get_youtube_api_key() -> str:
    return get_secret("youtube-api-key")

def get_soundcloud_credentials() -> tuple[str, str]:
    return get_secret("soundcloud-client-id"), get_secret("soundcloud-client-secret")

def get_mixcloud_credentials() -> tuple[str, str]:
    return get_secret("mixcloud-client-id"), get_secret("mixcloud-client-secret")
