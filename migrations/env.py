"""
Alembic environment config.

DB password is fetched from GCP Secret Manager via the gcloud CLI so this
works on any machine with gcloud authenticated — no hardcoded credentials.

To run migrations:
  1. Start Cloud SQL Auth Proxy on localhost:5432
     cloud-sql-proxy --port 5432 <YOUR_CONNECTION_NAME>
  2. Run: alembic upgrade head
"""

import os
import subprocess
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _get_db_password() -> str:
    """Fetch DB password via gcloud CLI (works wherever gcloud is authenticated)."""
    result = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest",
         "--secret=mixscope-db-password", "--project=mixsource"],
        capture_output=True, text=True,
        shell=(os.name == "nt"),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not fetch DB password: {result.stderr.strip()}")
    return result.stdout.strip()


def get_url() -> str:
    db_name = os.environ.get("DB_NAME", "mixscope")
    db_user = os.environ.get("DB_USER", "scraper")
    db_host = os.environ.get("DB_HOST", "127.0.0.1")
    db_port = os.environ.get("DB_PORT", "5432")
    db_pass = _get_db_password()
    return f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
