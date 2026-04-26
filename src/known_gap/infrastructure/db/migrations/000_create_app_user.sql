-- Create the app role used by the FastAPI service to connect to Postgres.
--
-- Local dev (docker-compose): this runs as part of /docker-entrypoint-initdb.d/
-- on first DB init, while POSTGRES_USER=postgres is the superuser.
--
-- Cloud SQL: no-op — the role is created out of band via
--   `gcloud sql users create known_gap --instance=... --password=...`
-- so the IF NOT EXISTS branch is skipped.
--
-- The hardcoded password matches DATABASE_URL in docker-compose.yml. It is
-- only used for local development; Cloud SQL has its own password supplied
-- via Secret Manager.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'known_gap') THEN
    CREATE ROLE known_gap WITH LOGIN PASSWORD 'known_gap';
  END IF;
END $$;
