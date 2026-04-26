-- Grant the app role (known_gap) the privileges it needs on schema objects
-- created by the migration superuser (postgres).
--
-- The 001_init.sql migration creates documents and chunks owned by postgres,
-- so the app role needs explicit GRANTs to read and write them. ALTER DEFAULT
-- PRIVILEGES extends the same grants to any future tables postgres creates,
-- so future migrations don't need a follow-up grant step.
--
-- Idempotent: re-granting the same privileges is a no-op.

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO known_gap;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO known_gap;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  GRANT ALL PRIVILEGES ON TABLES TO known_gap;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  GRANT ALL PRIVILEGES ON SEQUENCES TO known_gap;
