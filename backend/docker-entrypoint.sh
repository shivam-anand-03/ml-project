#!/usr/bin/env sh
# Wait for the database, apply migrations, then start the API.
set -e

if echo "$DATABASE_URL" | grep -q "^postgresql"; then
  echo "Waiting for PostgreSQL ..."
  python - <<'PY'
import os, time, sys
from sqlalchemy import create_engine, text
url = os.environ["DATABASE_URL"]
for i in range(60):
    try:
        create_engine(url).connect().execute(text("SELECT 1"))
        print("PostgreSQL is ready")
        sys.exit(0)
    except Exception as exc:
        time.sleep(2)
print("PostgreSQL did not become ready in time"); sys.exit(1)
PY
fi

echo "Applying database migrations ..."
alembic upgrade head

exec "$@"
