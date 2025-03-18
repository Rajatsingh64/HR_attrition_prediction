#!/bin/sh
set -e  # Exit immediately if a command fails

echo "Starting Airflow setup..."

# --------------------------
# 1️⃣ Sync models from S3 (if BUCKET_NAME is set)
# --------------------------
if [ -n "$BUCKET_NAME" ]; then
  echo "Syncing saved models from S3 bucket: $BUCKET_NAME"
  mkdir -p /app/saved_models
  aws s3 sync s3://"$BUCKET_NAME"/saved_models /app/saved_models
  echo "S3 sync complete."
else
  echo "BUCKET_NAME not set, skipping S3 sync."
fi

# --------------------------
# 2️⃣ Initialize Airflow DB
# --------------------------
echo "Initializing Airflow database..."
airflow db upgrade  # Upgrade safer than init (works even if DB exists)

# --------------------------
# 3️⃣ Create Admin User (if not exists)
# --------------------------
echo "Checking if user ${AIRFLOW_USERNAME} exists..."
if airflow users list | grep -w "${AIRFLOW_USERNAME}" > /dev/null; then
  echo "User ${AIRFLOW_USERNAME} already exists."
else
  echo "Creating admin user: ${AIRFLOW_USERNAME}"
  airflow users create \
      --username "${AIRFLOW_USERNAME}" \
      --firstname "Rajat" \
      --lastname "Singh" \
      --role Admin \
      --email "${AIRFLOW_EMAIL}" \
      --password "${AIRFLOW_PASSWORD}"
fi

# --------------------------
# 4️⃣ Start Scheduler & Webserver
# --------------------------
echo "Starting Airflow Scheduler..."
airflow scheduler &

# Optional: Short wait to ensure scheduler starts
sleep 5

echo "Starting Airflow Webserver..."
exec airflow webserver
