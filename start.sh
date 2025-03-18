#!/bin/sh
set -e  # Exit immediately if a command exits with a non-zero status

# Initialize DB only if not already initialized
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
  echo "Initializing Airflow DB..."
  airflow db init
fi

# Check if user exists
USER_EXISTS=$(airflow users list | grep "${AIRFLOW_USERNAME}" || true)

if [ -z "$USER_EXISTS" ]; then
  echo "User does not exist. Creating Airflow admin user..."
  airflow users create \
      --email "${AIRFLOW_EMAIL}" \
      --first "Rajat" \
      --last "Singh" \
      --password "${AIRFLOW_PASSWORD}" \
      --role "Admin" \
      --username "${AIRFLOW_USERNAME}"
else
  echo "User ${AIRFLOW_USERNAME} already exists. Skipping creation."
fi

# Start scheduler in background
echo "Starting Airflow Scheduler..."
nohup airflow scheduler > /dev/null 2>&1 &

# Optional small delay to allow scheduler startup
sleep 5

echo "Starting Airflow Webserver..."
exec airflow webserver
