FROM python:3.8-slim

USER root
RUN mkdir /app
COPY . /app/
WORKDIR /app/

# Upgrade pip
RUN python -m pip install --upgrade pip
# Install dependencies
RUN pip install -r requirements.txt

# Set Airflow configuration environment variables
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Allow script execution
RUN chmod 777 start.sh

# Install AWS CLI
RUN apt update -y && apt install awscli -y

# Set entrypoint to start Airflow
ENTRYPOINT [ "/bin/sh" ]
CMD ["start.sh"]