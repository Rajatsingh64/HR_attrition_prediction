# Human Resources Machine Learning Project: Predicting Employee Attrition

## Objective
To predict whether an employee will leave the company (attrition) based on various features such as age, job satisfaction, salary, etc.

## Author
Rajat Singh at Unified Mentor

## Project Structure

### Main Project Folder
- **src**: Contains the core components of the project.
  
  - **component**:
    - **data_ingestion.py**: For loading data.
    - **data_validation.py**: To ensure data integrity.
    - **data_transformation.py**: For data preprocessing.
    - **model_training.py**: To train machine learning models.
    - **model_evaluation.py**: To assess model performance.
    - **model_pusher.py**: For deploying models.
  
  - **entity**: 
    - **artifact_entity.py**: For managing output artifacts.
    - **config_entity.py**: For handling input configurations.
  
  - **pipeline**:
    - **training_pipeline**: For training ML models.
    - **batch_prediction_pipeline**: For making predictions.

  - **config.py**: For loading environment variables.
  - **data_dump.py**: For dumping data into MongoDB.
  - **utils.py**: For utility functions used across components.
  - **predictor.py**: For making predictions using the trained model


### Additional Files
- **research.ipynb**: Jupyter notebook for exploratory data analysis (EDA).
- **.env**: File for sensitive information like passwords and MongoDB URLs.

**Docker activatein awscli:**
  - curl -fsSL https://get.docker.com -o get-docker.sh
  - sudo sh get-docker.sh
  - sudo usermod -aG docker ubuntu
  - newgrp docker 

**Github actions Secrets:**
  - AWS_ACCESS_KEY_ID=
  - AWS_SECRET_ACCESS_KEY=
  - AWS_REGION=
  - AWS_ECR_LOGIN_URI=
  - ECR_REPOSITORY_NAME=
  - BUCKET_NAME=
  - MONGO_DB_URL=  