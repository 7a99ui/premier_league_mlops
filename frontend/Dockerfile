FROM python:3.11-slim

WORKDIR /app

# ✅ Installer libgomp1 pour LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ./frontend/requirements.txt .
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

COPY ./frontend/app.py ./app.py
COPY ./data ./data
COPY ./models ./models


# ✅ Configuration MLflow / DagsHub
ENV MLFLOW_TRACKING_URI=https://dagshub.com/7a99ui/premier_league_mlops.mlflow
ENV MLFLOW_TRACKING_USERNAME=7a99ui
ENV MLFLOW_TRACKING_PASSWORD=a694eb64959fe496eb2edb4bce9a1ed71391c4bc


# ✅ Configuration DVC (pour récupérer les dernières données au démarrage)
# Initialize DVC project (no SCM needed in container)
RUN dvc init --no-scm

# Configure DVC to use DagsHub
RUN dvc remote add -d origin s3://dvc
RUN dvc remote modify origin endpointurl https://dagshub.com/7a99ui/premier_league_mlops.s3

# Utiliser le token pour l'accès DVC via les variables d'environnement
# NOTE: DAGSHUB_USER_TOKEN doit être passé au runtime ou hérité de MLFLOW_TRACKING_PASSWORD si c'est le même token
CMD dvc remote modify origin --local access_key_id $MLFLOW_TRACKING_PASSWORD && \
    dvc remote modify origin --local secret_access_key $MLFLOW_TRACKING_PASSWORD && \
    dvc pull --force data/processed/test.parquet && \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0