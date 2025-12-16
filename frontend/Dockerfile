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


EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]