FROM python:3.11-slim

WORKDIR /app

# ✅ Installer git, libgomp1 pour LightGBM
RUN apt-get update && apt-get install -y \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ✅ Copier et installer les requirements
COPY ./frontend/requirements.txt .
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# ✅ Copier uniquement l'app (pas les données)
COPY ./frontend/app.py ./app.py

# ✅ Configuration MLflow / DagsHub
ENV MLFLOW_TRACKING_URI=https://dagshub.com/7a99ui/premier_league_mlops.mlflow
ENV MLFLOW_TRACKING_USERNAME=7a99ui
ENV MLFLOW_TRACKING_PASSWORD=a694eb64959fe496eb2edb4bce9a1ed71391c4bc

# ✅ Variables pour Git et DVC
ENV REPO_URL=https://github.com/7a99ui/premier_league_mlops.git
ENV GIT_BRANCH=main

# ✅ Script de démarrage
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🔄 Cloning repository..."\n\
git clone --depth 1 --branch $GIT_BRANCH $REPO_URL /tmp/repo\n\
\n\
echo "📂 Copying data structure and git metadata..."\n\
cp -r /tmp/repo/data /app/\n\
cp -r /tmp/repo/.dvc /app/\n\
cp -r /tmp/repo/.git /app/\n\
\n\
echo "🔧 Configuring DVC..."\n\
cd /app\n\
dvc remote modify origin --local access_key_id $MLFLOW_TRACKING_PASSWORD\n\
dvc remote modify origin --local secret_access_key $MLFLOW_TRACKING_PASSWORD\n\
\n\
echo "⬇️  Pulling latest data from DVC..."\n\
dvc pull --force data/processed\n\
\n\
echo "🚀 Starting Streamlit..."\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8501

CMD ["/app/start.sh"]