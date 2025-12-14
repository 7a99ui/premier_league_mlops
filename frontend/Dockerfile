# Frontend - Streamlit
FROM python:3.11-slim

WORKDIR /app

# Copier uniquement le nécessaire dans le container
COPY ./frontend/app.py ./app.py
COPY ./data ./data
COPY ./models ./models

# Installer dépendances
RUN pip install --no-cache-dir streamlit pandas joblib scikit-learn

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
