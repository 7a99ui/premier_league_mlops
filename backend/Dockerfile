# Backend - FastAPI
FROM python:3.11-slim

WORKDIR /app

COPY ./backend/src ./src
COPY ./data ./data
COPY ./models ./models

RUN pip install --no-cache-dir fastapi uvicorn pandas joblib scikit-learn

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
