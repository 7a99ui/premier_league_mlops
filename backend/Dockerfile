FROM python:3.11-slim

WORKDIR /app

# ÉTAPE 1: Copier SEULEMENT requirements.txt
COPY ./backend/requirements.txt .

# ÉTAPE 2: Installer les dépendances (cette couche sera mise en cache!)
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# ÉTAPE 3: Copier le code source (en dernier)
COPY ./backend/src ./src

# ÉTAPE 4: Copier les données et modèles (ces fichiers changent souvent)
COPY ./data ./data
COPY ./models ./models

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]