# 🏆 Premier League Final Standings Prediction - MLOps Project

Projet MLOps complet pour prédire le classement final de la Premier League avec re-entraînement automatique et prédictions incrémentales.

## 📋 Description

Ce projet utilise les données historiques de la Premier League (2015-présent) pour prédire le classement final de chaque équipe. Le système est conçu pour :
- S'entraîner automatiquement à la fin de chaque saison
- Faire de nouvelles prédictions tous les 3 jours ou après nouveaux matchs
- Versionner les données et les modèles
- Valider automatiquement la qualité des données

## 🏗️ Architecture

```
Data Ingestion → Feature Engineering → Model Training → Predictions
       ↓                ↓                    ↓              ↓
    DVC (v1)      Validation (GX)      MLflow Tracking   Monitoring
```

## 🚀 Quick Start

### Installation

```bash
# Cloner le repo
git clone <your-repo-url>
cd premier-league-mlops

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer DVC
dvc init
dvc remote add -d storage <your-remote-storage>

# Copier et configurer les variables d'environnement
cp .env.template .env
# Éditer .env avec vos configurations
```

### Première utilisation

```bash
# 1. Scrapper les données historiques (une seule fois)
python src/data/ingestion.py --mode historical --seasons 2015-2016 2022-2023

# 2. Créer les features
python src/data/features.py --input data/raw --output data/processed/v1

# 3. Versionner les données
dvc add data/raw data/processed
git add data/raw.dvc data/processed.dvc
git commit -m "Add historical data v1"
dvc push

# 4. Explorer les données
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 📁 Structure du Projet

```
premier-league-mlops/
├── data/
│   ├── raw/                    # Données brutes (versionnées DVC)
│   ├── processed/              # Features engineerées (versionnées DVC)
│   └── predictions/            # Prédictions historiques
├── src/
│   ├── data/
│   │   ├── ingestion.py       # Scrapping des données
│   │   ├── validation.py      # Validation avec Great Expectations
│   │   └── features.py        # Feature engineering
│   ├── models/
│   │   ├── train.py          # Pipeline d'entraînement
│   │   └── predict.py        # Pipeline de prédiction
│   └── utils/
│       └── logger.py         # Configuration logging
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_baseline.ipynb
├── tests/                     # Tests unitaires
├── configs/                   # Fichiers de configuration
├── logs/                      # Logs d'exécution
├── .dvc/                      # Configuration DVC
├── requirements.txt
├── .env.template
└── README.md
```

## 🔄 Workflow

### Mode Development (Simulation)
```bash
# Tester avec une saison passée comme "nouvelle" saison
python src/data/ingestion.py --mode incremental --season 2023-2024 --simulate
```

### Mode Production
```bash
# Récupérer les nouveaux matchs
python src/data/ingestion.py --mode incremental --season 2024-2025

# Faire des prédictions
python src/models/predict.py --season 2024-2025 --model-version latest
```

## 📊 Données

### Sources
- API: `footballapi.pulselive.com`
- Saisons: 2015-2016 à 2024-2025
- Données disponibles:
  - Résultats des matchs
  - Statistiques détaillées (possession, tirs, etc.)
  - Classements gameweek par gameweek

### Versioning
- **DVC** pour versionner les données
- **MLflow** pour versionner les modèles
- **Git** pour versionner le code

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Tests avec couverture
pytest tests/ --cov=src --cov-report=html
```

## 📈 Monitoring

- MLflow UI: `mlflow ui --port 5000`
- Logs: `tail -f logs/app.log`

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📝 TODO

- [ ] Setup initial du projet
- [ ] Scrapping données historiques
- [ ] Feature engineering pipeline
- [ ] Data validation avec Great Expectations
- [ ] Modèle baseline
- [ ] Pipeline d'entraînement automatique
- [ ] Pipeline de prédiction incrémentale
- [ ] Monitoring et alerting
- [ ] Documentation API
- [ ] Tests unitaires complets

## 📄 License

MIT License

## 👥 Auteurs

Votre Nom - [votre-email@example.com]