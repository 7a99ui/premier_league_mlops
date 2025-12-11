# 🎯 Guide de Gestion des Modèles

Ce document explique comment gérer les modèles dans le projet Premier League MLOps.

## 📁 Structure des Modèles

```
models/
├── production/                    # Modèles en production
│   ├── best_model_20231215_143052.joblib
│   ├── model_metadata_20231215_143052.json
│   ├── latest_model.joblib       # Symlink vers le dernier modèle
│   └── latest_metadata.json      # Métadonnées du dernier modèle
└── experiments/                   # Modèles expérimentaux (optionnel)
```

---

## 🚀 Entraînement et Sauvegarde

### 1. Via le Script (Production)

```bash
# Entraînement complet avec sauvegarde automatique
python src/models/train.py --phase all --top-n 3

# Le meilleur modèle sera automatiquement sauvegardé dans models/production/
```

**Avantages :**
- ✅ Automatique
- ✅ Reproductible
- ✅ Prêt pour CI/CD
- ✅ Versionnage automatique

### 2. Via le Notebook (Exploration)

```python
# Dans notebooks/04_model_training.ipynb
import joblib
from datetime import datetime

# Après avoir trouvé le meilleur modèle
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'../models/production/best_model_{timestamp}.joblib'

joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")
```

**Avantages :**
- ✅ Contrôle manuel
- ✅ Analyse détaillée
- ✅ Validation visuelle
- ✅ Documentation interactive

---

## 📊 Chargement des Modèles

### Méthode 1 : Charger le Dernier Modèle

```python
from src.models.utils import ModelLoader

# Charger le dernier modèle
loader = ModelLoader()
model, metadata = loader.load_latest_model()

# Utiliser le modèle
predictions = model.predict(X_new)
```

### Méthode 2 : Charger un Modèle Spécifique

```python
# Charger par timestamp
model, metadata = loader.load_model_by_timestamp('20231215_143052')
```

### Méthode 3 : Lister et Comparer

```python
# Lister tous les modèles
loader.list_available_models()

# Comparer les performances
df = loader.compare_models()
```

---

## 🎯 Faire des Prédictions

### Méthode Simple

```python
from src.models.utils import ModelPredictor

# Créer un predictor (charge automatiquement le dernier modèle)
predictor = ModelPredictor()

# Prédire
predictions = predictor.predict(X_test)

# Prédire le classement final
standings = predictor.predict_final_standings(features_df)
```

### Méthode avec Données Custom

```python
from src.models.utils import predict_from_latest_data

# Prédire à partir d'un fichier
predictions = predict_from_latest_data(
    data_path='data/processed/v1/test.parquet',
    output_path='predictions/standings_2024.csv'
)
```

---

## 📈 Tracking avec MLflow

### Visualiser les Expériences

```bash
# Lancer MLflow UI
mlflow ui --port 5000

# Ouvrir dans le navigateur
# http://localhost:5000
```

### Comparer les Runs

Dans l'UI MLflow :
1. Sélectionner plusieurs runs
2. Cliquer sur "Compare"
3. Visualiser les métriques côte à côte

### Filtrer les Runs

```python
import mlflow

# Rechercher les meilleurs modèles
runs = mlflow.search_runs(
    experiment_names=["premier-league-prediction"],
    filter_string="metrics.val_mae < 7.0",
    order_by=["metrics.val_mae ASC"]
)

print(runs[['run_id', 'metrics.val_mae', 'params.model_type']])
```

---

## 🔄 Workflow Recommandé

### Pour le Développement

1. **Expérimentation** (Notebook)
   ```
   notebooks/04_model_training.ipynb
   → Tester différentes approches
   → Analyser les résultats
   → Sauvegarder manuellement les modèles intéressants
   ```

2. **Validation** (Script)
   ```
   python src/models/train.py --phase all
   → Reproduire les meilleurs résultats
   → Sauvegarde automatique
   → Prêt pour production
   ```

### Pour la Production

1. **Entraînement Automatique**
   ```bash
   # Dans un pipeline CI/CD ou cron job
   python src/models/train.py --phase all --top-n 3
   ```

2. **Déploiement**
   ```python
   # L'application charge automatiquement le dernier modèle
   from src.models.utils import ModelPredictor
   predictor = ModelPredictor()
   ```

3. **Monitoring**
   ```bash
   # Vérifier les performances
   mlflow ui --port 5000
   ```

---

## 📋 Métadonnées des Modèles

Chaque modèle sauvegardé inclut :

```json
{
  "model_name": "ensemble_stacking",
  "timestamp": "20231215_143052",
  "metrics": {
    "val_mae": 6.75,
    "val_rmse": 8.92,
    "val_r2": 0.8745,
    "test_mae": 6.82,
    "test_rmse": 9.05,
    "test_r2": 0.8698
  },
  "model_file": "best_model_20231215_143052.joblib",
  "n_features": 35,
  "feature_names": ["current_points", "points_per_game", ...]
}
```

---

## 🚨 Best Practices

### ✅ DO

- **Toujours versionner** les modèles avec timestamp
- **Tracker avec MLflow** toutes les expériences
- **Sauvegarder les métadonnées** (features, métriques, hyperparamètres)
- **Tester sur un test set** avant déploiement
- **Documenter** les changements significatifs

### ❌ DON'T

- **Ne pas écraser** `latest_model.joblib` manuellement
- **Ne pas déployer** sans validation sur test set
- **Ne pas oublier** de sauvegarder les feature names
- **Ne pas ignorer** les warnings de compatibilité sklearn

---

## 🔧 Troubleshooting

### Le modèle ne se charge pas

```python
# Vérifier que le fichier existe
from pathlib import Path
model_path = Path('models/production/latest_model.joblib')
print(f"Exists: {model_path.exists()}")

# Vérifier la version de scikit-learn
import sklearn
print(f"sklearn version: {sklearn.__version__}")
```

### Features manquantes

```python
# Comparer les features
loader = ModelLoader()
_, metadata = loader.load_latest_model()
print(f"Expected features: {metadata['feature_names']}")
print(f"Got features: {X_new.columns.tolist()}")
```

### Prédictions incohérentes

```python
# Vérifier le scaling
from sklearn.preprocessing import StandardScaler
import joblib

scaler = joblib.load('data/processed/v1/scaler.joblib')
X_scaled = scaler.transform(X_new)
```

---

## 📚 Exemples d'Utilisation

### Exemple 1 : Prédire le Classement de la Saison en Cours

```python
from src.models.utils import ModelPredictor
import pandas as pd

# Charger les features de la saison actuelle
current_season = pd.read_parquet('data/processed/v1/current_season.parquet')

# Créer le predictor
predictor = ModelPredictor()

# Prédire le classement final
standings = predictor.predict_final_standings(current_season)

# Afficher
print(standings[['predicted_rank', 'team', 'predicted_final_points']])
```

### Exemple 2 : Comparer Plusieurs Modèles

```python
from src.models.utils import ModelLoader

loader = ModelLoader()

# Lister tous les modèles
models_df = loader.compare_models()

# Sélectionner le meilleur par test_mae
best_timestamp = models_df.loc[models_df['test_mae'].idxmin(), 'timestamp']

# Charger ce modèle
model, metadata = loader.load_model_by_timestamp(best_timestamp)
```

### Exemple 3 : Batch Predictions

```python
from src.models.utils import predict_from_latest_data
from pathlib import Path

# Prédire pour toutes les gameweeks d'une saison
season_data = Path('data/processed/v1/')

for gw in range(10, 39):
    gw_data = pd.read_parquet(season_data / f'gameweek_{gw}.parquet')
    predictions = predict_from_latest_data(
        gw_data,
        output_path=f'predictions/gw_{gw}_predictions.csv'
    )
```

---

## 🎯 Prochaines Étapes

Une fois que vous avez un modèle en production :

1. **Monitoring** : Suivre les performances en temps réel
2. **Retraining** : Réentraîner à la fin de chaque saison
3. **A/B Testing** : Tester de nouveaux modèles vs production
4. **API** : Créer une API pour servir les prédictions
5. **Dashboard** : Visualiser les prédictions et performances

---

## 📞 Support

- **Documentation** : Voir `README.md`
- **Issues** : Ouvrir une issue sur GitHub
- **MLflow** : `mlflow ui --port 5000`