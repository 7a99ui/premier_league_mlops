# 🔍 Monitoring Complet - Premier League MLOps

## 📋 Vue d'Ensemble

Ce projet implémente un système de monitoring complet en 3 couches, inspiré des meilleures pratiques MLOps (comme le projet de Bassem Benhamed) :

```
┌──────────────────────────────────────────────────────────────┐
│                    MONITORING ARCHITECTURE                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. GREAT EXPECTATIONS                                        │
│     └─ Validation des données brutes                         │
│     └─ Schéma, types, valeurs nulles                         │
│     └─ Règles métier (20 équipes, points cohérents, etc.)    │
│                                                               │
│  2. DEEPCHECKS                                                │
│     └─ Intégrité des données (duplicates, types mixtes)      │
│     └─ Validation train/test (drift, tailles, corrélations)  │
│     └─ Évaluation du modèle (performance, calibration)       │
│                                                               │
│  3. EVIDENTLY                                                 │
│     └─ Détection de drift en production                      │
│     └─ Monitoring continu                                    │
│     └─ Décision de réentraînement                            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Structure des Fichiers

```
premier_league_mlops/
├── src/
│   ├── data/
│   │   └── validation.py              # Great Expectations
│   └── monitoring/
│       ├── drift_detection.py         # Evidently
│       ├── deepchecks_validator.py    # DeepChecks (NOUVEAU)
│       └── integrated_monitoring.py   # Pipeline intégré (NOUVEAU)
│
├── configs/
│   ├── drift_config.yaml              # Config Evidently
│   └── deepchecks_config.yaml         # Config DeepChecks (NOUVEAU)
│
├── reports/
│   ├── drift/                         # Rapports Evidently
│   ├── deepchecks/                    # Rapports DeepChecks (NOUVEAU)
│   └── integrated/                    # Résumés consolidés (NOUVEAU)
│
└── logs/
    ├── drift_detection.log
    ├── deepchecks_validation.log
    └── integrated_monitoring.log
```

---

## 🚀 Installation

### 1. Installer les dépendances

```bash
pip install great-expectations evidently deepchecks
```

### 2. Créer les dossiers

```bash
mkdir -p reports/deepchecks reports/integrated logs
```

---

## 📖 Guide d'Utilisation

### **WORKFLOW COMPLET**

```
┌─────────────────────────────────────────────────────────────┐
│                    ML PIPELINE WORKFLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. COLLECTE DONNÉES                                         │
│     ↓                                                        │
│  2. VALIDATION (Great Expectations) ← Vous êtes ici         │
│     └─ python src/data/validation.py --mode raw             │
│     ↓                                                        │
│  3. FEATURE ENGINEERING                                      │
│     ↓                                                        │
│  4. VALIDATION (Great Expectations)                          │
│     └─ python src/data/validation.py --mode features        │
│     ↓                                                        │
│  5. TRAIN/TEST SPLIT                                         │
│     ↓                                                        │
│  6. VALIDATION (DeepChecks) ← NOUVEAU                        │
│     └─ python src/monitoring/deepchecks_validator.py        │
│     ↓                                                        │
│  7. ENTRAÎNEMENT MODÈLE                                      │
│     ↓                                                        │
│  8. ÉVALUATION (DeepChecks) ← NOUVEAU                        │
│     └─ python src/monitoring/deepchecks_validator.py        │
│     ↓                                                        │
│  9. DÉPLOIEMENT                                              │
│     ↓                                                        │
│  10. MONITORING (Evidently)                                  │
│      └─ python src/monitoring/drift_detection.py            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Utilisation Détaillée

### **1. Great Expectations (Déjà Implémenté)**

#### Valider les données brutes
```bash
python src/data/validation.py --mode raw
```

#### Valider les features
```bash
python src/data/validation.py --mode features \
    --features-path data/processed/v1/features.parquet
```

#### Validation complète
```bash
python src/data/validation.py --mode all
```

**Résultats :**
- `data/validation_reports/raw_data_validation.json`
- `data/validation_reports/features_validation.json`

---

### **2. DeepChecks (NOUVEAU)**

#### Check d'intégrité des données
```bash
python src/monitoring/deepchecks_validator.py \
    --mode integrity \
    --data-dir data/processed/v1
```

#### Validation train/test
```bash
python src/monitoring/deepchecks_validator.py \
    --mode train-test \
    --data-dir data/processed/v1
```

#### Validation complète
```bash
python src/monitoring/deepchecks_validator.py \
    --mode full \
    --data-dir data/processed/v1
```

**Résultats :**
- `reports/deepchecks/data_integrity_*.html`
- `reports/deepchecks/train_test_validation_*.html`
- `reports/deepchecks/model_evaluation_*.html`

---

### **3. Evidently (Déjà Implémenté)**

#### Détecter le drift
```bash
python src/monitoring/drift_detection.py \
    --config configs/drift_config.yaml \
    --split train
```

**Résultats :**
- `reports/drift/drift_report_*.html`
- `reports/drift/drift_metrics_*.json`
- `reports/drift/latest_decision.json`

---

### **4. Pipeline Intégré (NOUVEAU)**

#### Pipeline pré-entraînement (recommandé)
```bash
python src/monitoring/integrated_monitoring.py --mode pre-training
```

Ce pipeline exécute :
1. ✅ Great Expectations sur données brutes
2. ✅ Great Expectations sur features
3. ✅ DeepChecks sur train/test

#### Monitoring production
```bash
python src/monitoring/integrated_monitoring.py --mode production
```

Ce pipeline exécute :
1. ✅ Evidently drift detection

#### Tout en une fois
```bash
python src/monitoring/integrated_monitoring.py --mode full
```

**Résultats :**
- `reports/integrated/pre_training_validation.json`
- `reports/integrated/production_monitoring.json`

---

## 🔧 Configuration

### **drift_config.yaml** (Evidently)
```yaml
paths:
  data_base: "data/processed"
  reports: "reports/drift"

versions:
  reference: "v1"
  current: "v2"

detection:
  drift_threshold: 0.3
  target_column: "target_final_points"
```

### **deepchecks_config.yaml** (DeepChecks)
```yaml
paths:
  data_dir: "data/processed/v1"
  reports_dir: "reports/deepchecks"

thresholds:
  feature_drift_threshold: 0.15
  label_drift_threshold: 0.10
  min_model_score: 0.70

checks:
  data_integrity:
    enabled: true
  train_test_validation:
    enabled: true
  model_evaluation:
    enabled: true
```

---

## 📊 Rapports Générés

### **Great Expectations**
- Format : JSON
- Contenu : Expectations passed/failed
- Localisation : `data/validation_reports/`

### **DeepChecks**
- Format : HTML interactif
- Contenu : Graphiques, métriques, conditions
- Localisation : `reports/deepchecks/`

### **Evidently**
- Format : HTML + JSON
- Contenu : Drift scores, distributions, décision
- Localisation : `reports/drift/`

---

## 🔄 Workflow CI/CD (À Venir)

### **Intégration GitHub Actions**

```yaml
# .github/workflows/data-validation.yml
name: Data Validation

on:
  push:
    paths:
      - 'data/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Great Expectations
        run: python src/data/validation.py --mode all
      
      - name: Run DeepChecks
        run: python src/monitoring/deepchecks_validator.py --mode full
      
      - name: Check if validation passed
        run: |
          python scripts/check_validation_status.py
```

### **Intégration Jenkins (comme Bassem)**

```groovy
pipeline {
    agent any
    
    stages {
        stage('Data Validation') {
            steps {
                sh 'python src/data/validation.py --mode all'
                sh 'python src/monitoring/deepchecks_validator.py --mode full'
            }
        }
        
        stage('Train Model') {
            when {
                expression { 
                    return validationPassed() 
                }
            }
            steps {
                sh 'python src/models/train.py'
            }
        }
        
        stage('Model Evaluation') {
            steps {
                sh 'python src/monitoring/deepchecks_validator.py --mode train-test'
            }
        }
        
        stage('Drift Detection') {
            steps {
                sh 'python src/monitoring/drift_detection.py'
            }
        }
    }
}
```

---

## 🎯 Comparaison avec le Projet Bassem

| Aspect | Projet Bassem | Votre Projet |
|--------|---------------|--------------|
| **Great Expectations** | ❓ Probablement | ✅ Implémenté |
| **DeepChecks** | ✅ Implémenté | ✅ NOUVEAU |
| **Evidently** | ✅ Implémenté | ✅ Implémenté |
| **Pipeline Intégré** | ✅ Via Jenkins | ✅ NOUVEAU |
| **Dashboard** | ✅ Streamlit | ❌ À faire |
| **CI/CD** | ✅ Jenkins | ❌ À faire |
| **Alertes** | ❓ Probablement | ❌ À faire |

---

## 📝 Prochaines Étapes

### **Priorité 1 : Tester DeepChecks**
```bash
# 1. Placer les fichiers dans votre projet
cp deepchecks_validator.py src/monitoring/
cp deepchecks_config.yaml configs/

# 2. Tester
python src/monitoring/deepchecks_validator.py --mode full
```

### **Priorité 2 : Dashboard Streamlit**
Créer un dashboard pour visualiser tous les rapports en temps réel.

### **Priorité 3 : CI/CD**
Intégrer dans GitHub Actions ou Jenkins.

### **Priorité 4 : Alertes**
Notifications Slack/Email quand drift détecté.

---

## 🤝 Contribution

Ce système de monitoring est maintenant au niveau des projets MLOps professionnels !

**Points forts :**
- ✅ Triple validation (GE + DeepChecks + Evidently)
- ✅ Pipeline intégré automatisé
- ✅ Configuration centralisée (YAML)
- ✅ Logging complet
- ✅ Rapports HTML interactifs

**À améliorer :**
- Dashboard de visualisation
- CI/CD automatique
- Système d'alertes
- Historique des métriques

---

## 📚 Ressources

- [Great Expectations Docs](https://docs.greatexpectations.io/)
- [DeepChecks Docs](https://docs.deepchecks.com/)
- [Evidently Docs](https://docs.evidentlyai.com/)
- [Projet Bassem](https://github.com/bassambhamed/mlops_fraud)

---

## ✅ Checklist Finale

- [x] Great Expectations - Validation données brutes
- [x] Great Expectations - Validation features
- [x] DeepChecks - Data integrity
- [x] DeepChecks - Train-test validation
- [x] DeepChecks - Model evaluation
- [x] Evidently - Drift detection
- [x] Pipeline intégré
- [x] Configuration YAML
- [x] Logging
- [ ] Dashboard Streamlit
- [ ] CI/CD (GitHub Actions ou Jenkins)
- [ ] Alertes (Slack/Email)
- [ ] Tests unitaires