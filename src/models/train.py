"""
Training Pipeline with MLflow Tracking

Phases:
1. Baseline: Train models with default parameters
2. Fine-Tuning: Hyperparameter optimization
3. Stacking: Ensemble methods
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
import yaml
import joblib
from datetime import datetime
import argparse
import json

# ML Libraries
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# Utils
import warnings
warnings.filterwarnings('ignore')


class MLflowTracker:
    """Gestionnaire MLflow pour le tracking des expériences"""
    
    def __init__(self, config_path='configs/mlflow_config.yaml'):
        """Initialise MLflow avec la configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Configuration de MLflow
        mlflow.set_tracking_uri(self.config['tracking_uri'])
        mlflow.set_experiment(self.config['experiment_name'])
        
        # Auto-logging
        if self.config.get('autolog', True):
            mlflow.sklearn.autolog(log_models=False)  # Disable automatic model logging to avoid conflicts
        
        print(f"✅ MLflow configured:")
        print(f"   Tracking URI: {self.config['tracking_uri']}")
        print(f"   Experiment: {self.config['experiment_name']}")
    
    def start_run(self, run_name, tags=None):
        """Démarre une run MLflow"""
        return mlflow.start_run(run_name=run_name, tags=tags or {})
    
    def log_metrics(self, metrics):
        """Log les métriques"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
    def log_params(self, params):
        """Log les paramètres (with conflict handling)"""
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except mlflow.exceptions.MlflowException as e:
                if "Changing param values is not allowed" in str(e):
                    # Parameter already logged by autolog, skip
                    continue
                else:
                    raise
    
    def log_model(self, model, artifact_path="model"):
        """Log le modèle"""
        try:
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        except:
            # Si le logging échoue, continuer quand même
            pass


class BaselineTrainer:
    """Phase 1: Entraînement des modèles baseline"""
    
    def __init__(self, mlflow_tracker):
        self.tracker = mlflow_tracker
        self.models = {}
        self.results = {}
    
    def get_baseline_models(self):
        """Définit les modèles baseline avec paramètres par défaut"""
        return {
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            ),
            'xgboost': XGBRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }
    
    def train_all_baselines(self, X_train, y_train, X_val, y_val):
        """Entraîne tous les modèles baseline"""
        print(f"\n{'='*70}")
        print("PHASE 1: BASELINE MODELS")
        print(f"{'='*70}\n")
        
        models = self.get_baseline_models()
        
        for model_name, model in models.items():
            print(f"🔄 Training {model_name}...")
            
            # Start MLflow run
            with self.tracker.start_run(
                run_name=f"baseline_{model_name}",
                tags={"phase": "baseline", "model_type": model_name}
            ):
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Evaluate
                metrics = self._evaluate_model(
                    y_train, y_train_pred,
                    y_val, y_val_pred
                )
                
                # Log parameters
                self.tracker.log_params({
                    'model_type': model_name,
                    'phase': 'baseline',
                    'n_features': X_train.shape[1]
                })
                
                # Log metrics
                self.tracker.log_metrics(metrics)
                
                # Log model
                self.tracker.log_model(model)
                
                # Store results
                self.models[model_name] = model
                self.results[model_name] = metrics
                
                print(f"   ✓ Val MAE: {metrics['val_mae']:.2f}, Val R²: {metrics['val_r2']:.4f}")
        
        # Summary
        self._print_baseline_summary()
        
        return self.models, self.results
    
    def _evaluate_model(self, y_train, y_train_pred, y_val, y_val_pred):
        """Calcule les métriques d'évaluation"""
        return {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'val_r2': r2_score(y_val, y_val_pred)
        }
    
    def _print_baseline_summary(self):
        """Affiche un résumé des performances baseline"""
        print(f"\n{'='*70}")
        print("BASELINE RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        # Trier par val_mae
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['val_mae']
        )
        
        print(f"{'Model':<20} {'Val MAE':<12} {'Val RMSE':<12} {'Val R²':<10}")
        print(f"{'-'*70}")
        for model_name, metrics in sorted_results:
            print(f"{model_name:<20} {metrics['val_mae']:<12.2f} "
                  f"{metrics['val_rmse']:<12.2f} {metrics['val_r2']:<10.4f}")
        
        best_model = sorted_results[0][0]
        print(f"\n🏆 Best baseline model: {best_model}")


class FineTuner:
    """Phase 2: Fine-tuning avec hyperparameter optimization"""
    
    def __init__(self, mlflow_tracker, custom_hyperparameters=None):
        self.tracker = mlflow_tracker
        self.best_models = {}
        self.best_params = {}
        self.custom_hyperparameters = custom_hyperparameters or {}
    
    def get_param_distributions(self):
        """Définit les distributions de paramètres pour chaque modèle"""
        default_params = {
            'random_forest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [15, 31, 63, 127],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            },
            'ridge': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'elastic_net': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        }
        
        default_params.update(self.custom_hyperparameters)
        return default_params
    
    def fine_tune_model(self, model_name, base_model, X_train, y_train, 
                       X_val, y_val, n_iter=20, cv=3):
        """Fine-tune un modèle spécifique"""
        print(f"\n🔧 Fine-tuning {model_name}...")
        
        param_dist = self.get_param_distributions().get(model_name)
        
        if param_dist is None:
            print(f"   ⚠️  No hyperparameters defined for {model_name}, skipping...")
            return None
        
        with self.tracker.start_run(
            run_name=f"finetuned_{model_name}",
            tags={"phase": "fine_tuning", "model_type": model_name}
        ):
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_absolute_error',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            y_train_pred = best_model.predict(X_train)
            y_val_pred = best_model.predict(X_val)
            
            metrics = {
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'train_r2': r2_score(y_train, y_train_pred),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'val_r2': r2_score(y_val, y_val_pred),
                'cv_score': -search.best_score_
            }
            
            self.tracker.log_params({
                'model_type': model_name,
                'phase': 'fine_tuning',
                'n_iter': n_iter,
                'cv_folds': cv,
                **search.best_params_
            })
            self.tracker.log_metrics(metrics)
            self.tracker.log_model(best_model)
            
            self.best_models[model_name] = best_model
            self.best_params[model_name] = search.best_params_
            
            print(f"   ✓ Best Val MAE: {metrics['val_mae']:.2f}")
            print(f"   ✓ Best params: {search.best_params_}")
            
            return best_model, metrics
    
    def fine_tune_top_models(self, baseline_results, baseline_models,
                            X_train, y_train, X_val, y_val, top_n=3):
        """Fine-tune les N meilleurs modèles baseline"""
        print(f"\n{'='*70}")
        print("PHASE 2: FINE-TUNING TOP MODELS")
        print(f"{'='*70}\n")
        
        sorted_models = sorted(
            baseline_results.items(),
            key=lambda x: x[1]['val_mae']
        )[:top_n]
        
        top_model_names = [name for name, _ in sorted_models]
        print(f"📊 Fine-tuning top {top_n} models: {top_model_names}\n")
        
        results = {}
        for model_name in top_model_names:
            base_model = baseline_models[model_name]
            result = self.fine_tune_model(
                model_name, base_model, X_train, y_train, X_val, y_val
            )
            if result:
                results[model_name] = result
        
        return results


class EnsembleBuilder:
    """Phase 3: Création d'ensembles (Stacking + Voting)"""
    
    def __init__(self, mlflow_tracker):
        self.tracker = mlflow_tracker
        self.ensemble_models = {}
    
    def create_stacking_ensemble(self, base_models, X_train, y_train, X_val, y_val):
        """Crée un modèle de stacking"""
        print(f"\n🎯 Creating Stacking Ensemble...")
        
        estimators = [(name, model) for name, model in base_models.items()]
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1
        )
        
        with self.tracker.start_run(
            run_name="stacking_ensemble",
            tags={"phase": "ensemble", "ensemble_type": "stacking"}
        ):
            stacking.fit(X_train, y_train)
            
            y_train_pred = stacking.predict(X_train)
            y_val_pred = stacking.predict(X_val)
            
            metrics = {
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'train_r2': r2_score(y_train, y_train_pred),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'val_r2': r2_score(y_val, y_val_pred)
            }
            
            self.tracker.log_params({
                'ensemble_type': 'stacking',
                'n_base_models': len(base_models),
                'base_model_names': ','.join(base_models.keys()),
                'meta_learner': 'Ridge'
            })
            
            self.tracker.log_metrics(metrics)
            self.tracker.log_model(stacking)
            
            print(f"   ✓ Stacking Val MAE: {metrics['val_mae']:.2f}")
            
            self.ensemble_models['stacking'] = stacking
            return stacking, metrics
    
    def create_voting_ensemble(self, base_models, X_train, y_train, X_val, y_val):
        """Crée un modèle de voting"""
        print(f"\n🎯 Creating Voting Ensemble...")
        
        estimators = [(name, model) for name, model in base_models.items()]
        voting = VotingRegressor(estimators=estimators, n_jobs=-1)
        
        with self.tracker.start_run(
            run_name="voting_ensemble",
            tags={"phase": "ensemble", "ensemble_type": "voting"}
        ):
            voting.fit(X_train, y_train)
            
            y_train_pred = voting.predict(X_train)
            y_val_pred = voting.predict(X_val)
            
            metrics = {
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'train_r2': r2_score(y_train, y_train_pred),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'val_r2': r2_score(y_val, y_val_pred)
            }
            
            self.tracker.log_params({
                'ensemble_type': 'voting',
                'n_base_models': len(base_models),
                'base_model_names': ','.join(base_models.keys())
            })
            
            self.tracker.log_metrics(metrics)
            self.tracker.log_model(voting)
            
            print(f"   ✓ Voting Val MAE: {metrics['val_mae']:.2f}")
            
            self.ensemble_models['voting'] = voting
            return voting, metrics
    
    def build_ensembles(self, base_models, X_train, y_train, X_val, y_val):
        """Construit tous les ensembles"""
        print(f"\n{'='*70}")
        print("PHASE 3: ENSEMBLE METHODS")
        print(f"{'='*70}\n")
        
        results = {}
        
        stacking, stack_metrics = self.create_stacking_ensemble(
            base_models, X_train, y_train, X_val, y_val
        )
        results['stacking'] = stack_metrics
        
        voting, vote_metrics = self.create_voting_ensemble(
            base_models, X_train, y_train, X_val, y_val
        )
        results['voting'] = vote_metrics
        
        return results


class ModelSelector:
    """Sélectionne le meilleur modèle global"""
    
    def __init__(self, mlflow_tracker):
        self.tracker = mlflow_tracker
    
    def select_best_model(self, all_results, all_models):
        """Sélectionne le meilleur modèle parmi tous"""
        print(f"\n{'='*70}")
        print("FINAL MODEL SELECTION")
        print(f"{'='*70}\n")
        
        combined = {}
        for phase_name, phase_results in all_results.items():
            if isinstance(phase_results, dict):
                for model_name, metrics in phase_results.items():
                    key = f"{phase_name}_{model_name}"
                    combined[key] = metrics
        
        sorted_models = sorted(combined.items(), key=lambda x: x[1]['val_mae'])
        
        print(f"{'Model':<30} {'Val MAE':<12} {'Val RMSE':<12} {'Val R²':<10}")
        print(f"{'-'*70}")
        for model_key, metrics in sorted_models[:10]:
            print(f"{model_key:<30} {metrics['val_mae']:<12.2f} "
                  f"{metrics['val_rmse']:<12.2f} {metrics['val_r2']:<10.4f}")
        
        best_model_key = sorted_models[0][0]
        best_metrics = sorted_models[0][1]
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_model_key}")
        print(f"{'='*70}")
        print(f"   Val MAE:  {best_metrics['val_mae']:.2f} points")
        print(f"   Val RMSE: {best_metrics['val_rmse']:.2f} points")
        print(f"   Val R²:   {best_metrics['val_r2']:.4f}")
        
        return best_model_key, best_metrics


def load_data(data_dir='data/processed/v1'):
    """Charge les datasets train/val/test"""
    train = pd.read_parquet(Path(data_dir) / 'train.parquet')
    val = pd.read_parquet(Path(data_dir) / 'val.parquet')
    test = pd.read_parquet(Path(data_dir) / 'test.parquet')
    
    metadata_cols = ['season', 'team', 'gameweek']
    target_col = 'target_final_points'
    
    feature_cols = [col for col in train.columns 
                   if col not in metadata_cols + [target_col]]
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def _get_best_model(best_model_key, all_models, ensemble_builder):
    """Récupère le meilleur modèle à partir de sa clé"""
    if 'ensemble' in best_model_key:
        if ensemble_builder and hasattr(ensemble_builder, 'ensemble_models'):
            ensemble_type = best_model_key.split('_')[-1]
            return ensemble_builder.ensemble_models.get(ensemble_type)
    else:
        parts = best_model_key.split('_', 1)
        model_name = parts[1] if len(parts) > 1 else parts[0]
        return all_models.get(model_name)
    return None


def save_best_model(model, model_name, metrics, X_test, y_test, 
                   output_dir='models/production'):
    """Sauvegarde le meilleur modèle avec ses métadonnées"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_metrics = {
        'test_mae': float(mean_absolute_error(y_test, y_test_pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        'test_r2': float(r2_score(y_test, y_test_pred))
    }
    
    all_metrics = {**metrics, **test_metrics}
    
    # Save model
    model_filename = f'best_model_{timestamp}.joblib'
    model_path = output_path / model_filename
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'metrics': all_metrics,
        'model_file': model_filename,
        'n_features': X_test.shape[1],
        'feature_names': X_test.columns.tolist() if hasattr(X_test, 'columns') else []
    }
    
    metadata_path = output_path / f'model_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save as "latest" for easy access
    latest_model_path = output_path / 'latest_model.joblib'
    latest_metadata_path = output_path / 'latest_metadata.json'
    
    joblib.dump(model, latest_model_path)
    with open(latest_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print("💾 MODEL SAVED")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Latest: {latest_model_path}")
    print(f"\n📊 Test Set Performance:")
    print(f"   MAE:  {test_metrics['test_mae']:.2f} points")
    print(f"   RMSE: {test_metrics['test_rmse']:.2f} points")
    print(f"   R²:   {test_metrics['test_r2']:.4f}")
    print(f"{'='*70}")
    
    return model_path


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Full Training Pipeline')
    parser.add_argument('--data-dir', default='data/processed/v1',
                       help='Directory with train/val/test data')
    parser.add_argument('--phase', choices=['baseline', 'finetune', 'ensemble', 'all'],
                       default='all', help='Which phase to run')
    parser.add_argument('--top-n', type=int, default=3,
                       help='Number of top models to fine-tune')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("PREMIER LEAGUE PREDICTION - TRAINING PIPELINE")
    print(f"{'='*70}\n")
    
    # Load data
    print("📥 Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_dir)
    print(f"   Train: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Initialize MLflow
    tracker = MLflowTracker()
    
    all_results = {}
    all_models = {}
    
    # Phase 1: Baseline
    if args.phase in ['baseline', 'all']:
        baseline_trainer = BaselineTrainer(tracker)
        baseline_models, baseline_results = baseline_trainer.train_all_baselines(
            X_train, y_train, X_val, y_val
        )
        all_results['baseline'] = baseline_results
        all_models.update(baseline_models)
    
    # Phase 2: Fine-tuning
    if args.phase in ['finetune', 'all']:
        if 'baseline' not in all_results:
            print("⚠️  Need baseline results for fine-tuning. Run baseline first.")
        else:
            custom_hyperparams = {
                'elastic_net': {'alpha': [0.1, 0.5, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
                'lasso': {'alpha': [0.01, 0.1, 1.0]},
                'ridge': {'alpha': [0.01, 0.1, 1.0]}
            }
            
            finetuner = FineTuner(tracker, custom_hyperparameters=custom_hyperparams)
            finetuned_results = finetuner.fine_tune_top_models(
                baseline_results, baseline_models,
                X_train, y_train, X_val, y_val, top_n=args.top_n
            )
            all_results['finetuned'] = {k: v[1] for k, v in finetuned_results.items()}
            all_models.update(finetuner.best_models)
    
    # Phase 3: Ensemble
    ensemble_builder = None
    if args.phase in ['ensemble', 'all']:
        if len(all_models) < 2:
            print("⚠️  Need at least 2 models for ensemble. Run baseline first.")
        else:
            top_models = dict(sorted(
                all_models.items(),
                key=lambda x: baseline_results.get(x[0], {'val_mae': float('inf')})['val_mae']
            )[:3])
            
            ensemble_builder = EnsembleBuilder(tracker)
            ensemble_results = ensemble_builder.build_ensembles(
                top_models, X_train, y_train, X_val, y_val
            )
            all_results['ensemble'] = ensemble_results
            all_models.update(ensemble_builder.ensemble_models)
    
    # Final model selection and saving
    if args.phase == 'all':
        selector = ModelSelector(tracker)
        best_model_key, best_metrics = selector.select_best_model(
            all_results, all_models
        )
        
        print(f"\n💾 Saving best model...")
        best_model = _get_best_model(best_model_key, all_models, ensemble_builder)
        
        if best_model:
            model_path = save_best_model(
                model=best_model,
                model_name=best_model_key,
                metrics=best_metrics,
                X_test=X_test,
                y_test=y_test
            )
        else:
            print("⚠️  Could not retrieve best model for saving")
    
    print(f"\n✅ Training pipeline complete!")
    print(f"   View results: mlflow ui --port 5000")


if __name__ == '__main__':
    main()