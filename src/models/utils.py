"""
Utility functions for model management with MLflow Registry support
Supports both local file-based models and MLflow Registry models
"""

import joblib
import json
from pathlib import Path
from datetime import datetime
import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings('ignore')


class ModelLoader:
    """Charge et gère les modèles sauvegardés (local + MLflow Registry)"""
    
    def __init__(self, models_dir='models/production', use_mlflow=True, mlflow_config_path='configs/mlflow_config.yaml'):
        """
        Args:
            models_dir: Dossier des modèles locaux
            use_mlflow: Si True, privilégie MLflow Registry
            mlflow_config_path: Chemin vers la config MLflow
        """
        self.use_mlflow = use_mlflow
        self.mlflow_client = None
        
        # Setup MLflow si demandé
        if self.use_mlflow:
            self._setup_mlflow(mlflow_config_path)
        
        # Setup local models directory
        base_path = Path.cwd()
        
        possible_paths = [
            base_path / models_dir,
            base_path / '..' / models_dir,
            base_path / 'notebooks' / '..' / models_dir,
            Path(models_dir).resolve(),
        ]
        
        for path in possible_paths:
            path = path.resolve()
            if path.exists():
                self.models_dir = path
                print(f"✅ Local models directory found: {self.models_dir}")
                break
        else:
            self.models_dir = base_path / models_dir
            self.models_dir.mkdir(parents=True, exist_ok=True)
            print(f"⚠️  Created local models directory: {self.models_dir}")
    
    def _setup_mlflow(self, config_path):
        """Configure MLflow"""
        try:
            import yaml
            config_path = Path(config_path)
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                mlflow.set_tracking_uri(config['tracking_uri'])
            else:
                print(f"⚠️  MLflow config not found at {config_path}, using default tracking URI")
            
            self.mlflow_client = MlflowClient()
            print(f"✅ MLflow Registry enabled")
        except Exception as e:
            print(f"⚠️  Could not setup MLflow: {e}")
            print(f"   Falling back to local model loading only")
            self.use_mlflow = False
    
    def load_latest_model(self, prefer_mlflow=True):
        """
        Charge le dernier modèle (MLflow Registry ou local)
        
        Args:
            prefer_mlflow: Si True, charge depuis MLflow Registry si disponible
        
        Returns:
            tuple: (model, metadata)
        """
        # Try MLflow Registry first if enabled
        if self.use_mlflow and prefer_mlflow:
            try:
                return self.load_from_registry(stage="Production")
            except Exception as e:
                print(f"⚠️  Could not load from MLflow Registry: {e}")
                print(f"   Falling back to local model...")
        
        # Fallback to local model
        return self._load_latest_local_model()
    
    def load_from_registry(self, stage="Production", model_name="PremierLeagueModel"):
        """
        Charge un modèle depuis MLflow Registry
        
        Args:
            stage: Stage du modèle (Production, Staging, Archived)
            model_name: Nom du modèle dans le Registry
        
        Returns:
            tuple: (model, metadata)
        """
        if not self.use_mlflow:
            raise ValueError("MLflow is not enabled. Set use_mlflow=True in constructor.")
        
        print(f"🔍 Loading model from MLflow Registry...")
        print(f"   Model name: {model_name}")
        print(f"   Stage: {stage}")
        
        # Load model from registry
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get model metadata
        model_versions = self.mlflow_client.get_latest_versions(model_name, stages=[stage])
        
        if not model_versions:
            raise ValueError(f"No model found in stage '{stage}' for {model_name}")
        
        model_version = model_versions[0]
        run = self.mlflow_client.get_run(model_version.run_id)
        
        # Try to load feature names from artifacts
        feature_names = None
        try:
            import tempfile
            import os
            
            # Download feature names artifact
            artifact_path = self.mlflow_client.download_artifacts(
                model_version.run_id, 
                "features/feature_names.json"
            )
            
            with open(artifact_path, 'r') as f:
                import json
                feature_data = json.load(f)
                feature_names = feature_data.get('feature_names', [])
            
            print(f"   ✓ Loaded {len(feature_names)} feature names from artifacts")
        except Exception as e:
            print(f"   ⚠️  Could not load feature names from artifacts: {e}")
            # Try to get from model signature
            try:
                model_info = self.mlflow_client.get_model_version(model_name, model_version.version)
                if hasattr(model_info, 'signature') and model_info.signature:
                    feature_names = [inp.name for inp in model_info.signature.inputs]
                    print(f"   ✓ Loaded {len(feature_names)} feature names from signature")
            except:
                pass
        
        metadata = {
            'model_name': model_name,
            'version': model_version.version,
            'stage': stage,
            'run_id': model_version.run_id,
            'timestamp': datetime.fromtimestamp(model_version.creation_timestamp / 1000).strftime('%Y%m%d_%H%M%S'),
            'metrics': run.data.metrics,
            'params': run.data.params,
            'feature_names': feature_names or [],
            'source': 'mlflow_registry'
        }
        
        print(f"✅ Loaded model from MLflow Registry")
        print(f"   Version: {metadata['version']}")
        print(f"   Stage: {metadata['stage']}")
        if feature_names:
            print(f"   Features: {len(feature_names)}")
        if 'val_mae' in metadata['metrics']:
            print(f"   Val MAE: {metadata['metrics']['val_mae']:.2f}")
        if 'test_mae' in metadata['metrics']:
            print(f"   Test MAE: {metadata['metrics']['test_mae']:.2f}")
        
        return model, metadata
    
    def _load_latest_local_model(self):
        """Charge le dernier modèle local (fallback)"""
        model_path = self.models_dir / 'latest_model.joblib'
        metadata_path = self.models_dir / 'latest_metadata.json'
        
        print(f"🔍 Looking for local model at: {model_path}")
        print(f"🔍 Looking for metadata at: {metadata_path}")
        
        if not model_path.exists():
            print(f"⚠️  latest_model.joblib not found, searching for latest model...")
            model_files = list(self.models_dir.glob('best_model_*.joblib'))
            
            if not model_files:
                raise FileNotFoundError(f"No models found in {self.models_dir}")
            
            latest_model = sorted(model_files)[-1]
            timestamp = latest_model.stem.replace('best_model_', '')
            
            print(f"✅ Found latest model: {latest_model.name}")
            return self.load_model_by_timestamp(timestamp)
        
        model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['source'] = 'local_file'
        
        print(f"✅ Loaded local model: {metadata['model_name']}")
        print(f"   Timestamp: {metadata['timestamp']}")
        print(f"   Val MAE: {metadata['metrics']['val_mae']:.2f}")
        if 'test_mae' in metadata['metrics']:
            print(f"   Test MAE: {metadata['metrics']['test_mae']:.2f}")
        
        return model, metadata
    
    def load_model_by_timestamp(self, timestamp):
        """
        Charge un modèle local spécifique par timestamp
        
        Args:
            timestamp: Format YYYYMMDD_HHMMSS
        
        Returns:
            tuple: (model, metadata)
        """
        model_path = self.models_dir / f'best_model_{timestamp}.joblib'
        metadata_path = self.models_dir / f'model_metadata_{timestamp}.json'
        
        print(f"🔍 Loading local model with timestamp: {timestamp}")
        print(f"   Model path: {model_path}")
        print(f"   Metadata path: {metadata_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"No model found for timestamp {timestamp}")
        
        model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['source'] = 'local_file'
        
        print(f"✅ Loaded local model: {metadata['model_name']}")
        
        return model, metadata
    
    def list_available_models(self, include_mlflow=True):
        """
        Liste tous les modèles disponibles (local + MLflow)
        
        Args:
            include_mlflow: Si True, inclut les modèles du Registry
        
        Returns:
            dict: {'local': [...], 'mlflow': [...]}
        """
        result = {'local': [], 'mlflow': []}
        
        # Local models
        for metadata_file in self.models_dir.glob('model_metadata_*.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                result['local'].append({
                    'source': 'local',
                    'timestamp': metadata['timestamp'],
                    'name': metadata['model_name'],
                    'val_mae': metadata['metrics']['val_mae'],
                    'test_mae': metadata['metrics'].get('test_mae', None),
                    'test_r2': metadata['metrics'].get('test_r2', None),
                    'file': metadata['model_file']
                })
        
        result['local'].sort(key=lambda x: x['timestamp'], reverse=True)
        
        # MLflow Registry models
        if self.use_mlflow and include_mlflow:
            try:
                model_versions = self.mlflow_client.search_model_versions("name='PremierLeagueModel'")
                
                for mv in model_versions:
                    try:
                        run = self.mlflow_client.get_run(mv.run_id)
                        result['mlflow'].append({
                            'source': 'mlflow',
                            'version': mv.version,
                            'stage': mv.current_stage,
                            'name': mv.name,
                            'val_mae': run.data.metrics.get('val_mae', None),
                            'test_mae': run.data.metrics.get('test_mae', None),
                            'test_r2': run.data.metrics.get('test_r2', None),
                            'run_id': mv.run_id,
                            'timestamp': datetime.fromtimestamp(mv.creation_timestamp / 1000).strftime('%Y%m%d_%H%M%S')
                        })
                    except Exception as e:
                        print(f"⚠️  Could not load metadata for version {mv.version}: {e}")
                
                result['mlflow'].sort(key=lambda x: x['timestamp'], reverse=True)
            except Exception as e:
                print(f"⚠️  Could not list MLflow models: {e}")
        
        return result
    
    def compare_models(self, include_mlflow=True):
        """
        Compare tous les modèles disponibles (local + MLflow)
        
        Args:
            include_mlflow: Si True, inclut les modèles du Registry
        
        Returns:
            tuple: (local_df, mlflow_df)
        """
        import pandas as pd
        
        models = self.list_available_models(include_mlflow=include_mlflow)
        
        local_df = None
        mlflow_df = None
        
        # Local models
        if models['local']:
            local_df = pd.DataFrame(models['local'])
            print(f"\n📊 Local Models ({len(models['local'])}):\n")
            print(local_df.to_string(index=False))
        else:
            print(f"\n⚠️  No local models found in {self.models_dir}")
        
        # MLflow models
        if models['mlflow']:
            mlflow_df = pd.DataFrame(models['mlflow'])
            print(f"\n📊 MLflow Registry Models ({len(models['mlflow'])}):\n")
            print(mlflow_df.to_string(index=False))
        elif include_mlflow:
            print(f"\n⚠️  No models found in MLflow Registry")
        
        return local_df, mlflow_df
    
    def get_production_model_info(self):
        """Récupère les infos du modèle en production"""
        if not self.use_mlflow:
            print("⚠️  MLflow not enabled, cannot get production model info")
            return None
        
        try:
            model_versions = self.mlflow_client.get_latest_versions(
                "PremierLeagueModel", 
                stages=["Production"]
            )
            
            if not model_versions:
                print("⚠️  No model in Production stage")
                return None
            
            mv = model_versions[0]
            run = self.mlflow_client.get_run(mv.run_id)
            
            info = {
                'version': mv.version,
                'stage': mv.current_stage,
                'run_id': mv.run_id,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'timestamp': datetime.fromtimestamp(mv.creation_timestamp / 1000)
            }
            
            print(f"\n📊 Production Model Info:")
            print(f"   Version: {info['version']}")
            print(f"   Created: {info['timestamp']}")
            if 'val_mae' in info['metrics']:
                print(f"   Val MAE: {info['metrics']['val_mae']:.2f}")
            if 'test_mae' in info['metrics']:
                print(f"   Test MAE: {info['metrics']['test_mae']:.2f}")
            
            return info
        except Exception as e:
            print(f"⚠️  Error getting production model info: {e}")
            return None


class ModelPredictor:
    """Classe pour faire des prédictions avec un modèle chargé"""
    
    def __init__(self, model_path=None, models_dir=None, use_mlflow=True, 
                 mlflow_stage="Production", mlflow_config_path='configs/mlflow_config.yaml'):
        """
        Args:
            model_path: Chemin vers le modèle local. Si None, charge depuis MLflow ou latest.
            models_dir: Dossier des modèles locaux. Si None, utilise le chemin par défaut.
            use_mlflow: Si True, privilégie MLflow Registry
            mlflow_stage: Stage du modèle MLflow (Production, Staging, Archived)
            mlflow_config_path: Chemin vers la config MLflow
        """
        if models_dir:
            loader = ModelLoader(models_dir, use_mlflow=use_mlflow, mlflow_config_path=mlflow_config_path)
        else:
            loader = ModelLoader(use_mlflow=use_mlflow, mlflow_config_path=mlflow_config_path)
        
        # Load model
        if model_path:
            # Load specific local model
            timestamp = Path(model_path).stem.replace('best_model_', '')
            self.model, self.metadata = loader.load_model_by_timestamp(timestamp)
        elif use_mlflow:
            # Load from MLflow Registry
            try:
                self.model, self.metadata = loader.load_from_registry(stage=mlflow_stage)
            except Exception as e:
                print(f"⚠️  Could not load from MLflow: {e}")
                print(f"   Falling back to local model...")
                self.model, self.metadata = loader.load_latest_model(prefer_mlflow=False)
        else:
            # Load latest local model
            self.model, self.metadata = loader.load_latest_model(prefer_mlflow=False)
        
        # Get feature names from metadata
        if 'feature_names' in self.metadata:
            self.training_feature_order = self.metadata['feature_names']
        elif 'params' in self.metadata and 'n_features' in self.metadata['params']:
            # MLflow model without explicit feature names
            self.training_feature_order = []
            print(f"⚠️  Feature names not found in metadata, relying on feature order")
        else:
            self.training_feature_order = []
        
        self.feature_names = self.training_feature_order
        
        print(f"📋 Model ready for prediction")
        if self.feature_names:
            print(f"   Features: {len(self.feature_names)}")
        print(f"   Source: {self.metadata.get('source', 'unknown')}")
    
    def predict(self, X):
        """
        Fait une prédiction
        
        Args:
            X: Features (DataFrame ou array)
        
        Returns:
            array: Prédictions
        """
        import pandas as pd
        
        if isinstance(X, pd.DataFrame):
            if self.feature_names:
                missing_features = set(self.feature_names) - set(X.columns)
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")
                
                X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_final_standings(self, features_df):
        """
        Prédit le classement final pour une saison
        
        Args:
            features_df: DataFrame avec colonnes ['team', 'gameweek', ...features]
        
        Returns:
            DataFrame: Prédictions de classement avec ['team', 'predicted_points', 'rank']
        """
        import pandas as pd
        
        print(f"🔍 Checking feature alignment...")
        
        available_features = [col for col in features_df.columns 
                            if col not in ['team', 'season', 'gameweek', 'target_final_points']]
        
        expected_features = self.feature_names
        
        print(f"   Available features: {len(available_features)}")
        print(f"   Expected features: {len(expected_features)}")
        
        if not expected_features:
            print(f"⚠️  No feature names in metadata, using all available features")
            expected_features = available_features
        
        missing_features = set(expected_features) - set(available_features)
        if missing_features:
            raise ValueError(f"❌ Missing features required by model: {missing_features}")
        
        extra_features = set(available_features) - set(expected_features)
        if extra_features:
            print(f"   ⚠️  Extra features (will be ignored): {len(extra_features)} features")
        
        X = features_df[expected_features]
        print(f"   ✅ Using {len(expected_features)} features in correct order")
        
        predictions = self.model.predict(X)
        
        results = pd.DataFrame({
            'team': features_df['team'],
            'gameweek': features_df['gameweek'],
            'predicted_final_points': predictions
        })
        
        latest_predictions = results.sort_values('gameweek').groupby('team').last()
        latest_predictions['predicted_rank'] = latest_predictions['predicted_final_points'].rank(
            ascending=False, method='min'
        ).astype(int)
        
        latest_predictions = latest_predictions.sort_values('predicted_rank')
        
        return latest_predictions.reset_index()


def predict_from_latest_data(data_path, output_path=None, models_dir=None, 
                             use_mlflow=True, mlflow_stage="Production"):
    """
    Fonction utilitaire pour faire des prédictions rapides
    
    Args:
        data_path: Chemin vers les données à prédire
        output_path: Chemin de sortie (optionnel)
        models_dir: Dossier des modèles locaux (optionnel)
        use_mlflow: Si True, utilise MLflow Registry
        mlflow_stage: Stage du modèle MLflow
    """
    import pandas as pd
    
    predictor = ModelPredictor(
        models_dir=models_dir, 
        use_mlflow=use_mlflow,
        mlflow_stage=mlflow_stage
    )
    
    if isinstance(data_path, str):
        if data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            data = pd.read_csv(data_path)
    else:
        data = data_path
    
    predictions = predictor.predict_final_standings(data)
    
    if output_path:
        predictions.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")
    
    print("\n📊 Predicted Final Standings:\n")
    print(predictions[['predicted_rank', 'team', 'predicted_final_points']].to_string(index=False))
    
    return predictions


if __name__ == '__main__':
    """Exemple d'utilisation"""
    
    print("="*70)
    print("MODEL LOADER - DEMO")
    print("="*70)
    
    # Initialize loader with MLflow support
    loader = ModelLoader(use_mlflow=True)
    
    # Compare all models
    print("\n" + "="*70)
    print("COMPARING ALL MODELS")
    print("="*70)
    loader.compare_models(include_mlflow=True)
    
    # Get production model info
    print("\n" + "="*70)
    print("PRODUCTION MODEL INFO")
    print("="*70)
    loader.get_production_model_info()
    
    # Initialize predictor (will use MLflow Production by default)
    print("\n" + "="*70)
    print("LOADING PREDICTOR")
    print("="*70)
    predictor = ModelPredictor(use_mlflow=True, mlflow_stage="Production")
    print(f"\n✅ Model loaded and ready for predictions!")