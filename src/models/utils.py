"""
Utility functions for model management
"""

import joblib
import json
from pathlib import Path
from datetime import datetime
import sys


class ModelLoader:
    """Charge et gère les modèles sauvegardés"""
    
    def __init__(self, models_dir='models/production'):
        # Convertir en Path object
        base_path = Path.cwd()  # Répertoire actuel
        
        # Essayer plusieurs chemins possibles
        possible_paths = [
            base_path / models_dir,               # Depuis la racine du projet
            base_path / '..' / models_dir,        # Depuis notebooks/
            base_path / 'notebooks' / '..' / models_dir,  # Depuis notebooks
            Path(models_dir).resolve(),           # Chemin absolu
        ]
        
        for path in possible_paths:
            path = path.resolve()  # Convertir en chemin absolu
            if path.exists():
                self.models_dir = path
                print(f"✅ Models directory found: {self.models_dir}")
                break
        else:
            # Si aucun chemin ne fonctionne, créer le répertoire
            self.models_dir = base_path / models_dir
            self.models_dir.mkdir(parents=True, exist_ok=True)
            print(f"⚠️  Created models directory: {self.models_dir}")
    
    def load_latest_model(self):
        """
        Charge le dernier modèle sauvegardé
        
        Returns:
            tuple: (model, metadata)
        """
        model_path = self.models_dir / 'latest_model.joblib'
        metadata_path = self.models_dir / 'latest_metadata.json'
        
        print(f"🔍 Looking for model at: {model_path}")
        print(f"🔍 Looking for metadata at: {metadata_path}")
        
        if not model_path.exists():
            # Chercher le modèle le plus récent si latest_model.joblib n'existe pas
            print(f"⚠️  latest_model.joblib not found, searching for latest model...")
            model_files = list(self.models_dir.glob('best_model_*.joblib'))
            
            if not model_files:
                raise FileNotFoundError(f"No models found in {self.models_dir}")
            
            # Prendre le modèle le plus récent
            latest_model = sorted(model_files)[-1]
            timestamp = latest_model.stem.replace('best_model_', '')
            
            print(f"✅ Found latest model: {latest_model.name}")
            return self.load_model_by_timestamp(timestamp)
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Loaded model: {metadata['model_name']}")
        print(f"   Timestamp: {metadata['timestamp']}")
        print(f"   Val MAE: {metadata['metrics']['val_mae']:.2f}")
        print(f"   Test MAE: {metadata['metrics']['test_mae']:.2f}")
        
        return model, metadata
    
    def load_model_by_timestamp(self, timestamp):
        """
        Charge un modèle spécifique par timestamp
        
        Args:
            timestamp: Format YYYYMMDD_HHMMSS
        
        Returns:
            tuple: (model, metadata)
        """
        model_path = self.models_dir / f'best_model_{timestamp}.joblib'
        metadata_path = self.models_dir / f'model_metadata_{timestamp}.json'
        
        print(f"🔍 Loading model with timestamp: {timestamp}")
        print(f"   Model path: {model_path}")
        print(f"   Metadata path: {metadata_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"No model found for timestamp {timestamp}")
        
        model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Loaded model: {metadata['model_name']}")
        
        return model, metadata
    
    def list_available_models(self):
        """
        Liste tous les modèles disponibles
        
        Returns:
            list: Liste de dictionnaires avec infos sur chaque modèle
        """
        models = []
        
        for metadata_file in self.models_dir.glob('model_metadata_*.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                models.append({
                    'timestamp': metadata['timestamp'],
                    'name': metadata['model_name'],
                    'val_mae': metadata['metrics']['val_mae'],
                    'test_mae': metadata['metrics']['test_mae'],
                    'test_r2': metadata['metrics']['test_r2'],
                    'file': metadata['model_file']
                })
        
        # Trier par timestamp (plus récent en premier)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return models
    
    def compare_models(self):
        """
        Compare tous les modèles disponibles
        
        Returns:
            DataFrame: Comparaison des modèles
        """
        import pandas as pd
        
        models = self.list_available_models()
        
        if not models:
            print(f"⚠️  No models found in {self.models_dir}")
            print(f"    Directory contents: {list(self.models_dir.glob('*'))}")
            return None
        
        df = pd.DataFrame(models)
        
        print(f"\n📊 Available Models ({len(models)}):\n")
        print(df.to_string(index=False))
        
        return df


class ModelPredictor:
    """Classe pour faire des prédictions avec un modèle chargé"""
    
    def __init__(self, model_path=None, models_dir=None):
        """
        Args:
            model_path: Chemin vers le modèle. Si None, charge le latest.
            models_dir: Dossier des modèles. Si None, utilise le chemin par défaut.
        """
        # Utiliser le models_dir spécifié ou le chemin par défaut
        if models_dir:
            loader = ModelLoader(models_dir)
        else:
            loader = ModelLoader()
        
        if model_path:
            # Load specific model
            timestamp = Path(model_path).stem.replace('best_model_', '')
            self.model, self.metadata = loader.load_model_by_timestamp(timestamp)
        else:
            # Load latest
            self.model, self.metadata = loader.load_latest_model()
        
        self.feature_names = self.metadata.get('feature_names', [])
    
    def predict(self, X):
        """
        Fait une prédiction
        
        Args:
            X: Features (DataFrame ou array)
        
        Returns:
            array: Prédictions
        """
        import pandas as pd
        
        # Vérifier que les features correspondent
        if isinstance(X, pd.DataFrame):
            if self.feature_names:
                # Réordonner les colonnes si nécessaire
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
        
        # Obtenir les prédictions
        predictions = self.predict(features_df[self.feature_names])
        
        # Créer le DataFrame de résultats
        results = pd.DataFrame({
            'team': features_df['team'],
            'gameweek': features_df['gameweek'],
            'predicted_final_points': predictions
        })
        
        # Pour avoir le classement, prendre la prédiction la plus récente par équipe
        latest_predictions = results.sort_values('gameweek').groupby('team').last()
        latest_predictions['predicted_rank'] = latest_predictions['predicted_final_points'].rank(
            ascending=False, method='min'
        ).astype(int)
        
        # Trier par classement prédit
        latest_predictions = latest_predictions.sort_values('predicted_rank')
        
        return latest_predictions.reset_index()


def predict_from_latest_data(data_path, output_path=None, models_dir=None):
    """
    Fonction utilitaire pour faire des prédictions rapides
    
    Args:
        data_path: Chemin vers les données à prédire
        output_path: Chemin de sortie (optionnel)
        models_dir: Dossier des modèles (optionnel)
    """
    import pandas as pd
    
    # Load predictor avec le models_dir spécifié
    predictor = ModelPredictor(models_dir=models_dir)
    
    # Load data
    if isinstance(data_path, str):
        if data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            data = pd.read_csv(data_path)
    else:
        data = data_path
    
    # Predict
    predictions = predictor.predict_final_standings(data)
    
    # Save if output path provided
    if output_path:
        predictions.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")
    
    # Display
    print("\n📊 Predicted Final Standings:\n")
    print(predictions[['predicted_rank', 'team', 'predicted_final_points']].to_string(index=False))
    
    return predictions


if __name__ == '__main__':
    """Exemple d'utilisation"""
    
    # List available models
    loader = ModelLoader()
    loader.compare_models()
    
    # Load and use latest model
    predictor = ModelPredictor()
    print(f"\nModel loaded and ready for predictions!")