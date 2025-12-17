"""
Prediction Pipeline with MLflow Registry Support
Fait des prédictions du classement final à partir d'un gameweek donné
Version simplifiée sans versioning des données
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
import sys
import joblib
import mlflow

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.utils import ModelPredictor


class PredictionPipeline:
    """Pipeline pour générer et sauvegarder des prédictions"""
    
    def __init__(self, model_path=None, use_mlflow=True, mlflow_stage="Production"):
        """
        Args:
            model_path: Chemin vers un modèle local spécifique. Si None, utilise MLflow ou latest.
            use_mlflow: Si True, charge depuis MLflow Registry
            mlflow_stage: Stage du modèle MLflow (Production, Staging, Archived)
        """
        self.predictor = ModelPredictor(
            model_path=model_path,
            use_mlflow=use_mlflow,
            mlflow_stage=mlflow_stage
        )
        
        # Utiliser le chemin de base du projet
        self.predictions_dir = Path(__file__).parent.parent.parent / 'predictions'
        self.predictions_dir.mkdir(exist_ok=True)
        self.features_cache = None  # Cache pour les features
        
        # Log model source
        model_source = self.predictor.metadata.get('source', 'unknown')
        print(f"🔍 Model source: {model_source}")
        if model_source == 'mlflow_registry':
            print(f"   Stage: {self.predictor.metadata.get('stage', 'unknown')}")
            print(f"   Version: {self.predictor.metadata.get('version', 'unknown')}")
        
        # ===== Charger le scaler depuis data/processed/ =====
        self.scaler = None
        project_root = self._get_project_root()
        
        # Chemin simplifié sans versioning
        scaler_path = project_root / 'data' / 'processed' / 'scaler.joblib'
        
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Scaler loaded from: {scaler_path}")
            except Exception as e:
                print(f"⚠️  Failed to load scaler: {e}")
        else:
            print(f"⚠️  Scaler not found at: {scaler_path}")
            print(f"   Predictions may be inaccurate without scaling!")
    
    def _get_project_root(self):
        """Retourne le chemin racine du projet"""
        return Path(__file__).parent.parent.parent
    
    def load_all_features(self):
        """
        Charge toutes les features depuis data/processed/
        Utilise features.parquet en priorité, sinon train+val+test
        
        Returns:
            DataFrame: Données combinées sans duplications
        """
        if self.features_cache is not None:
            return self.features_cache
            
        print("📂 Chargement de toutes les données...")
        
        project_root = self._get_project_root()
        data_dir = project_root / 'data' / 'processed'
        
        print(f"   📍 Chemin utilisé: {data_dir.resolve()}")
        
        # Charger SEULEMENT features.parquet si disponible
        features_file = data_dir / 'features.parquet'
        
        if features_file.exists():
            print(f"  ✓ Chargement: features.parquet")
            combined_data = pd.read_parquet(features_file)
            print(f"  Total: {len(combined_data):,} lignes, {combined_data['team'].nunique()} équipes")
            
            # Afficher les saisons
            print(f"  📅 Saisons présentes:")
            for season in sorted(combined_data['season'].unique()):
                count = len(combined_data[combined_data['season'] == season])
                print(f"     - {season}: {count} lignes")
            
            # Vérification des colonnes
            feature_cols = [col for col in combined_data.columns 
                        if col not in ['team', 'season', 'gameweek', 'target_final_points', 'target_final_position']]
            print(f"  Features: {len(feature_cols)}")
            
            # Vérifier qu'il n'y a pas de duplications
            key_cols = ['team', 'season', 'gameweek']
            if all(col in combined_data.columns for col in key_cols):
                duplicates = combined_data.duplicated(subset=key_cols).sum()
                if duplicates > 0:
                    print(f"  ⚠️  {duplicates} duplications trouvées, nettoyage...")
                    combined_data = combined_data.drop_duplicates(
                        subset=key_cols,
                        keep='first'
                    ).reset_index(drop=True)
                    print(f"  ✅ Données nettoyées: {len(combined_data):,} lignes")
            
            # Vérifier la compatibilité avec le modèle
            model_features = self._get_model_feature_names()
            if model_features:
                missing_features = set(model_features) - set(combined_data.columns)
                if missing_features:
                    print(f"  ⚠️  Features manquantes: {len(missing_features)} features")
            
            self.features_cache = combined_data
            return combined_data
        
        else:
            # FALLBACK: Si features.parquet n'existe pas, charger train+val+test
            print(f"  ⚠️  features.parquet non trouvé, chargement train+val+test...")
            
            possible_paths = [
                data_dir / 'train.parquet',
                data_dir / 'val.parquet', 
                data_dir / 'test.parquet',
            ]
            
            all_data = []
            
            for path in possible_paths:
                if path.exists():
                    print(f"  ✓ Chargement: {path.name}")
                    try:
                        df = pd.read_parquet(path)
                        all_data.append(df)
                    except Exception as e:
                        print(f"  ❌ Erreur: {e}")
            
            if not all_data:
                print(f"\n❌ ERREUR: Aucune donnée trouvée dans {data_dir}")
                print(f"   Fichiers attendus:")
                print(f"   - features.parquet (préféré)")
                print(f"   - OU train.parquet + val.parquet + test.parquet")
                raise FileNotFoundError(f"Aucune donnée trouvée dans {data_dir}")
            
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Déduplication OBLIGATOIRE
            print(f"  📊 Données brutes: {len(combined_data):,} lignes")
            
            key_cols = ['team', 'season', 'gameweek']
            if all(col in combined_data.columns for col in key_cols):
                duplicates_count = combined_data.duplicated(subset=key_cols).sum()
                
                if duplicates_count > 0:
                    print(f"  ⚠️  DUPLICATIONS: {duplicates_count} lignes")
                    combined_data = combined_data.drop_duplicates(
                        subset=key_cols,
                        keep='first'
                    ).reset_index(drop=True)
                    print(f"  ✅ Après déduplication: {len(combined_data):,} lignes")
            
            # Vérification des colonnes
            feature_cols = [col for col in combined_data.columns 
                        if col not in ['team', 'season', 'gameweek', 'target_final_points', 'target_final_position']]
            print(f"  Features: {len(feature_cols)}")
            
            self.features_cache = combined_data
            return combined_data
    
    def _get_model_feature_names(self):
        """Récupère les noms des features depuis les métadonnées du modèle"""
        if 'feature_names' in self.predictor.metadata:
            return self.predictor.metadata['feature_names']
        elif 'params' in self.predictor.metadata:
            return []
        else:
            return []
    
    def _prepare_features_for_prediction(self, features_df):
        """
        Prépare les features pour la prédiction (ordre + scaling)
        
        Args:
            features_df: DataFrame avec features brutes
        
        Returns:
            DataFrame: Features préparées (ordonnées et scalées)
        """
        print(f"🔍 Préparation des features pour la prédiction...")
        
        # 1. Obtenir l'ordre exact des features d'entraînement
        feature_order = self._get_model_feature_names()
        
        if not feature_order:
            print(f"⚠️  Pas d'information sur l'ordre des features dans les métadonnées")
            feature_order = [col for col in features_df.columns 
                           if col not in ['team', 'season', 'gameweek', 'target_final_points', 'target_final_position']]
        
        print(f"   Nombre de features attendues: {len(feature_order)}")
        
        # 2. Créer les features manquantes au lieu de lever une exception
        missing_features = set(feature_order) - set(features_df.columns)
        if missing_features:
            print(f"   ⚠️  Features manquantes: {len(missing_features)}")
            print(f"      Création avec valeur 0...")
            
            # Créer les features manquantes avec valeur 0
            for feat in missing_features:
                features_df[feat] = 0
        
        # 3. Réorganiser les features dans le bon ordre
        X_ordered = features_df[feature_order].copy()
        print(f"   ✅ Features réorganisées ({X_ordered.shape[1]})")
        
        # 4. DEBUG: Afficher les statistiques avant scaling
        if len(X_ordered) > 0:
            print(f"   📊 Statistiques avant scaling (moyenne des 3 premières features):")
            for i, col in enumerate(X_ordered.columns[:3]):
                print(f"      {col}: mean={X_ordered[col].mean():.3f}, std={X_ordered[col].std():.3f}")
        
        # 5. Appliquer le scaling si disponible
        if self.scaler is not None:
            print(f"   🔄 Application du StandardScaler...")
                # Filter features to match scaler requirements if possible
                if hasattr(self.scaler, 'feature_names_in_'):
                    scaler_features = self.scaler.feature_names_in_
                    # Check for missing or extra features
                    missing = set(scaler_features) - set(X_ordered.columns)
                    extra = set(X_ordered.columns) - set(scaler_features)
                    
                    if missing:
                        print(f"   ⚠️  Missing features for scaler: {len(missing)}")
                        for col in missing:
                            X_ordered[col] = 0
                            
                    if extra:
                        print(f"   ⚠️  Ignoring extra features not in scaler: {extra}")
                        X_ordered = X_ordered[list(scaler_features)]
                    else:
                         X_ordered = X_ordered[list(scaler_features)]
                
                X_scaled = self.scaler.transform(X_ordered)
                X_prepared = pd.DataFrame(X_scaled, columns=X_ordered.columns, index=X_ordered.index)
                print(f"   ✅ Features scalées")
                
                # DEBUG: Vérifier après scaling
                print(f"   📊 Statistiques après scaling (moyenne des 3 premières features):")
                for i, col in enumerate(X_prepared.columns[:3]):
                    print(f"      {col}: mean={X_prepared[col].mean():.3f}, std={X_prepared[col].std():.3f}")
                
                return X_prepared
            except Exception as e:
                print(f"   ⚠️  Erreur lors du scaling: {e}")
                print(f"   ⚠️  Utilisation des features brutes (non scalées)")
                return X_ordered
        else:
            print(f"   ⚠️  Aucun scaler disponible, utilisation des features brutes")
            return X_ordered
    
    def predict_at_gameweek(self, season, gameweek, features_path=None):
        """
        Prédit le classement final à partir d'un gameweek donné
        
        Args:
            season: Saison (ex: '2024-2025')
            gameweek: Gameweek actuel (ex: 15)
            features_path: Chemin vers les features. Si None, cherche automatiquement
        
        Returns:
            DataFrame: Prédictions avec classement
        """
        print(f"\n{'='*70}")
        print(f"PREDICTING FINAL STANDINGS")
        print(f"{'='*70}")
        print(f"Season: {season}")
        print(f"After Gameweek: {gameweek}")
        
        # Display model info
        model_name = self.predictor.metadata.get('model_name', 'Unknown')
        model_source = self.predictor.metadata.get('source', 'unknown')
        
        print(f"Model: {model_name}")
        print(f"Source: {model_source}")
        
        if model_source == 'mlflow_registry':
            print(f"Stage: {self.predictor.metadata.get('stage', 'unknown')}")
            print(f"Version: {self.predictor.metadata.get('version', 'unknown')}")
        
        # Display metrics if available
        metrics = self.predictor.metadata.get('metrics', {})
        if 'val_mae' in metrics:
            print(f"Val MAE: {metrics['val_mae']:.2f}")
        if 'test_mae' in metrics:
            print(f"Test MAE: {metrics['test_mae']:.2f}")
        
        print(f"{'='*70}\n")
        
        # Charger les features
        if features_path:
            print(f"📂 Chargement des données depuis: {features_path}")
            features_df = pd.read_parquet(features_path)
        else:
            features_df = self.load_all_features()
        
        # Essayer différents formats de saison
        season_formats = [season, season.replace('-', '/'), season.replace('-', '_')]
        season_data = None
        
        for season_format in season_formats:
            temp_data = features_df[
                (features_df['season'] == season_format) & 
                (features_df['gameweek'] == gameweek)
            ].copy()
            
            if len(temp_data) > 0:
                season_data = temp_data
                print(f"✅ Données trouvées avec le format de saison: '{season_format}'")
                break
        
        if season_data is None or len(season_data) == 0:
            # Vérifier toutes les saisons disponibles
            print(f"\n🔍 Saisons disponibles dans les données:")
            unique_seasons = features_df['season'].unique()
            for s in sorted(unique_seasons):
                print(f"   - '{s}'")
            
            raise ValueError(f"No data found for season {season} at gameweek {gameweek}")
        
        # Vérifier et supprimer les duplications
        print(f"📊 Données brutes: {len(season_data)} lignes")
        
        duplicates = season_data.duplicated(subset=['team'], keep=False)
        if duplicates.any():
            print(f"⚠️  DUPLICATIONS DÉTECTÉES: {duplicates.sum()} lignes")
            print(f"   Équipes dupliquées:")
            for team in season_data[duplicates]['team'].unique():
                count = (season_data['team'] == team).sum()
                print(f"     - {team}: {count} fois")
            
            # Supprimer les duplications (garder la première)
            season_data = season_data.drop_duplicates(subset=['team'], keep='first')
            print(f"✅ Données après déduplication: {len(season_data)} équipes")
        else:
            print(f"✅ Pas de duplication détectée: {len(season_data)} équipes")
        
        # Vérifier la target si disponible
        if 'target_final_points' in season_data.columns:
            print(f"🔍 Target (points finaux): {season_data['target_final_points'].min():.1f} à {season_data['target_final_points'].max():.1f}")
        
        # Préparer les features pour la prédiction
        X_prepared = self._prepare_features_for_prediction(season_data)
        
        # Faire les prédictions
        print(f"\n🎯 Génération des prédictions...")
        predictions = self.predictor.model.predict(X_prepared)
        
        print(f"📊 Gamme des prédictions: {predictions.min():.1f} à {predictions.max():.1f}")
        
        # Créer le DataFrame de résultats
        results = pd.DataFrame({
            'team': season_data['team'].values,
            'gameweek': gameweek,
            'predicted_final_points': predictions
        })
        
        # Calculer le classement
        results['predicted_rank'] = results['predicted_final_points'].rank(
            ascending=False, method='min'
        ).astype(int)
        
        # Ajouter des informations supplémentaires
        results['season'] = season
        results['prediction_gameweek'] = gameweek
        results['prediction_timestamp'] = datetime.now().isoformat()
        results['model_name'] = model_name
        results['model_source'] = model_source
        
        if model_source == 'mlflow_registry':
            results['model_version'] = self.predictor.metadata.get('version', 'unknown')
            results['model_stage'] = self.predictor.metadata.get('stage', 'unknown')
        elif 'timestamp' in self.predictor.metadata:
            results['model_timestamp'] = self.predictor.metadata['timestamp']
        
        # Si disponible, ajouter le classement réel
        if 'target_final_points' in season_data.columns:
            # Check if we have valid data (not all NaNs)
            if season_data['target_final_points'].notna().any():
                actual_standings = season_data[['team', 'target_final_points']].copy()
                actual_standings.columns = ['team', 'actual_final_points']
                actual_standings['actual_rank'] = actual_standings['actual_final_points'].rank(
                    ascending=False, method='min'
                ).astype(int)
                
                results = results.merge(actual_standings, on='team', how='left')
            else:
                print(f"   ℹ️  Target points are all NaN (future season?), skipping accuracy metrics")
            
            # Calculer les erreurs si les données réelles sont disponibles
            if 'actual_final_points' in results.columns:
                results['points_error'] = (
                    results['actual_final_points'] - results['predicted_final_points']
                )
                results['rank_error'] = (
                    results['actual_rank'] - results['predicted_rank']
                )
                
                # Debug: Vérifier les premières prédictions
                print(f"\n🧪 Comparaison prédictions vs réalité:")
                print(f"{'Team':<25} {'Predicted':<10} {'Actual':<10} {'Error':<10}")
                print("-" * 60)
                for _, row in results.head(5).iterrows():
                    print(f"{row['team'][:24]:<25} {row['predicted_final_points']:9.1f} {row['actual_final_points']:9.1f} {row['points_error']:+9.1f}")
        
        # Trier par classement prédit
        results = results.sort_values('predicted_rank').reset_index(drop=True)
        
        return results
    
    def save_predictions(self, predictions, season, gameweek):
        """
        Sauvegarde les prédictions
        
        Args:
            predictions: DataFrame avec prédictions
            season: Saison
            gameweek: Gameweek
        
        Returns:
            Path: Chemin du fichier sauvegardé
        """
        # Créer un nom de fichier unique
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{season.replace('-', '_')}_gw{gameweek:02d}_{timestamp}.csv"
        filepath = self.predictions_dir / filename
        
        # Sauvegarder
        predictions.to_csv(filepath, index=False)
        print(f"\n💾 Predictions saved to: {filepath}")
        
        # Sauvegarder aussi en "latest" pour cette saison
        latest_filename = f"{season.replace('-', '_')}_latest.csv"
        latest_filepath = self.predictions_dir / latest_filename
        predictions.to_csv(latest_filepath, index=False)
        print(f"💾 Latest prediction saved to: {latest_filepath}")
        
        return filepath
    
    def display_predictions(self, predictions):
        """Affiche les prédictions de manière formatée"""
        print(f"\n{'='*70}")
        print("PREDICTED FINAL STANDINGS")
        print(f"{'='*70}\n")
        
        print(f"{'Rank':<6} {'Team':<25} {'Predicted':<10} {'Actual':<10} {'Error':<10}")
        print("-" * 70)
        
        for _, row in predictions.iterrows():
            rank = int(row['predicted_rank'])
            team = row['team'][:24]  # Tronquer si trop long
            
            # Emoji basé sur le classement
            if rank <= 4:
                emoji = "🏆"  # Top 4 (Champions League)
            elif rank <= 6:
                emoji = "⚽"  # Europa League
            elif rank >= 18:
                emoji = "⚠️"   # Relegation zone
            else:
                emoji = "  "
            
            if 'actual_final_points' in predictions.columns and not pd.isna(row.get('actual_final_points')):
                pred = f"{row['predicted_final_points']:6.1f}"
                actual = f"{row['actual_final_points']:6.1f}"
                error = f"{row['points_error']:+7.1f}"
                print(f"{rank:<6} {team:<25} {pred:<10} {actual:<10} {error:<10} {emoji}")
            else:
                pred = f"{row['predicted_final_points']:6.1f}"
                print(f"{rank:<6} {team:<25} {pred:<10} {'N/A':<10} {'N/A':<10} {emoji}")
        
        print(f"\n{'='*70}")
        print("Legend: 🏆 Top 4 (UCL) | ⚽ Europa spots | ⚠️ Relegation zone")
        print(f"{'='*70}")
    
    def predict_and_save(self, season, gameweek, features_path=None, display=True):
        """
        Pipeline complet : prédire, sauvegarder et afficher
        
        Args:
            season: Saison
            gameweek: Gameweek
            features_path: Chemin vers les features
            display: Afficher les résultats
        
        Returns:
            DataFrame: Prédictions
        """
        # Prédictions
        predictions = self.predict_at_gameweek(season, gameweek, features_path)
        
        # Afficher
        if display:
            self.display_predictions(predictions)
        
        # Sauvegarder
        filepath = self.save_predictions(predictions, season, gameweek)
        
        # Statistiques
        if 'actual_final_points' in predictions.columns:
            mae = predictions['points_error'].abs().mean()
            rmse = np.sqrt((predictions['points_error'] ** 2).mean())
            print(f"\n📊 Prediction Quality (if actual data available):")
            print(f"   Mean Absolute Error: {mae:.2f} points")
            print(f"   Root Mean Squared Error: {rmse:.2f} points")
            
            # Vérifier si les prédictions sont réalistes
            if predictions['predicted_final_points'].max() > 100:
                print(f"\n⚠️  WARNING: Predicted points seem too high!")
                print(f"   Premier League realistic range: ~15-100 points")
                print(f"   Max prediction: {predictions['predicted_final_points'].max():.1f}")
                print(f"   Check: 1) Feature scaling, 2) Feature order, 3) Model training")
            
            # Top 4 accuracy
            top4_pred = set(predictions.nsmallest(4, 'predicted_rank')['team'])
            top4_actual = set(predictions.nsmallest(4, 'actual_rank')['team'])
            top4_acc = len(top4_pred & top4_actual) / 4 * 100
            print(f"   Top 4 Accuracy: {top4_acc:.1f}% ({len(top4_pred & top4_actual)}/4)")
            
            # Relegation accuracy
            releg_pred = set(predictions.nlargest(3, 'predicted_rank')['team'])
            releg_actual = set(predictions.nlargest(3, 'actual_rank')['team'])
            releg_acc = len(releg_pred & releg_actual) / 3 * 100
            print(f"   Relegation Accuracy: {releg_acc:.1f}% ({len(releg_pred & releg_actual)}/3)")
        
        return predictions


def predict_evolution(season, start_gw, end_gw, features_path=None,
                     use_mlflow=True, mlflow_stage="Production"):
    """
    Prédit l'évolution du classement sur plusieurs gameweeks
    
    Args:
        season: Saison
        start_gw: Gameweek de début
        end_gw: Gameweek de fin
        features_path: Chemin vers les features
        use_mlflow: Utiliser MLflow Registry
        mlflow_stage: Stage du modèle MLflow
    
    Returns:
        dict: Prédictions pour chaque gameweek
    """
    print(f"\n{'='*70}")
    print(f"PREDICTION EVOLUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Season: {season}")
    print(f"Gameweeks: {start_gw} to {end_gw}")
    print(f"{'='*70}\n")
    
    pipeline = PredictionPipeline(use_mlflow=use_mlflow, mlflow_stage=mlflow_stage)
    evolution = {}
    
    for gw in range(start_gw, end_gw + 1):
        try:
            print(f"\n🔄 Predicting at gameweek {gw}...")
            predictions = pipeline.predict_at_gameweek(season, gw, features_path)
            evolution[gw] = predictions
            
            if 'points_error' in predictions.columns:
                mae = predictions['points_error'].abs().mean()
                print(f"   ✓ Success - MAE: {mae:.2f}")
            else:
                print(f"   ✓ Success")
                
        except Exception as e:
            print(f"   ✗ Failed: {e}")
    
    # Sauvegarder l'évolution
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'predictions' / 'evolution'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"{season.replace('-', '_')}_evolution_{timestamp}.json"
    
    # Convertir en format sérialisable
    evolution_data = {
        str(gw): pred.to_dict('records') 
        for gw, pred in evolution.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(evolution_data, f, indent=2)
    
    print(f"\n💾 Evolution saved to: {output_file}")
    
    # Calculer les statistiques d'évolution
    if evolution:
        all_predictions = pd.concat([df.assign(gameweek=gw) for gw, df in evolution.items()])
        
        print(f"\n📈 Evolution Statistics:")
        print(f"   Number of gameweeks: {len(evolution)}")
        print(f"   Teams analyzed: {all_predictions['team'].nunique()}")
        
        # Vérifier la stabilité des prédictions
        if 'predicted_final_points' in all_predictions.columns:
            volatility = all_predictions.groupby('team')['predicted_final_points'].std()
            if len(volatility) > 0:
                print(f"   Most stable team: {volatility.idxmin()} (std: {volatility.min():.2f})")
                print(f"   Most volatile team: {volatility.idxmax()} (std: {volatility.max():.2f})")
    
    return evolution


def main():
    parser = argparse.ArgumentParser(description='Predict Premier League Final Standings with MLflow Support')
    parser.add_argument('--season', required=True, help='Season (e.g., 2024-2025)')
    parser.add_argument('--gameweek', type=int, required=True, 
                       help='Current gameweek (e.g., 15)')
    parser.add_argument('--features-path', default=None,
                       help='Path to features file')
    parser.add_argument('--model-path', default=None,
                       help='Path to specific local model (default: uses MLflow Production)')
    parser.add_argument('--evolution', action='store_true',
                       help='Predict evolution from gameweek 10 to current')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow Registry, use local models only')
    parser.add_argument('--mlflow-stage', default='Production',
                       choices=['Production', 'Staging', 'Archived'],
                       help='MLflow model stage (default: Production)')
    
    args = parser.parse_args()
    
    use_mlflow = not args.no_mlflow
    
    if args.evolution:
        # Prédire l'évolution
        predict_evolution(
            args.season, 
            10, 
            args.gameweek, 
            args.features_path,
            use_mlflow=use_mlflow,
            mlflow_stage=args.mlflow_stage
        )
    else:
        # Prédiction simple
        pipeline = PredictionPipeline(
            model_path=args.model_path,
            use_mlflow=use_mlflow,
            mlflow_stage=args.mlflow_stage
        )
        predictions = pipeline.predict_and_save(
            args.season, 
            args.gameweek, 
            args.features_path
        )
    print(f"\n✅ Prediction complete!")
    if use_mlflow:
        print(f"\n💡 Using MLflow Registry model (stage: {args.mlflow_stage})")
        print(f"   To use local model instead: add --no-mlflow flag")
    else:
        print(f"\n💡 Using local model")
        print(f"   To use MLflow Registry: remove --no-mlflow flag")
    
if __name__ == '__main__':
    main()