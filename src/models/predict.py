"""
Prediction Pipeline
Fait des prédictions du classement final à partir d'un gameweek donné
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
import sys

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.utils import ModelPredictor


class PredictionPipeline:
    """Pipeline pour générer et sauvegarder des prédictions"""
    
    def __init__(self, model_path=None):
        """
        Args:
            model_path: Chemin vers un modèle spécifique. Si None, utilise le dernier.
        """
        self.predictor = ModelPredictor(model_path)
        # Utiliser le chemin de base du projet
        self.predictions_dir = Path(__file__).parent.parent.parent / 'predictions'
        self.predictions_dir.mkdir(exist_ok=True)
        self.features_cache = None  # Cache pour les features
    
    def _get_project_root(self):
        """Retourne le chemin racine du projet"""
        return Path(__file__).parent.parent.parent
    
    def load_all_features(self):
        """
        Charge toutes les features (train + val + test)
        """
        if self.features_cache is not None:
            return self.features_cache
            
        print("📂 Chargement de toutes les données...")
        
        # Chemins des données - depuis la racine du projet
        project_root = self._get_project_root()
        data_dir = project_root / 'data' / 'processed' / 'v1'
        train_path = data_dir / 'train.parquet'
        val_path = data_dir / 'val.parquet'
        test_path = data_dir / 'test.parquet'
        features_path = data_dir / 'features.parquet'
        
        all_data = []
        
        # Charger train, val, test
        for path in [train_path, val_path, test_path]:
            if path.exists():
                print(f"  ✓ Chargement: {path.name}")
                df = pd.read_parquet(path)
                all_data.append(df)
        
        # Si on a des données, les combiner
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"  Total: {len(combined_data):,} lignes, {combined_data['team'].nunique()} équipes")
            
            # Vérification des colonnes
            feature_cols = [col for col in combined_data.columns 
                          if col not in ['team', 'season', 'gameweek', 'target_final_points']]
            print(f"  Features: {len(feature_cols)}")
            
            # Vérifier que c'est compatible avec le modèle
            model_features = self.predictor.metadata.get('feature_names', [])
            if model_features:
                missing_features = set(model_features) - set(combined_data.columns)
                if missing_features:
                    print(f"  ⚠️  Features manquantes: {missing_features}")
            
            self.features_cache = combined_data
            return combined_data
        else:
            # Fallback: utiliser features.parquet
            print(f"  ⚠️  Aucune donnée train/val/test trouvée, utilisation de {features_path.name}")
            if features_path.exists():
                features_df = pd.read_parquet(features_path)
                self.features_cache = features_df
                return features_df
            else:
                raise FileNotFoundError(f"Aucune donnée trouvée dans {data_dir}")
    
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
        print(f"Model: {self.predictor.metadata['model_name']}")
        print(f"Val MAE: {self.predictor.metadata['metrics']['val_mae']:.2f}")
        print(f"Test MAE: {self.predictor.metadata['metrics']['test_mae']:.2f}")
        print(f"{'='*70}\n")
        
        # Charger les features
        if features_path:
            print(f"📂 Chargement des données depuis: {features_path}")
            features_df = pd.read_parquet(features_path)
        else:
            features_df = self.load_all_features()
        
        # Filtrer par saison et gameweek
        season_data = features_df[
            (features_df['season'] == season) & 
            (features_df['gameweek'] == gameweek)
        ].copy()
        
        if len(season_data) == 0:
            # Essayer avec le format de saison sans tiret
            season_alt = season.replace('-', '/')
            season_data = features_df[
                (features_df['season'] == season_alt) & 
                (features_df['gameweek'] == gameweek)
            ].copy()
            
            if len(season_data) == 0:
                raise ValueError(f"No data found for season {season} at gameweek {gameweek}")
        
        print(f"📊 Data loaded: {len(season_data)} teams")
        
        # DEBUG: Vérifier la target
        if 'target_final_points' in season_data.columns:
            print(f"🔍 Target range: {season_data['target_final_points'].min():.1f} to {season_data['target_final_points'].max():.1f}")
        
        # DEBUG: Vérifier les features
        feature_cols = [col for col in season_data.columns 
                       if col not in ['team', 'season', 'gameweek', 'target_final_points']]
        print(f"🔍 Features utilisées: {len(feature_cols)}")
        
        # Faire les prédictions
        predictions = self.predictor.predict_final_standings(season_data)
        
        # Ajouter des informations supplémentaires
        predictions['season'] = season
        predictions['prediction_gameweek'] = gameweek
        predictions['prediction_timestamp'] = datetime.now().isoformat()
        
        # Si disponible, ajouter le classement réel
        if 'target_final_points' in season_data.columns:
            actual_standings = season_data.groupby('team').agg({
                'target_final_points': 'first'
            }).reset_index()
            actual_standings.columns = ['team', 'actual_final_points']
            actual_standings['actual_rank'] = actual_standings['actual_final_points'].rank(
                ascending=False, method='min'
            ).astype(int)
            
            predictions = predictions.merge(actual_standings, on='team', how='left')
            
            # Calculer les erreurs si les données réelles sont disponibles
            if 'actual_final_points' in predictions.columns:
                predictions['points_error'] = (
                    predictions['actual_final_points'] - predictions['predicted_final_points']
                )
                predictions['rank_error'] = (
                    predictions['actual_rank'] - predictions['predicted_rank']
                )
                
                # Debug: Vérifier les premières prédictions
                print(f"\n🧪 Premières prédictions vs réalité:")
                for _, row in predictions.head(3).iterrows():
                    print(f"   {row['team']:25} Prédit: {row['predicted_final_points']:6.1f} | "
                          f"Réel: {row['actual_final_points']:6.1f} | "
                          f"Erreur: {row['points_error']:+7.1f}")
        
        return predictions
    
    def save_predictions(self, predictions, season, gameweek):
        """
        Sauvegarde les prédictions
        
        Args:
            predictions: DataFrame avec prédictions
            season: Saison
            gameweek: Gameweek
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
        
        # Colonnes à afficher
        display_cols = ['predicted_rank', 'team', 'predicted_final_points']
        
        # Ajouter les colonnes réelles si disponibles
        if 'actual_final_points' in predictions.columns:
            display_cols.extend(['actual_final_points', 'points_error'])
        
        # Formater et afficher
        print(f"{'Rank':<6} {'Team':<30} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
        print("-"*70)
        
        for _, row in predictions.iterrows():
            rank = int(row['predicted_rank'])
            team = row['team']
            pred_points = row['predicted_final_points']
            
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
                actual = row['actual_final_points']
                error = row['points_error']
                print(f"{rank:<6} {team:<30} {pred_points:<12.1f} {actual:<12.1f} {error:+10.1f} {emoji}")
            else:
                print(f"{rank:<6} {team:<30} {pred_points:<12.1f} {'N/A':<12} {'N/A':<10} {emoji}")
        
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


def predict_evolution(season, start_gw, end_gw, features_path=None):
    """
    Prédit l'évolution du classement sur plusieurs gameweeks
    
    Args:
        season: Saison
        start_gw: Gameweek de début
        end_gw: Gameweek de fin
        features_path: Chemin vers les features
    
    Returns:
        dict: Prédictions pour chaque gameweek
    """
    print(f"\n{'='*70}")
    print(f"PREDICTION EVOLUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Season: {season}")
    print(f"Gameweeks: {start_gw} to {end_gw}")
    print(f"{'='*70}\n")
    
    pipeline = PredictionPipeline()
    evolution = {}
    
    for gw in range(start_gw, end_gw + 1):
        try:
            print(f"\n🔄 Predicting at gameweek {gw}...")
            predictions = pipeline.predict_at_gameweek(season, gw, features_path)
            evolution[gw] = predictions
            print(f"   ✓ Success - MAE: {predictions['points_error'].abs().mean():.2f}")
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
        
        # Calculer la volatilité
        volatility = all_predictions.groupby('team').agg({
            'predicted_final_points': ['mean', 'std'],
            'predicted_rank': ['mean', 'std']
        })
        
        print(f"\n📈 Evolution Statistics:")
        print(f"   Number of gameweeks: {len(evolution)}")
        print(f"   Most stable team: {volatility[('predicted_final_points', 'std')].idxmin()}")
        print(f"   Most volatile team: {volatility[('predicted_final_points', 'std')].idxmax()}")
    
    return evolution


def main():
    parser = argparse.ArgumentParser(description='Predict Premier League Final Standings')
    parser.add_argument('--season', required=True, help='Season (e.g., 2024-2025)')
    parser.add_argument('--gameweek', type=int, required=True, 
                       help='Current gameweek (e.g., 15)')
    parser.add_argument('--features-path', default=None,
                       help='Path to features file')
    parser.add_argument('--model-path', default=None,
                       help='Path to specific model (default: latest)')
    parser.add_argument('--evolution', action='store_true',
                       help='Predict evolution from gameweek 10 to current')
    
    args = parser.parse_args()
    
    if args.evolution:
        # Prédire l'évolution
        predict_evolution(args.season, 10, args.gameweek, args.features_path)
    else:
        # Prédiction simple
        pipeline = PredictionPipeline(args.model_path)
        predictions = pipeline.predict_and_save(
            args.season, 
            args.gameweek, 
            args.features_path
        )
    
    print(f"\n✅ Prediction complete!")


if __name__ == '__main__':
    main()