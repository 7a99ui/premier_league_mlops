"""
Corrected Prediction Pipeline with Feature Alignment
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import sys
from pathlib import Path

# Ajoutez le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import depuis le même dossier
from .feature_alignment import FeatureAligner
from .utils import ModelPredictor

class AlignedPredictionPipeline:
    """Prediction pipeline with automatic feature alignment"""
    
    def __init__(self, model_path=None):
        self.predictor = ModelPredictor(model_path)
        self.aligner = FeatureAligner()
        
        # Load model features and scaler
        project_root = Path(__file__).parent.parent.parent
        self.aligner.load_model_features(
            project_root / 'models' / 'production' / 'latest_metadata.json'
        )
        self.aligner.load_scaler(
            project_root / 'data' / 'processed' / 'v1' / 'scaler.joblib'
        )
        
        self.predictions_dir = project_root / 'predictions_aligned'
        self.predictions_dir.mkdir(exist_ok=True)
    
    def predict(self, season, gameweek, features_path):
        """Make predictions with automatic feature alignment"""
        
        print(f"\n{'='*70}")
        print(f"PREDICTING WITH FEATURE ALIGNMENT")
        print(f"{'='*70}")
        print(f"Season: {season}")
        print(f"Gameweek: {gameweek}")
        print(f"Data: {features_path}")
        print(f"{'='*70}\n")
        
        # 1. Load new data
        df = pd.read_parquet(features_path)
        
        # 2. Filter for specific season/gameweek
        season_data = self._find_season_data(df, season, gameweek)
        
        # 3. Align features to model format
        print(f"🔧 Aligning features to model format...")
        aligned_data = self.aligner.align_features(season_data)
        
        # 4. Apply scaling
        X_scaled = self.aligner.transform_with_scaler(aligned_data)
        
        # 5. Make predictions
        predictions = self.predictor.model.predict(X_scaled)
        
        # 6. Create results
        results = self._create_results(aligned_data, predictions, season, gameweek)
        
        # 7. Save and display
        self._save_results(results, season, gameweek)
        self._display_results(results)
        
        return results
    
    def _find_season_data(self, df, season, gameweek):
        """Find data for specific season and gameweek"""
        season_formats = [season, season.replace('-', '/'), season.replace('-', '_')]
        
        for season_format in season_formats:
            filtered = df[
                (df['season'] == season_format) & 
                (df['gameweek'] == gameweek)
            ]
            if len(filtered) > 0:
                print(f"✅ Found data with season format: '{season_format}'")
                return filtered
        
        raise ValueError(f"No data found for season {season} at gameweek {gameweek}")
    
    def _create_results(self, aligned_data, predictions, season, gameweek):
        """Create results DataFrame"""
        
        results = pd.DataFrame({
            'team': aligned_data['team'] if 'team' in aligned_data.columns else [f"Team_{i}" for i in range(len(predictions))],
            'predicted_final_points': predictions,
            'prediction_timestamp': datetime.now().isoformat(),
            'model_version': self.predictor.metadata['timestamp'],
            'data_version': Path(features_path).parent.name if 'features_path' in locals() else 'unknown'
        })
        
        # Add actual points if available
        if 'target_final_points' in aligned_data.columns:
            results['actual_final_points'] = aligned_data['target_final_points'].values
            results['error'] = results['actual_final_points'] - results['predicted_final_points']
        
        # Calculate ranks
        results['predicted_rank'] = results['predicted_final_points'].rank(
            ascending=False, method='min'
        ).astype(int)
        
        if 'actual_final_points' in results.columns:
            results['actual_rank'] = results['actual_final_points'].rank(
                ascending=False, method='min'
            ).astype(int)
        
        return results.sort_values('predicted_rank')
    
    def _save_results(self, results, season, gameweek):
        """Save predictions to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{season.replace('-', '_')}_gw{gameweek:02d}_{timestamp}.csv"
        filepath = self.predictions_dir / filename
        
        results.to_csv(filepath, index=False)
        print(f"\n💾 Predictions saved to: {filepath}")
    
    def _display_results(self, results):
        """Display formatted results"""
        print(f"\n{'='*70}")
        print("PREDICTED FINAL STANDINGS")
        print(f"{'='*70}\n")
        
        print(f"{'Rank':<6} {'Team':<25} {'Predicted':<10} {'Actual':<10} {'Error':<10}")
        print("-" * 70)
        
        for _, row in results.iterrows():
            rank = int(row['predicted_rank'])
            team = str(row['team'])[:24]
            
            # Emojis for visualization
            if rank <= 4: emoji = "🏆"
            elif rank <= 6: emoji = "⚽"
            elif rank >= 18: emoji = "⚠️"
            else: emoji = "  "
            
            pred = f"{row['predicted_final_points']:6.1f}"
            
            if 'actual_final_points' in row:
                actual = f"{row['actual_final_points']:6.1f}"
                error = f"{row['error']:+7.1f}"
                print(f"{rank:<6} {team:<25} {pred:<10} {actual:<10} {error:<10} {emoji}")
            else:
                print(f"{rank:<6} {team:<25} {pred:<10} {'N/A':<10} {'N/A':<10} {emoji}")
        
        if 'error' in results.columns:
            mae = results['error'].abs().mean()
            print(f"\n📊 Mean Absolute Error: {mae:.2f} points")
            
            # Check if predictions are realistic
            if results['predicted_final_points'].max() > 100:
                print(f"⚠️  Warning: Some predictions seem high")
                print(f"   Max in Premier League history: ~100 points")

def main():
    parser = argparse.ArgumentParser(description='Predict with Feature Alignment')
    parser.add_argument('--season', required=True, help='Season (e.g., 2023-2024)')
    parser.add_argument('--gameweek', type=int, required=True, help='Gameweek number')
    parser.add_argument('--features-path', required=True, help='Path to features file')
    
    args = parser.parse_args()
    
    pipeline = AlignedPredictionPipeline()
    results = pipeline.predict(args.season, args.gameweek, args.features_path)
    
    print(f"\n✅ Prediction complete with feature alignment!")

if __name__ == '__main__':
    main()