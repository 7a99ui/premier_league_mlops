"""
Data Validation Pipeline using Great Expectations
Validates data quality at different stages of the pipeline
"""

import pandas as pd
import numpy as np
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from pathlib import Path
import json
from datetime import datetime
import argparse


class DataValidator:
    """
    Classe pour valider les données à différentes étapes
    - Données brutes (raw data)
    - Features engineerées
    """
    
    def __init__(self, ge_context_dir='great_expectations'):
        """
        Args:
            ge_context_dir: Chemin vers le context Great Expectations
        """
        self.context = gx.get_context(context_root_dir=ge_context_dir)
        self.validation_results = []
    
    def validate_raw_results(self, results_df, season):
        """
        Valide les résultats des matchs (données brutes)
        
        Checks:
        - Colonnes requises présentes
        - Pas de valeurs nulles dans les colonnes critiques
        - Gameweek entre 1 et 38
        - Buts >= 0
        - Match IDs uniques
        """
        print(f"\n🔍 Validating raw results for {season}...")
        
        expectations = []
        
        # 1. Vérifier les colonnes requises
        required_columns = ['match_id', 'gameweek', 'home_team', 'away_team', 
                          'home_goals', 'away_goals', 'result']
        
        for col in required_columns:
            if col not in results_df.columns:
                expectations.append({
                    'expectation': 'column_exists',
                    'column': col,
                    'success': False,
                    'message': f"Missing required column: {col}"
                })
            else:
                expectations.append({
                    'expectation': 'column_exists',
                    'column': col,
                    'success': True
                })
        
        # 2. Vérifier les valeurs nulles
        null_checks = {
            'match_id': results_df['match_id'].isnull().sum(),
            'gameweek': results_df['gameweek'].isnull().sum(),
            'home_team': results_df['home_team'].isnull().sum(),
            'away_team': results_df['away_team'].isnull().sum(),
        }
        
        for col, null_count in null_checks.items():
            expectations.append({
                'expectation': 'values_not_null',
                'column': col,
                'success': null_count == 0,
                'message': f"Found {null_count} null values" if null_count > 0 else "No null values"
            })
        
        # 3. Vérifier les ranges
        if 'gameweek' in results_df.columns:
            min_gw = results_df['gameweek'].min()
            max_gw = results_df['gameweek'].max()
            valid_range = 1 <= min_gw and max_gw <= 38
            
            expectations.append({
                'expectation': 'gameweek_in_range',
                'success': valid_range,
                'message': f"Gameweek range: {min_gw}-{max_gw} {'✓' if valid_range else '✗'}"
            })
        
        # 4. Vérifier que les buts sont >= 0
        if 'home_goals' in results_df.columns:
            negative_goals = (results_df['home_goals'] < 0).sum()
            expectations.append({
                'expectation': 'goals_non_negative',
                'column': 'home_goals',
                'success': negative_goals == 0,
                'message': f"Found {negative_goals} negative goals" if negative_goals > 0 else "All goals valid"
            })
        
        if 'away_goals' in results_df.columns:
            negative_goals = (results_df['away_goals'] < 0).sum()
            expectations.append({
                'expectation': 'goals_non_negative',
                'column': 'away_goals',
                'success': negative_goals == 0,
                'message': f"Found {negative_goals} negative goals" if negative_goals > 0 else "All goals valid"
            })
        
        # 5. Vérifier l'unicité des match_ids
        if 'match_id' in results_df.columns:
            duplicates = results_df['match_id'].duplicated().sum()
            expectations.append({
                'expectation': 'match_id_unique',
                'success': duplicates == 0,
                'message': f"Found {duplicates} duplicate match IDs" if duplicates > 0 else "All match IDs unique"
            })
        
        # Résumé
        success_count = sum(1 for e in expectations if e['success'])
        total_count = len(expectations)
        
        print(f"   Expectations passed: {success_count}/{total_count}")
        
        if success_count == total_count:
            print(f"   ✅ All validations passed for {season}")
        else:
            print(f"   ⚠️  Some validations failed for {season}")
            for exp in expectations:
                if not exp['success']:
                    print(f"      ✗ {exp['expectation']}: {exp['message']}")
        
        return {
            'season': season,
            'data_type': 'raw_results',
            'timestamp': datetime.now().isoformat(),
            'expectations': expectations,
            'success_rate': success_count / total_count
        }
    
    def validate_standings(self, standings_df, season):
        """
        Valide les classements
        
        Checks:
        - Colonnes requises présentes
        - 20 équipes par gameweek
        - Points cohérents avec W/D/L
        - Positions de 1 à 20
        """
        print(f"\n🔍 Validating standings for {season}...")
        
        expectations = []
        
        # 1. Vérifier les colonnes requises
        required_columns = ['team', 'season', 'gameweek', 'position', 'played', 
                          'won', 'drawn', 'lost', 'points', 'goals_for', 
                          'goals_against', 'goal_difference']
        
        for col in required_columns:
            if col not in standings_df.columns:
                expectations.append({
                    'expectation': 'column_exists',
                    'column': col,
                    'success': False,
                    'message': f"Missing required column: {col}"
                })
            else:
                expectations.append({
                    'expectation': 'column_exists',
                    'column': col,
                    'success': True
                })
        
        # 2. Vérifier qu'il y a 20 équipes par gameweek
        teams_per_gameweek = standings_df.groupby('gameweek')['team'].nunique()
        all_have_20 = (teams_per_gameweek == 20).all()
        
        expectations.append({
            'expectation': '20_teams_per_gameweek',
            'success': all_have_20,
            'message': f"All gameweeks have 20 teams" if all_have_20 else f"Some gameweeks missing teams"
        })
        
        # 3. Vérifier la cohérence des points (points = wins*3 + draws*1)
        if all(col in standings_df.columns for col in ['won', 'drawn', 'points']):
            standings_df['calculated_points'] = standings_df['won'] * 3 + standings_df['drawn'] * 1
            points_match = (standings_df['points'] == standings_df['calculated_points']).all()
            
            expectations.append({
                'expectation': 'points_calculation_correct',
                'success': points_match,
                'message': "Points calculation correct" if points_match else "Points calculation mismatch"
            })
        
        # 4. Vérifier les positions (1 à 20)
        if 'position' in standings_df.columns:
            min_pos = standings_df['position'].min()
            max_pos = standings_df['position'].max()
            valid_positions = min_pos == 1 and max_pos == 20
            
            expectations.append({
                'expectation': 'positions_1_to_20',
                'success': valid_positions,
                'message': f"Positions range: {min_pos}-{max_pos} {'✓' if valid_positions else '✗'}"
            })
        
        # 5. Vérifier la différence de buts
        if all(col in standings_df.columns for col in ['goals_for', 'goals_against', 'goal_difference']):
            standings_df['calculated_gd'] = standings_df['goals_for'] - standings_df['goals_against']
            gd_match = (standings_df['goal_difference'] == standings_df['calculated_gd']).all()
            
            expectations.append({
                'expectation': 'goal_difference_correct',
                'success': gd_match,
                'message': "Goal difference correct" if gd_match else "Goal difference mismatch"
            })
        
        # Résumé
        success_count = sum(1 for e in expectations if e['success'])
        total_count = len(expectations)
        
        print(f"   Expectations passed: {success_count}/{total_count}")
        
        if success_count == total_count:
            print(f"   ✅ All validations passed for {season}")
        else:
            print(f"   ⚠️  Some validations failed for {season}")
        
        return {
            'season': season,
            'data_type': 'standings',
            'timestamp': datetime.now().isoformat(),
            'expectations': expectations,
            'success_rate': success_count / total_count
        }
    
    def validate_features(self, features_df):
        """
        Valide les features engineerées
        
        Checks:
        - Colonnes requises présentes
        - Pas trop de valeurs manquantes
        - Valeurs dans des ranges raisonnables
        - Target disponible pour l'entraînement
        """
        print(f"\n🔍 Validating engineered features...")
        
        expectations = []
        
        # 1. Vérifier les colonnes requises
        required_columns = ['season', 'team', 'gameweek', 'current_points', 
                          'target_final_points']
        
        for col in required_columns:
            if col not in features_df.columns:
                expectations.append({
                    'expectation': 'column_exists',
                    'column': col,
                    'success': False,
                    'message': f"Missing required column: {col}"
                })
            else:
                expectations.append({
                    'expectation': 'column_exists',
                    'column': col,
                    'success': True
                })
        
        # 2. Vérifier le pourcentage de valeurs manquantes
        total_rows = len(features_df)
        missing_pct = (features_df.isnull().sum() / total_rows * 100)
        
        # Ne devrait pas avoir plus de 20% de valeurs manquantes pour les features critiques
        critical_features = ['current_points', 'target_final_points', 'points_per_game']
        
        for feat in critical_features:
            if feat in features_df.columns:
                pct = missing_pct[feat]
                success = pct < 20
                expectations.append({
                    'expectation': 'low_missing_rate',
                    'column': feat,
                    'success': success,
                    'message': f"Missing rate: {pct:.2f}% {'✓' if success else '✗'}"
                })
        
        # 3. Vérifier les ranges raisonnables
        if 'current_points' in features_df.columns:
            max_points = features_df['current_points'].max()
            valid_points = 0 <= max_points <= 114  # 38 matchs * 3 points max
            
            expectations.append({
                'expectation': 'points_in_valid_range',
                'success': valid_points,
                'message': f"Max points: {max_points} {'✓' if valid_points else '✗ (impossible value)'}"
            })
        
        if 'gameweek' in features_df.columns:
            min_gw = features_df['gameweek'].min()
            max_gw = features_df['gameweek'].max()
            valid_gw = 1 <= min_gw and max_gw <= 38
            
            expectations.append({
                'expectation': 'gameweek_in_range',
                'success': valid_gw,
                'message': f"Gameweek range: {min_gw}-{max_gw} {'✓' if valid_gw else '✗'}"
            })
        
        # 4. Vérifier que les targets sont disponibles
        if 'target_final_points' in features_df.columns:
            null_targets = features_df['target_final_points'].isnull().sum()
            target_availability = (total_rows - null_targets) / total_rows * 100
            
            expectations.append({
                'expectation': 'target_available',
                'success': target_availability > 90,
                'message': f"Target availability: {target_availability:.2f}%"
            })
        
        # 5. Vérifier les duplicatas
        duplicate_rows = features_df.duplicated(subset=['season', 'team', 'gameweek']).sum()
        
        expectations.append({
            'expectation': 'no_duplicates',
            'success': duplicate_rows == 0,
            'message': f"Found {duplicate_rows} duplicate rows" if duplicate_rows > 0 else "No duplicates"
        })
        
        # Résumé
        success_count = sum(1 for e in expectations if e['success'])
        total_count = len(expectations)
        
        print(f"   Expectations passed: {success_count}/{total_count}")
        print(f"   Total records: {len(features_df):,}")
        print(f"   Total features: {len(features_df.columns)}")
        
        if success_count == total_count:
            print(f"   ✅ All validations passed")
        else:
            print(f"   ⚠️  Some validations failed")
            for exp in expectations:
                if not exp['success']:
                    print(f"      ✗ {exp['expectation']}: {exp['message']}")
        
        return {
            'data_type': 'features',
            'timestamp': datetime.now().isoformat(),
            'n_records': len(features_df),
            'n_features': len(features_df.columns),
            'expectations': expectations,
            'success_rate': success_count / total_count
        }
    
    def validate_all_raw_data(self, seasons, raw_data_dir='data/raw'):
        """Valide les données brutes pour toutes les saisons"""
        print(f"\n{'='*70}")
        print("VALIDATING RAW DATA FOR ALL SEASONS")
        print(f"{'='*70}")
        
        all_results = []
        
        for season in seasons:
            season_dir = Path(raw_data_dir) / season
            
            # Valider les résultats
            results_path = season_dir / 'results.csv'
            if results_path.exists():
                results_df = pd.read_csv(results_path)
                result = self.validate_raw_results(results_df, season)
                all_results.append(result)
            
            # Valider les standings
            standings_path = season_dir / 'standings.csv'
            if standings_path.exists():
                standings_df = pd.read_csv(standings_path)
                result = self.validate_standings(standings_df, season)
                all_results.append(result)
        
        # Sauvegarder les résultats
        self._save_validation_results(all_results, 'data/validation_reports/raw_data_validation.json')
        
        return all_results
    
    def _save_validation_results(self, results, output_path):
        """Sauvegarde les résultats de validation"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir les types NumPy en types Python natifs
        def convert_to_native_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_validations': len(results),
            'results': convert_to_native_types(results),
            'overall_success_rate': sum(r['success_rate'] for r in results) / len(results) if results else 0
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Validation results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Data Validation Pipeline')
    parser.add_argument(
        '--mode',
        choices=['raw', 'features', 'all'],
        default='all',
        help='Validation mode'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2015-2016', '2016-2017', '2017-2018', '2018-2019', 
                '2019-2020', '2020-2021', '2021-2022', '2022-2023'],
        help='Seasons to validate'
    )
    parser.add_argument(
        '--raw-data-dir',
        default='data/raw',
        help='Directory containing raw data'
    )
    parser.add_argument(
        '--features-path',
        default='data/processed/v1/features.parquet',
        help='Path to features file'
    )
    
    args = parser.parse_args()
    
    validator = DataValidator()
    
    if args.mode in ['raw', 'all']:
        print("\n📋 Validating raw data...")
        validator.validate_all_raw_data(args.seasons, args.raw_data_dir)
    
    if args.mode in ['features', 'all']:
        print("\n📋 Validating features...")
        features_path = Path(args.features_path)
        if features_path.exists():
            if features_path.suffix == '.parquet':
                features_df = pd.read_parquet(features_path)
            else:
                features_df = pd.read_csv(features_path)
            
            result = validator.validate_features(features_df)
            validator._save_validation_results(
                [result], 
                'data/validation_reports/features_validation.json'
            )
        else:
            print(f"⚠️  Features file not found: {features_path}")
    
    print("\n✅ Validation complete!")


if __name__ == '__main__':
    main()