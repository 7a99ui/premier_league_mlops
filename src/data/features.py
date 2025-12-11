"""
Feature Engineering Pipeline for Premier League Prediction
Creates features from raw match results and standings data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Classe principale pour la création de features
    
    Features générées:
    1. Form features (derniers N matchs)
    2. Cumulative stats (depuis début de saison)
    3. Head-to-head history
    4. Strength of schedule
    5. Historical performance (saisons précédentes)
    """
    
    def __init__(self, form_window=5):
        """
        Args:
            form_window: Nombre de matchs pour calculer la forme récente
        """
        self.form_window = form_window
        self.feature_metadata = {}
    
    def create_features_for_season(self, season, raw_data_dir='data/raw'):
        """
        Crée les features pour une saison complète
        
        Args:
            season: Nom de la saison (ex: '2022-2023')
            raw_data_dir: Dossier contenant les données brutes
        
        Returns:
            DataFrame avec toutes les features
        """
        print(f"\n{'='*70}")
        print(f"CREATING FEATURES FOR {season}")
        print(f"{'='*70}\n")
        
        # Charger les données de la saison
        season_dir = Path(raw_data_dir) / season
        
        results_df = pd.read_csv(season_dir / 'results.csv')
        standings_df = pd.read_csv(season_dir / 'standings.csv')
        
        # Charger les stats si disponibles
        stats_path = season_dir / 'match_stats.csv'
        if stats_path.exists():
            stats_df = pd.read_csv(stats_path)
        else:
            stats_df = None
            print("⚠️  Pas de statistiques détaillées disponibles")
        
        # Obtenir la liste des équipes
        teams = standings_df['team'].unique()
        
        all_features = []
        
        print(f"🔄 Processing {len(teams)} teams across 38 gameweeks...")
        
        # Pour chaque équipe et chaque gameweek
        for team in teams:
            for gameweek in range(1, 39):
                features = self._create_features_for_team_gameweek(
                    team=team,
                    gameweek=gameweek,
                    season=season,
                    results_df=results_df,
                    standings_df=standings_df,
                    stats_df=stats_df
                )
                
                if features:
                    all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        
        print(f"\n✅ Features created successfully!")
        print(f"   Total records: {len(features_df)}")
        print(f"   Total features: {len(features_df.columns)}")
        
        return features_df
    
    def _create_features_for_team_gameweek(self, team, gameweek, season, 
                                          results_df, standings_df, stats_df):
        """Crée les features pour une équipe à un gameweek donné"""
        
        # Obtenir le standing actuel
        current_standing = standings_df[
            (standings_df['team'] == team) & 
            (standings_df['gameweek'] == gameweek)
        ]
        
        if current_standing.empty:
            return None
        
        current_standing = current_standing.iloc[0]
        
        # Initialiser le dictionnaire de features
        features = {
            'season': season,
            'team': team,
            'gameweek': gameweek,
        }
        
        # 1. CUMULATIVE FEATURES (depuis le début de la saison)
        features.update(self._get_cumulative_features(current_standing))
        
        # 2. FORM FEATURES (derniers N matchs)
        features.update(self._get_form_features(
            team, gameweek, season, results_df
        ))
        
        # 3. HOME/AWAY SPLIT
        features.update(self._get_home_away_split(
            team, gameweek, results_df
        ))
        
        # 4. GOAL SCORING PATTERNS
        features.update(self._get_goal_patterns(
            team, gameweek, results_df
        ))
        
        # 5. MATCH STATS FEATURES (si disponibles)
        if stats_df is not None:
            features.update(self._get_match_stats_features(
                team, gameweek, results_df, stats_df
            ))
        
        # 6. REMAINING FIXTURES FEATURES
        features.update(self._get_remaining_fixtures_features(
            team, gameweek, standings_df
        ))
        
        # 7. TARGET: Points finaux de la saison
        final_standing = standings_df[
            (standings_df['team'] == team) & 
            (standings_df['gameweek'] == 38)
        ]
        
        if not final_standing.empty:
            features['target_final_points'] = final_standing.iloc[0]['points']
            features['target_final_position'] = final_standing.iloc[0]['position']
        else:
            features['target_final_points'] = None
            features['target_final_position'] = None
        
        return features
    
    def _get_cumulative_features(self, standing):
        """Features cumulatives depuis le début de la saison"""
        return {
            'current_position': standing['position'],
            'current_points': standing['points'],
            'matches_played': standing['played'],
            'wins': standing['won'],
            'draws': standing['drawn'],
            'losses': standing['lost'],
            'goals_for': standing['goals_for'],
            'goals_against': standing['goals_against'],
            'goal_difference': standing['goal_difference'],
            
            # Ratios
            'points_per_game': standing['points'] / standing['played'] if standing['played'] > 0 else 0,
            'win_rate': standing['won'] / standing['played'] if standing['played'] > 0 else 0,
            'goals_per_game': standing['goals_for'] / standing['played'] if standing['played'] > 0 else 0,
            'goals_conceded_per_game': standing['goals_against'] / standing['played'] if standing['played'] > 0 else 0,
        }
    
    def _get_form_features(self, team, gameweek, season, results_df):
        """Features basées sur la forme récente"""
        # Obtenir les derniers matchs
        team_matches = results_df[
            (results_df['gameweek'] < gameweek) &
            ((results_df['home_team'] == team) | (results_df['away_team'] == team))
        ].sort_values('gameweek', ascending=False).head(self.form_window)
        
        if team_matches.empty:
            return {
                f'form_last_{self.form_window}_points': 0,
                f'form_last_{self.form_window}_wins': 0,
                f'form_last_{self.form_window}_goals_for': 0,
                f'form_last_{self.form_window}_goals_against': 0,
            }
        
        points = 0
        wins = 0
        goals_for = 0
        goals_against = 0
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                goals_for += match['home_goals']
                goals_against += match['away_goals']
                if match['home_goals'] > match['away_goals']:
                    points += 3
                    wins += 1
                elif match['home_goals'] == match['away_goals']:
                    points += 1
            else:
                goals_for += match['away_goals']
                goals_against += match['home_goals']
                if match['away_goals'] > match['home_goals']:
                    points += 3
                    wins += 1
                elif match['away_goals'] == match['home_goals']:
                    points += 1
        
        return {
            f'form_last_{self.form_window}_points': points,
            f'form_last_{self.form_window}_wins': wins,
            f'form_last_{self.form_window}_goals_for': goals_for,
            f'form_last_{self.form_window}_goals_against': goals_against,
            f'form_last_{self.form_window}_goal_diff': goals_for - goals_against,
        }
    
    def _get_home_away_split(self, team, gameweek, results_df):
        """Performance à domicile vs à l'extérieur"""
        matches = results_df[
            (results_df['gameweek'] < gameweek) &
            ((results_df['home_team'] == team) | (results_df['away_team'] == team))
        ]
        
        home_matches = matches[matches['home_team'] == team]
        away_matches = matches[matches['away_team'] == team]
        
        # Stats domicile
        home_points = 0
        home_goals_for = 0
        home_goals_against = 0
        
        for _, match in home_matches.iterrows():
            home_goals_for += match['home_goals']
            home_goals_against += match['away_goals']
            if match['home_goals'] > match['away_goals']:
                home_points += 3
            elif match['home_goals'] == match['away_goals']:
                home_points += 1
        
        # Stats extérieur
        away_points = 0
        away_goals_for = 0
        away_goals_against = 0
        
        for _, match in away_matches.iterrows():
            away_goals_for += match['away_goals']
            away_goals_against += match['home_goals']
            if match['away_goals'] > match['home_goals']:
                away_points += 3
            elif match['away_goals'] == match['home_goals']:
                away_points += 1
        
        home_played = len(home_matches)
        away_played = len(away_matches)
        
        return {
            'home_points': home_points,
            'away_points': away_points,
            'home_ppg': home_points / home_played if home_played > 0 else 0,
            'away_ppg': away_points / away_played if away_played > 0 else 0,
            'home_goals_per_game': home_goals_for / home_played if home_played > 0 else 0,
            'away_goals_per_game': away_goals_for / away_played if away_played > 0 else 0,
        }
    
    def _get_goal_patterns(self, team, gameweek, results_df):
        """Patterns de buts (clean sheets, failing to score, etc.)"""
        matches = results_df[
            (results_df['gameweek'] < gameweek) &
            ((results_df['home_team'] == team) | (results_df['away_team'] == team))
        ]
        
        clean_sheets = 0
        failed_to_score = 0
        scored_2_plus = 0
        conceded_2_plus = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                goals_for = match['home_goals']
                goals_against = match['away_goals']
            else:
                goals_for = match['away_goals']
                goals_against = match['home_goals']
            
            if goals_against == 0:
                clean_sheets += 1
            if goals_for == 0:
                failed_to_score += 1
            if goals_for >= 2:
                scored_2_plus += 1
            if goals_against >= 2:
                conceded_2_plus += 1
        
        total_matches = len(matches)
        
        return {
            'clean_sheets': clean_sheets,
            'failed_to_score': failed_to_score,
            'clean_sheet_rate': clean_sheets / total_matches if total_matches > 0 else 0,
            'btts_rate': (total_matches - failed_to_score) / total_matches if total_matches > 0 else 0,
            'scored_2plus_rate': scored_2_plus / total_matches if total_matches > 0 else 0,
        }
    
    def _get_match_stats_features(self, team, gameweek, results_df, stats_df):
        """Features basées sur les statistiques détaillées des matchs"""
        # Obtenir les match_ids des matchs de cette équipe
        team_matches = results_df[
            (results_df['gameweek'] < gameweek) &
            ((results_df['home_team'] == team) | (results_df['away_team'] == team))
        ]
        
        if team_matches.empty:
            return {}
        
        match_ids = team_matches['match_id'].values
        
        # Filtrer les stats pour ces matchs
        team_stats = stats_df[stats_df['match_id'].isin(match_ids)]
        
        if team_stats.empty:
            return {}
        
        # Agréger les stats
        features = {}
        
        # Liste des stats à agréger (si disponibles)
        stats_to_aggregate = [
            'possession', 'total_scoring_att', 'ontarget_scoring_att',
            'total_pass', 'accurate_pass', 'total_tackle', 'fouls',
            'total_offside', 'won_contest', 'total_corners', 'saves'
        ]
        
        for stat in stats_to_aggregate:
            home_col = f'home_{stat}'
            away_col = f'away_{stat}'
            
            if home_col in team_stats.columns and away_col in team_stats.columns:
                # Calculer la moyenne pour cette équipe
                home_mask = team_stats['home_team'] == team
                away_mask = team_stats['away_team'] == team
                
                home_values = team_stats[home_mask][home_col].dropna()
                away_values = team_stats[away_mask][away_col].dropna()
                
                all_values = pd.concat([home_values, away_values])
                
                if len(all_values) > 0:
                    features[f'avg_{stat}'] = all_values.mean()
        
        return features
    
    def _get_remaining_fixtures_features(self, team, gameweek, standings_df):
        """Features basées sur les matchs restants"""
        matches_remaining = 38 - gameweek
        
        return {
            'matches_remaining': matches_remaining,
            'projected_points': None,  # Sera calculé plus tard avec un modèle
        }
    
    def create_features_for_all_seasons(self, seasons, raw_data_dir='data/raw',
                                       output_dir='data/processed/v1'):
        """
        Crée les features pour plusieurs saisons
        
        Args:
            seasons: Liste des saisons
            raw_data_dir: Dossier des données brutes
            output_dir: Dossier de sortie
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_features = []
        
        for season in seasons:
            season_features = self.create_features_for_season(season, raw_data_dir)
            all_features.append(season_features)
        
        # Combiner toutes les saisons
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Sauvegarder
        output_path = Path(output_dir) / 'features.parquet'
        combined_df.to_parquet(output_path, index=False)
        
        # Sauvegarder aussi en CSV pour inspection
        csv_path = Path(output_dir) / 'features.csv'
        combined_df.to_csv(csv_path, index=False)
        
        # Sauvegarder les métadonnées
        self._save_feature_metadata(combined_df, output_dir)
        
        print(f"\n{'='*70}")
        print("✅ FEATURE ENGINEERING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total records: {len(combined_df)}")
        print(f"Total features: {len(combined_df.columns)}")
        print(f"Seasons: {combined_df['season'].nunique()}")
        print(f"Teams: {combined_df['team'].nunique()}")
        print(f"\nOutput files:")
        print(f"  - {output_path}")
        print(f"  - {csv_path}")
        print(f"  - {output_dir}/feature_metadata.json")
        
        return combined_df
    
    def _save_feature_metadata(self, df, output_dir):
        """Sauvegarde les métadonnées des features"""
        metadata = {
            'created_at': datetime.now().isoformat(),
            'n_records': len(df),
            'n_features': len(df.columns),
            'seasons': df['season'].unique().tolist(),
            'teams': df['team'].unique().tolist(),
            'feature_names': df.columns.tolist(),
            'feature_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'form_window': self.form_window,
        }
        
        metadata_path = Path(output_dir) / 'feature_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Feature Engineering Pipeline')
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2015-2016', '2016-2017', '2017-2018', '2018-2019', 
                '2019-2020', '2020-2021', '2021-2022', '2022-2023'],
        help='Seasons to process'
    )
    parser.add_argument(
        '--raw-data-dir',
        default='data/raw',
        help='Directory containing raw data'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed/v1',
        help='Output directory for features'
    )
    parser.add_argument(
        '--form-window',
        type=int,
        default=5,
        help='Number of recent matches for form calculation'
    )
    
    args = parser.parse_args()
    
    # Créer le feature engineer
    engineer = FeatureEngineer(form_window=args.form_window)
    
    # Générer les features
    features_df = engineer.create_features_for_all_seasons(
        seasons=args.seasons,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir
    )
    
    print("\n🎉 Feature engineering completed successfully!")


if __name__ == '__main__':
    main()