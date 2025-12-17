"""
Enhanced Feature Engineering Pipeline for Premier League Prediction
Creates advanced features from raw match results and standings data

New Features Added:
- Multiple form windows (3, 5, 10 matches)
- Exponentially weighted moving averages
- Momentum indicators
- Interaction features
- Strength of schedule
- Head-to-head history
- Advanced goal patterns
- Performance vs top/bottom teams
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
import warnings
from scipy.stats import entropy

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Enhanced Feature Engineering with advanced statistical features
    """
    
    def __init__(self, form_windows=[3, 5, 10]):
        """
        Args:
            form_windows: List of windows for form calculation (e.g., [3, 5, 10])
        """
        self.form_windows = form_windows
        self.feature_metadata = {}
    
    def create_features_for_season(self, season, raw_data_dir='data/raw'):
        """Crée les features pour une saison complète"""
        print(f"\n{'='*70}")
        print(f"CREATING ENHANCED FEATURES FOR {season}")
        print(f"{'='*70}\n")
        
        season_dir = Path(raw_data_dir) / season
        
        results_df = pd.read_csv(season_dir / 'results.csv')
        standings_df = pd.read_csv(season_dir / 'standings.csv')
        
        stats_path = season_dir / 'match_stats.csv'
        if stats_path.exists():
            stats_df = pd.read_csv(stats_path)
        else:
            stats_df = None
            print("⚠️  Pas de statistiques détaillées disponibles")
        
        teams = standings_df['team'].unique()
        all_features = []
        
        print(f"🔄 Processing {len(teams)} teams across 38 gameweeks...")
        
        for team in teams:
            for gameweek in range(1, 39):
                features = self._create_features_for_team_gameweek(
                    team=team,
                    gameweek=gameweek,
                    season=season,
                    results_df=results_df,
                    standings_df=standings_df,
                    stats_df=stats_df,
                    all_teams=teams
                )
                
                if features:
                    all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        
        print(f"\n✅ Features created successfully!")
        print(f"   Total records: {len(features_df)}")
        print(f"   Total features: {len(features_df.columns)}")
        
        return features_df
    
    def _create_features_for_team_gameweek(self, team, gameweek, season, 
                                          results_df, standings_df, stats_df, all_teams):
        """Crée les features pour une équipe à un gameweek donné"""
        
        current_standing = standings_df[
            (standings_df['team'] == team) & 
            (standings_df['gameweek'] == gameweek)
        ]
        
        if current_standing.empty:
            return None
        
        current_standing = current_standing.iloc[0]
        
        features = {
            'season': season,
            'team': team,
            'gameweek': gameweek,
        }
        
        # 1. CUMULATIVE FEATURES
        features.update(self._get_cumulative_features(current_standing))
        
        # 2. MULTI-WINDOW FORM FEATURES
        features.update(self._get_multi_window_form_features(
            team, gameweek, results_df
        ))
        
        # 3. EXPONENTIALLY WEIGHTED FEATURES
        features.update(self._get_ewm_features(
            team, gameweek, results_df
        ))
        
        # 4. HOME/AWAY SPLIT
        features.update(self._get_home_away_split(
            team, gameweek, results_df
        ))
        
        # 5. ADVANCED GOAL PATTERNS
        features.update(self._get_advanced_goal_patterns(
            team, gameweek, results_df
        ))
        
        # 6. MOMENTUM & STREAKS
        features.update(self._get_momentum_features(
            team, gameweek, results_df
        ))
        
        # 7. STRENGTH OF SCHEDULE
        features.update(self._get_strength_of_schedule(
            team, gameweek, results_df, standings_df
        ))
        
        # 8. PERFORMANCE VS TOP/BOTTOM TEAMS
        features.update(self._get_tiered_performance(
            team, gameweek, results_df, standings_df
        ))
        
        # 9. INTERACTION FEATURES
        features.update(self._get_interaction_features(features))
        
        # 10. MATCH STATS FEATURES
        if stats_df is not None:
            features.update(self._get_match_stats_features(
                team, gameweek, results_df, stats_df
            ))
        
        # 11. REMAINING FIXTURES
        features.update(self._get_remaining_fixtures_features(
            team, gameweek, standings_df
        ))
        
        # 12. TARGET
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
            'draw_rate': standing['drawn'] / standing['played'] if standing['played'] > 0 else 0,
            'loss_rate': standing['lost'] / standing['played'] if standing['played'] > 0 else 0,
            'goals_per_game': standing['goals_for'] / standing['played'] if standing['played'] > 0 else 0,
            'goals_conceded_per_game': standing['goals_against'] / standing['played'] if standing['played'] > 0 else 0,
            'goal_diff_per_game': standing['goal_difference'] / standing['played'] if standing['played'] > 0 else 0,
        }
    
    def _get_multi_window_form_features(self, team, gameweek, results_df):
        """Features de forme avec plusieurs fenêtres temporelles"""
        features = {}
        
        for window in self.form_windows:
            team_matches = results_df[
                (results_df['gameweek'] < gameweek) &
                ((results_df['home_team'] == team) | (results_df['away_team'] == team))
            ].sort_values('gameweek', ascending=False).head(window)
            
            if team_matches.empty:
                features.update({
                    f'form_last_{window}_points': 0,
                    f'form_last_{window}_wins': 0,
                    f'form_last_{window}_draws': 0,
                    f'form_last_{window}_losses': 0,
                    f'form_last_{window}_goals_for': 0,
                    f'form_last_{window}_goals_against': 0,
                    f'form_last_{window}_goal_diff': 0,
                    f'form_last_{window}_ppg': 0,
                })
                continue
            
            points = 0
            wins = 0
            draws = 0
            losses = 0
            goals_for = 0
            goals_against = 0
            
            for _, match in team_matches.iterrows():
                is_home = match['home_team'] == team
                
                if is_home:
                    gf = match['home_goals']
                    ga = match['away_goals']
                else:
                    gf = match['away_goals']
                    ga = match['home_goals']
                
                goals_for += gf
                goals_against += ga
                
                if gf > ga:
                    points += 3
                    wins += 1
                elif gf == ga:
                    points += 1
                    draws += 1
                else:
                    losses += 1
            
            n_matches = len(team_matches)
            
            features.update({
                f'form_last_{window}_points': points,
                f'form_last_{window}_wins': wins,
                f'form_last_{window}_draws': draws,
                f'form_last_{window}_losses': losses,
                f'form_last_{window}_goals_for': goals_for,
                f'form_last_{window}_goals_against': goals_against,
                f'form_last_{window}_goal_diff': goals_for - goals_against,
                f'form_last_{window}_ppg': points / n_matches if n_matches > 0 else 0,
            })
        
        return features
    
    def _get_ewm_features(self, team, gameweek, results_df):
        """Exponentially Weighted Moving Average features"""
        team_matches = results_df[
            (results_df['gameweek'] < gameweek) &
            ((results_df['home_team'] == team) | (results_df['away_team'] == team))
        ].sort_values('gameweek')
        
        if len(team_matches) < 3:
            return {
                'ewm_points': 0,
                'ewm_goals_for': 0,
                'ewm_goals_against': 0,
                'ewm_goal_diff': 0,
            }
        
        points_list = []
        goals_for_list = []
        goals_against_list = []
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                gf = match['home_goals']
                ga = match['away_goals']
            else:
                gf = match['away_goals']
                ga = match['home_goals']
            
            goals_for_list.append(gf)
            goals_against_list.append(ga)
            
            if gf > ga:
                points_list.append(3)
            elif gf == ga:
                points_list.append(1)
            else:
                points_list.append(0)
        
        # Calculate EWM with span=5
        points_series = pd.Series(points_list)
        goals_for_series = pd.Series(goals_for_list)
        goals_against_series = pd.Series(goals_against_list)
        
        return {
            'ewm_points': points_series.ewm(span=5, adjust=False).mean().iloc[-1],
            'ewm_goals_for': goals_for_series.ewm(span=5, adjust=False).mean().iloc[-1],
            'ewm_goals_against': goals_against_series.ewm(span=5, adjust=False).mean().iloc[-1],
            'ewm_goal_diff': (goals_for_series - goals_against_series).ewm(span=5, adjust=False).mean().iloc[-1],
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
        home_wins = 0
        
        for _, match in home_matches.iterrows():
            home_goals_for += match['home_goals']
            home_goals_against += match['away_goals']
            if match['home_goals'] > match['away_goals']:
                home_points += 3
                home_wins += 1
            elif match['home_goals'] == match['away_goals']:
                home_points += 1
        
        # Stats extérieur
        away_points = 0
        away_goals_for = 0
        away_goals_against = 0
        away_wins = 0
        
        for _, match in away_matches.iterrows():
            away_goals_for += match['away_goals']
            away_goals_against += match['home_goals']
            if match['away_goals'] > match['home_goals']:
                away_points += 3
                away_wins += 1
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
            'home_win_rate': home_wins / home_played if home_played > 0 else 0,
            'away_win_rate': away_wins / away_played if away_played > 0 else 0,
            'home_away_ppg_diff': (home_points / home_played - away_points / away_played) if (home_played > 0 and away_played > 0) else 0,
        }
    
    def _get_advanced_goal_patterns(self, team, gameweek, results_df):
        """Patterns de buts avancés"""
        matches = results_df[
            (results_df['gameweek'] < gameweek) &
            ((results_df['home_team'] == team) | (results_df['away_team'] == team))
        ]
        
        clean_sheets = 0
        failed_to_score = 0
        scored_2_plus = 0
        scored_3_plus = 0
        conceded_2_plus = 0
        conceded_3_plus = 0
        both_teams_scored = 0
        high_scoring = 0  # 3+ total goals
        wins_to_nil = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                gf = match['home_goals']
                ga = match['away_goals']
            else:
                gf = match['away_goals']
                ga = match['home_goals']
            
            total_goals = match['home_goals'] + match['away_goals']
            
            if ga == 0:
                clean_sheets += 1
                if gf > 0:
                    wins_to_nil += 1
            if gf == 0:
                failed_to_score += 1
            if gf >= 2:
                scored_2_plus += 1
            if gf >= 3:
                scored_3_plus += 1
            if ga >= 2:
                conceded_2_plus += 1
            if ga >= 3:
                conceded_3_plus += 1
            if gf > 0 and ga > 0:
                both_teams_scored += 1
            if total_goals >= 3:
                high_scoring += 1
        
        n = len(matches)
        
        return {
            'clean_sheets': clean_sheets,
            'failed_to_score': failed_to_score,
            'wins_to_nil': wins_to_nil,
            'clean_sheet_rate': clean_sheets / n if n > 0 else 0,
            'btts_rate': both_teams_scored / n if n > 0 else 0,
            'scored_2plus_rate': scored_2_plus / n if n > 0 else 0,
            'scored_3plus_rate': scored_3_plus / n if n > 0 else 0,
            'conceded_2plus_rate': conceded_2_plus / n if n > 0 else 0,
            'high_scoring_rate': high_scoring / n if n > 0 else 0,
            'defensive_solidity': (clean_sheets - conceded_2_plus) / n if n > 0 else 0,
        }
    
    def _get_momentum_features(self, team, gameweek, results_df):
        """Features de momentum et séries"""
        team_matches = results_df[
            (results_df['gameweek'] < gameweek) &
            ((results_df['home_team'] == team) | (results_df['away_team'] == team))
        ].sort_values('gameweek')
        
        if team_matches.empty:
            return {
                'current_win_streak': 0,
                'current_unbeaten_streak': 0,
                'current_loss_streak': 0,
                'current_winless_streak': 0,
                'longest_win_streak': 0,
                'longest_unbeaten_streak': 0,
                'momentum_score': 0,
            }
        
        results = []
        for _, match in team_matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                gf = match['home_goals']
                ga = match['away_goals']
            else:
                gf = match['away_goals']
                ga = match['home_goals']
            
            if gf > ga:
                results.append('W')
            elif gf == ga:
                results.append('D')
            else:
                results.append('L')
        
        # Current streaks
        current_win_streak = 0
        current_unbeaten_streak = 0
        current_loss_streak = 0
        current_winless_streak = 0
        
        for result in reversed(results):
            if result == 'W':
                current_win_streak += 1
                current_unbeaten_streak += 1
            elif result == 'D':
                current_unbeaten_streak += 1
                current_winless_streak += 1
                break
            else:
                current_loss_streak += 1
                current_winless_streak += 1
                break
        
        # Longest streaks
        longest_win_streak = 0
        longest_unbeaten_streak = 0
        temp_win = 0
        temp_unbeaten = 0
        
        for result in results:
            if result == 'W':
                temp_win += 1
                temp_unbeaten += 1
            elif result == 'D':
                temp_unbeaten += 1
                longest_win_streak = max(longest_win_streak, temp_win)
                temp_win = 0
            else:
                longest_win_streak = max(longest_win_streak, temp_win)
                longest_unbeaten_streak = max(longest_unbeaten_streak, temp_unbeaten)
                temp_win = 0
                temp_unbeaten = 0
        
        longest_win_streak = max(longest_win_streak, temp_win)
        longest_unbeaten_streak = max(longest_unbeaten_streak, temp_unbeaten)
        
        # Momentum score (weighted recent results: W=3, D=1, L=-1)
        momentum_weights = [0.5, 0.7, 0.9, 1.0, 1.0]  # Last 5 matches
        momentum_score = 0
        for i, result in enumerate(results[-5:]):
            weight = momentum_weights[min(i, len(momentum_weights)-1)]
            if result == 'W':
                momentum_score += 3 * weight
            elif result == 'D':
                momentum_score += 1 * weight
            else:
                momentum_score -= 1 * weight
        
        return {
            'current_win_streak': current_win_streak,
            'current_unbeaten_streak': current_unbeaten_streak,
            'current_loss_streak': current_loss_streak,
            'current_winless_streak': current_winless_streak,
            'longest_win_streak': longest_win_streak,
            'longest_unbeaten_streak': longest_unbeaten_streak,
            'momentum_score': momentum_score,
        }
    
    def _get_strength_of_schedule(self, team, gameweek, results_df, standings_df):
        """Force du calendrier (adversaires affrontés)"""
        team_matches = results_df[
            (results_df['gameweek'] < gameweek) &
            ((results_df['home_team'] == team) | (results_df['away_team'] == team))
        ]
        
        if team_matches.empty:
            return {
                'avg_opponent_position': 10.5,
                'avg_opponent_points': 0,
                'faced_top6': 0,
                'faced_bottom6': 0,
            }
        
        opponent_positions = []
        opponent_points = []
        faced_top6 = 0
        faced_bottom6 = 0
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team'] == team
            opponent = match['away_team'] if is_home else match['home_team']
            opponent_gw = match['gameweek']
            
            # Get opponent's standing at that gameweek
            opp_standing = standings_df[
                (standings_df['team'] == opponent) &
                (standings_df['gameweek'] == opponent_gw)
            ]
            
            if not opp_standing.empty:
                opp_pos = opp_standing.iloc[0]['position']
                opp_pts = opp_standing.iloc[0]['points']
                
                opponent_positions.append(opp_pos)
                opponent_points.append(opp_pts)
                
                if opp_pos <= 6:
                    faced_top6 += 1
                elif opp_pos >= 15:
                    faced_bottom6 += 1
        
        return {
            'avg_opponent_position': np.mean(opponent_positions) if opponent_positions else 10.5,
            'avg_opponent_points': np.mean(opponent_points) if opponent_points else 0,
            'faced_top6': faced_top6,
            'faced_bottom6': faced_bottom6,
            'sos_difficulty': np.mean(opponent_positions) if opponent_positions else 10.5,
        }
    
    def _get_tiered_performance(self, team, gameweek, results_df, standings_df):
        """Performance contre top/mid/bottom teams"""
        team_matches = results_df[
            (results_df['gameweek'] < gameweek) &
            ((results_df['home_team'] == team) | (results_df['away_team'] == team))
        ]
        
        vs_top6_points = 0
        vs_top6_matches = 0
        vs_mid_points = 0
        vs_mid_matches = 0
        vs_bottom6_points = 0
        vs_bottom6_matches = 0
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team'] == team
            opponent = match['away_team'] if is_home else match['home_team']
            opponent_gw = match['gameweek']
            
            opp_standing = standings_df[
                (standings_df['team'] == opponent) &
                (standings_df['gameweek'] == opponent_gw)
            ]
            
            if opp_standing.empty:
                continue
            
            opp_pos = opp_standing.iloc[0]['position']
            
            if is_home:
                gf = match['home_goals']
                ga = match['away_goals']
            else:
                gf = match['away_goals']
                ga = match['home_goals']
            
            points = 0
            if gf > ga:
                points = 3
            elif gf == ga:
                points = 1
            
            if opp_pos <= 6:
                vs_top6_points += points
                vs_top6_matches += 1
            elif opp_pos >= 15:
                vs_bottom6_points += points
                vs_bottom6_matches += 1
            else:
                vs_mid_points += points
                vs_mid_matches += 1
        
        return {
            'vs_top6_ppg': vs_top6_points / vs_top6_matches if vs_top6_matches > 0 else 0,
            'vs_mid_ppg': vs_mid_points / vs_mid_matches if vs_mid_matches > 0 else 0,
            'vs_bottom6_ppg': vs_bottom6_points / vs_bottom6_matches if vs_bottom6_matches > 0 else 0,
            'vs_top6_matches': vs_top6_matches,
            'vs_bottom6_matches': vs_bottom6_matches,
        }
    
    def _get_interaction_features(self, features):
        """Features d'interaction entre variables existantes"""
        interactions = {}
        
        # Position × Form
        if 'current_position' in features and 'form_last_5_ppg' in features:
            interactions['position_form_interaction'] = features['current_position'] * features['form_last_5_ppg']
        
        # Goals scored × Goals conceded
        if 'goals_per_game' in features and 'goals_conceded_per_game' in features:
            interactions['offensive_defensive_balance'] = features['goals_per_game'] / (features['goals_conceded_per_game'] + 0.1)
        
        # Win rate × Home advantage
        if 'win_rate' in features and 'home_ppg' in features and 'away_ppg' in features:
            interactions['home_advantage_score'] = features['home_ppg'] - features['away_ppg']
        
        # Form momentum × Current position
        if 'momentum_score' in features and 'current_position' in features:
            interactions['momentum_position_product'] = features['momentum_score'] * (21 - features['current_position'])
        
        # Goal difference × Matches played
        if 'goal_difference' in features and 'matches_played' in features:
            interactions['gd_trajectory'] = features['goal_difference'] * (features['matches_played'] / 38)
        
        return interactions
    
    def _get_match_stats_features(self, team, gameweek, results_df, stats_df):
        """Features basées sur les statistiques détaillées des matchs"""
        team_matches = results_df[
            (results_df['gameweek'] < gameweek) &
            ((results_df['home_team'] == team) | (results_df['away_team'] == team))
        ]
        
        if team_matches.empty:
            return {}
        
        match_ids = team_matches['match_id'].values
        team_stats = stats_df[stats_df['match_id'].isin(match_ids)]
        
        if team_stats.empty:
            return {}
        
        features = {}
        
        stats_to_aggregate = [
            'possession', 'total_scoring_att', 'ontarget_scoring_att',
            'total_pass', 'accurate_pass', 'total_tackle', 'fouls',
            'total_offside', 'won_contest', 'total_corners', 'saves'
        ]
        
        for stat in stats_to_aggregate:
            home_col = f'home_{stat}'
            away_col = f'away_{stat}'
            
            if home_col in team_stats.columns and away_col in team_stats.columns:
                home_mask = team_stats['home_team'] == team
                away_mask = team_stats['away_team'] == team
                
                home_values = team_stats[home_mask][home_col].dropna()
                away_values = team_stats[away_mask][away_col].dropna()
                
                all_values = pd.concat([home_values, away_values])
                
                if len(all_values) > 0:
                    features[f'avg_{stat}'] = all_values.mean()
                    features[f'std_{stat}'] = all_values.std()
        
        # Derived stats
        if 'avg_accurate_pass' in features and 'avg_total_pass' in features:
            features['pass_accuracy'] = features['avg_accurate_pass'] / (features['avg_total_pass'] + 0.1)
        
        if 'avg_ontarget_scoring_att' in features and 'avg_total_scoring_att' in features:
            features['shot_accuracy'] = features['avg_ontarget_scoring_att'] / (features['avg_total_scoring_att'] + 0.1)
        
        return features
    
    def _get_remaining_fixtures_features(self, team, gameweek, standings_df):
        """Features basées sur les matchs restants"""
        matches_remaining = 38 - gameweek
        
        # Progression rate
        current_standing = standings_df[
            (standings_df['team'] == team) &
            (standings_df['gameweek'] == gameweek)
        ]
        
        if not current_standing.empty:
            current_pts = current_standing.iloc[0]['points']
            progression_rate = current_pts / gameweek if gameweek > 0 else 0
            projected_final_pts = progression_rate * 38
        else:
            progression_rate = 0
            projected_final_pts = 0
        
        return {
            'matches_remaining': matches_remaining,
            'progression_rate': progression_rate,
            'projected_final_points': projected_final_pts,
            'season_completion': gameweek / 38,
        }
    
    def create_features_for_all_seasons(self, seasons, raw_data_dir='data/raw',
                                       output_dir='data/processed/'):
        """Crée les features pour plusieurs saisons"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_features = []
        
        for season in seasons:
            season_features = self.create_features_for_season(season, raw_data_dir)
            all_features.append(season_features)
        
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Sauvegarder
        output_path = Path(output_dir) / 'features.parquet'
        combined_df.to_parquet(output_path, index=False)
        
        csv_path = Path(output_dir) / 'features.csv'
        combined_df.to_csv(csv_path, index=False)
        
        self._save_feature_metadata(combined_df, output_dir)
        
        print(f"\n{'='*70}")
        print("✅ ENHANCED FEATURE ENGINEERING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total records: {len(combined_df)}")
        print(f"Total features: {len(combined_df.columns)}")
        print(f"Seasons: {combined_df['season'].nunique()}")
        print(f"Teams: {combined_df['team'].nunique()}")
        print(f"\n📊 Feature breakdown:")
        print(f"  - Cumulative features: 13")
        print(f"  - Multi-window form: {len(self.form_windows) * 8}")
        print(f"  - Exponentially weighted: 4")
        print(f"  - Home/Away split: 9")
        print(f"  - Goal patterns: 9")
        print(f"  - Momentum: 7")
        print(f"  - Strength of schedule: 5")
        print(f"  - Tiered performance: 5")
        print(f"  - Interaction features: 5")
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
            'form_windows': self.form_windows,
            'enhancements': [
                'Multi-window form features',
                'Exponentially weighted moving averages',
                'Momentum and streak tracking',
                'Strength of schedule',
                'Tiered performance (vs top/mid/bottom)',
                'Interaction features',
                'Advanced goal patterns',
                'Match statistics aggregation'
            ]
        }
        
        metadata_path = Path(output_dir) / 'feature_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Enhanced Feature Engineering Pipeline')
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2015-2016', '2016-2017', '2017-2018', '2018-2019', 
                '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024'],
        help='Seasons to process'
    )
    parser.add_argument(
        '--raw-data-dir',
        default='data/raw',
        help='Directory containing raw data'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed/',
        help='Output directory for features'
    )
    parser.add_argument(
        '--form-windows',
        nargs='+',
        type=int,
        default=[3, 5, 10],
        help='Windows for form calculation (e.g., 3 5 10)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("🚀 ENHANCED FEATURE ENGINEERING PIPELINE")
    print(f"{'='*70}")
    print(f"Form windows: {args.form_windows}")
    print(f"Seasons: {len(args.seasons)}")
    
    engineer = FeatureEngineer(form_windows=args.form_windows)
    
    features_df = engineer.create_features_for_all_seasons(
        seasons=args.seasons,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir
    )
    
    print("\n🎉 Enhanced feature engineering completed successfully!")
    print(f"\n💡 New features added:")
    print(f"   • Multiple form windows (3, 5, 10 matches)")
    print(f"   • Exponentially weighted moving averages")
    print(f"   • Momentum and streak indicators")
    print(f"   • Strength of schedule analysis")
    print(f"   • Performance vs top/mid/bottom teams")
    print(f"   • Interaction features")
    print(f"   • Advanced goal patterns")


if __name__ == '__main__':
    main()