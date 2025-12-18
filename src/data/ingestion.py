"""
Module pour l'ingestion des données de la Premier League
Supporte deux modes:
- historical: Récupération de toutes les données historiques
- incremental: Récupération des nouveaux matchs uniquement
"""

import os
import json
import requests
import progressbar
import pandas as pd
import time
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class PremierLeagueAPI:
    """Client pour l'API Premier League"""
    
    def __init__(self):
        self.api_base = os.getenv('API_BASE_URL', 'https://footballapi.pulselive.com/football')
        self.origin = os.getenv('API_ORIGIN', 'https://www.premierleague.com')
        self.headers = {'Origin': self.origin}
    
    def get_team_ids(self, season_id):
        """Récupère les IDs des équipes pour une saison"""
        api = f'{self.api_base}/compseasons/{season_id}/teams'
        response = requests.get(api, headers=self.headers)
        teams = json.loads(response.text)
        team_ids = [team['id'] for team in teams]
        return ','.join(map(str, team_ids))
    
    def fetch_fixtures(self, season_id, team_ids, status='C'):
        """
        Récupère les matchs pour une saison
        status: 'C' (Complete), 'U' (Upcoming), 'L' (Live)
        """
        params = {
            'comps': '1',
            'compSeasons': season_id,
            'teams': team_ids,
            'page': '0',
            'pageSize': '380',
            'sort': 'asc',
            'statuses': status,
            'altIds': 'true'
        }
        response = requests.get(f'{self.api_base}/fixtures', params=params, headers=self.headers)
        return json.loads(response.text)
    
    def fetch_match_stats(self, match_id):
        """Récupère les statistiques détaillées d'un match"""
        try:
            response = requests.get(
                f"{self.api_base}/stats/match/{match_id}", 
                headers=self.headers, 
                timeout=10
            )
            
            if response.status_code != 200 or not response.text or response.text.strip() == '':
                return None
            
            data = json.loads(response.text)
            
            if 'data' not in data or 'entity' not in data:
                return None
            
            return data
        except Exception as e:
            print(f"Error fetching stats for match {match_id}: {e}")
            return None


class HistoricalDataFetcher:
    """Récupération des données historiques complètes"""
    
    # Mapping des saisons aux IDs de l'API
    SEASON_IDS = {
        '2015-2016': 42,
        '2016-2017': 54,
        '2017-2018': 79,
        '2018-2019': 210,
        '2019-2020': 274,
        '2020-2021': 363,
        '2021-2022': 418,
        '2022-2023': 489,
        '2023-2024': 578,
        '2024-2025': 719,
        '2025-2026': 777
    }
    
    def __init__(self, seasons=None):
        """
        Args:
            seasons: Liste des saisons à récupérer (ex: ['2015-2016', '2016-2017'])
                    Si None, récupère toutes les saisons disponibles
        """
        self.api = PremierLeagueAPI()
        if seasons:
            self.seasons = {s: self.SEASON_IDS[s] for s in seasons if s in self.SEASON_IDS}
        else:
            self.seasons = self.SEASON_IDS
    
    def fetch_all_data(self, output_dir='data/raw'):
        """Récupère toutes les données pour les saisons sélectionnées"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 🔍 Detect existing seasons
        existing_seasons = set()
        output_path = Path(output_dir)
        if output_path.exists():
            for season_dir in output_path.iterdir():
                if season_dir.is_dir() and (season_dir / 'results.csv').exists():
                    existing_seasons.add(season_dir.name)
        
        # Filter out already downloaded seasons
        seasons_to_fetch = {
            name: id for name, id in self.seasons.items() 
            if name not in existing_seasons
        }
        
        if existing_seasons:
            print(f"\n✅ Found {len(existing_seasons)} existing seasons, skipping them:")
            for season in sorted(existing_seasons):
                print(f"   • {season}")
        
        if not seasons_to_fetch:
            print(f"\n{'='*70}")
            print("✅ ALL SEASONS ALREADY DOWNLOADED - NOTHING TO DO!")
            print(f"{'='*70}")
            return True
        
        print(f"\n{'='*70}")
        print(f"FETCHING HISTORICAL DATA FOR {len(seasons_to_fetch)} NEW SEASONS")
        print(f"{'='*70}\n")
        
        for season_name, season_id in seasons_to_fetch.items():
            print(f"\n📅 Processing season: {season_name}")
            
            # Créer le dossier pour la saison
            season_dir = Path(output_dir) / season_name
            season_dir.mkdir(exist_ok=True)
            
            # 1. Récupérer les résultats des matchs
            print("  ├─ Fetching match results...")
            results_df = self._fetch_season_results(season_name, season_id)
            results_path = season_dir / 'results.csv'
            results_df.to_csv(results_path, index=False)
            print(f"  │  ✓ Saved {len(results_df)} matches to {results_path}")
            
            # 2. Récupérer les statistiques détaillées
            print("  ├─ Fetching match statistics...")
            stats_df = self._fetch_season_stats(results_df)
            if not stats_df.empty:
                stats_path = season_dir / 'match_stats.csv'
                stats_df.to_csv(stats_path, index=False)
                print(f"  │  ✓ Saved stats for {len(stats_df)} matches to {stats_path}")
            else:
                print(f"  │  ⚠️  No statistics available for {season_name}")
            
            # 3. Calculer les classements gameweek par gameweek
            print("  └─ Calculating standings...")
            standings_df = self._calculate_standings(results_df, season_name)
            standings_path = season_dir / 'standings.csv'
            standings_df.to_csv(standings_path, index=False)
            print(f"     ✓ Saved {len(standings_df)} standing records to {standings_path}")
            
            time.sleep(0.5)  # Pause entre les saisons
        
        print(f"\n{'='*70}")
        print("✅ HISTORICAL DATA FETCH COMPLETE!")
        print(f"{'='*70}")
        print(f"Data saved in: {output_dir}/")
        print(f"Seasons processed: {len(seasons_to_fetch)}")
        print(f"Seasons skipped: {len(existing_seasons)}")
        
        return True
    
    def _fetch_season_results(self, season_name, season_id):
        """Récupère les résultats d'une saison"""
        team_ids = self.api.get_team_ids(season_id)
        results = self.api.fetch_fixtures(season_id, team_ids, status='C')
        
        df_list = [
            [
                result['id'],
                result['gameweek']['gameweek'] if 'gameweek' in result else None,
                result['kickoff']['label'] if 'kickoff' in result else None,
                result['teams'][0]['team']['name'],
                result['teams'][1]['team']['name'],
                result['teams'][0]['score'],
                result['teams'][1]['score'],
                result['outcome']
            ]
            for result in results['content']
        ]
        
        df = pd.DataFrame(
            df_list,
            columns=['match_id', 'gameweek', 'kickoff', 'home_team', 'away_team',
                    'home_goals', 'away_goals', 'result']
        )
        
        return df
    
    def _fetch_season_stats(self, results_df):
        """Récupère les statistiques pour tous les matchs d'une saison"""
        match_ids = results_df['match_id'].astype(int).tolist()
        all_stats = []
        
        bar = progressbar.ProgressBar(maxval=len(match_ids))
        bar.start()
        
        for i, match_id in enumerate(match_ids):
            data = self.api.fetch_match_stats(match_id)
            if data:
                stats = self._parse_match_stats(match_id, data)
                if stats:
                    all_stats.append(stats)
            
            bar.update(i + 1)
            time.sleep(0.2)  # Rate limiting
        
        bar.finish()
        
        return pd.DataFrame(all_stats) if all_stats else pd.DataFrame()
    
    def _parse_match_stats(self, match_id, data):
        """Parse les statistiques d'un match"""
        try:
            teams = data['entity']['teams']
            if len(teams) < 2:
                return None
            
            home_team_id = str(teams[0]['team']['id'])
            away_team_id = str(teams[1]['team']['id'])
            home_team_name = teams[0]['team']['name']
            away_team_name = teams[1]['team']['name']
            
            stats_data = data['data']
            
            match_stats = {
                'match_id': match_id,
                'home_team': home_team_name,
                'away_team': away_team_name
            }
            
            # Statistiques de l'équipe à domicile
            if home_team_id in stats_data:
                team_data = stats_data[home_team_id]
                stats_list = team_data.get('M', team_data) if isinstance(team_data, dict) else team_data
                for stat in stats_list:
                    match_stats[f"home_{stat['name']}"] = stat['value']
            
            # Statistiques de l'équipe à l'extérieur
            if away_team_id in stats_data:
                team_data = stats_data[away_team_id]
                stats_list = team_data.get('M', team_data) if isinstance(team_data, dict) else team_data
                for stat in stats_list:
                    match_stats[f"away_{stat['name']}"] = stat['value']
            
            return match_stats
        except Exception as e:
            return None
    
    def _calculate_standings(self, matches_df, season):
        """Calcule les classements gameweek par gameweek"""
        all_standings = []
        max_gameweek = matches_df['gameweek'].max()
        
        for gameweek in range(1, int(max_gameweek) + 1):
            matches_so_far = matches_df[matches_df['gameweek'] <= gameweek]
            standings = self._calculate_gameweek_standings(matches_so_far, gameweek, season)
            all_standings.extend(standings)
        
        return pd.DataFrame(all_standings)
    
    def _calculate_gameweek_standings(self, matches_df, gameweek, season):
        """Calcule le classement après un gameweek spécifique"""
        teams = set(matches_df['home_team'].unique()) | set(matches_df['away_team'].unique())
        team_stats = {team: {
            'team': team,
            'season': season,
            'gameweek': gameweek,
            'played': 0,
            'won': 0,
            'drawn': 0,
            'lost': 0,
            'goals_for': 0,
            'goals_against': 0,
            'goal_difference': 0,
            'points': 0
        } for team in teams}
        
        for _, match in matches_df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            
            # Mise à jour des statistiques
            team_stats[home_team]['played'] += 1
            team_stats[home_team]['goals_for'] += home_goals
            team_stats[home_team]['goals_against'] += away_goals
            
            team_stats[away_team]['played'] += 1
            team_stats[away_team]['goals_for'] += away_goals
            team_stats[away_team]['goals_against'] += home_goals
            
            # Déterminer le résultat
            if home_goals > away_goals:
                team_stats[home_team]['won'] += 1
                team_stats[home_team]['points'] += 3
                team_stats[away_team]['lost'] += 1
            elif home_goals < away_goals:
                team_stats[away_team]['won'] += 1
                team_stats[away_team]['points'] += 3
                team_stats[home_team]['lost'] += 1
            else:
                team_stats[home_team]['drawn'] += 1
                team_stats[home_team]['points'] += 1
                team_stats[away_team]['drawn'] += 1
                team_stats[away_team]['points'] += 1
        
        # Calculer la différence de buts et trier
        standings = []
        for team, stats in team_stats.items():
            stats['goal_difference'] = stats['goals_for'] - stats['goals_against']
            standings.append(stats)
        
        standings.sort(key=lambda x: (x['points'], x['goal_difference'], x['goals_for']), reverse=True)
        
        # Ajouter la position
        for i, team_data in enumerate(standings, 1):
            team_data['position'] = i
        
        return standings


class IncrementalDataFetcher:
    """Récupération incrémentale des nouveaux matchs"""
    
    def __init__(self, season='2024-2025'):
        self.api = PremierLeagueAPI()
        self.season = season
        self.season_id = HistoricalDataFetcher.SEASON_IDS.get(season)
        
        if not self.season_id:
            raise ValueError(f"Season {season} not found in available seasons")
    
    def fetch_new_matches(self, output_dir='data/raw'):
        """Récupère les nouveaux matchs depuis la dernière extraction"""
        season_dir = Path(output_dir) / self.season
        season_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = season_dir / 'results.csv'
        
        # Charger les matchs existants
        if results_path.exists():
            existing_df = pd.read_csv(results_path)
            existing_ids = set(existing_df['match_id'].values)
            print(f"📊 Found {len(existing_ids)} existing matches")
        else:
            existing_df = pd.DataFrame()
            existing_ids = set()
            print("📊 No existing matches found - fetching all")
        
        # Récupérer tous les matchs terminés
        print(f"\n🔄 Fetching completed matches for {self.season}...")
        team_ids = self.api.get_team_ids(self.season_id)
        results = self.api.fetch_fixtures(self.season_id, team_ids, status='C')
        
        # Identifier les nouveaux matchs
        new_matches = []
        for result in results['content']:
            match_id = result['id']
            if match_id not in existing_ids:
                new_matches.append([
                    result['id'],
                    result['gameweek']['gameweek'] if 'gameweek' in result else None,
                    result['kickoff']['label'] if 'kickoff' in result else None,
                    result['teams'][0]['team']['name'],
                    result['teams'][1]['team']['name'],
                    result['teams'][0]['score'],
                    result['teams'][1]['score'],
                    result['outcome']
                ])
        
        if not new_matches:
            print("✓ No new matches found")
            return False
        
        print(f"✨ Found {len(new_matches)} new matches!")
        
        # Créer DataFrame des nouveaux matchs
        new_df = pd.DataFrame(
            new_matches,
            columns=['match_id', 'gameweek', 'kickoff', 'home_team', 'away_team',
                    'home_goals', 'away_goals', 'result']
        )
        
        # Combiner avec les matchs existants
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Sauvegarder
        combined_df.to_csv(results_path, index=False)
        print(f"💾 Saved {len(combined_df)} total matches to {results_path}")
        
        # Récupérer les stats pour les nouveaux matchs
        print("\n📈 Fetching statistics for new matches...")
        self._fetch_new_match_stats(new_df, season_dir)
        
        # Recalculer les standings
        print("\n📊 Recalculating standings...")
        fetcher = HistoricalDataFetcher([self.season])
        standings_df = fetcher._calculate_standings(combined_df, self.season)
        standings_df.to_csv(season_dir / 'standings.csv', index=False)
        print(f"✓ Updated standings")
        
        return True
    
    def _fetch_new_match_stats(self, new_matches_df, season_dir):
        """Récupère les stats pour les nouveaux matchs uniquement"""
        stats_path = season_dir / 'match_stats.csv'
        
        # Charger les stats existantes
        if stats_path.exists():
            existing_stats = pd.read_csv(stats_path)
            existing_match_ids = set(existing_stats['match_id'].values)
        else:
            existing_stats = pd.DataFrame()
            existing_match_ids = set()
        
        # Récupérer les stats pour les nouveaux matchs
        match_ids = new_matches_df['match_id'].astype(int).tolist()
        new_stats = []
        
        for match_id in match_ids:
            if match_id not in existing_match_ids:
                data = self.api.fetch_match_stats(match_id)
                if data:
                    fetcher = HistoricalDataFetcher()
                    stats = fetcher._parse_match_stats(match_id, data)
                    if stats:
                        new_stats.append(stats)
                time.sleep(0.2)
        
        if new_stats:
            new_stats_df = pd.DataFrame(new_stats)
            if not existing_stats.empty:
                combined_stats = pd.concat([existing_stats, new_stats_df], ignore_index=True)
            else:
                combined_stats = new_stats_df
            
            combined_stats.to_csv(stats_path, index=False)
            print(f"✓ Saved stats for {len(new_stats)} new matches")


def main():
    parser = argparse.ArgumentParser(description='Premier League Data Ingestion')
    parser.add_argument(
        '--mode',
        choices=['historical', 'incremental'],
        required=True,
        help='Mode: historical (all seasons) or incremental (new matches)'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        help='Seasons to fetch (e.g., 2015-2016 2016-2017). If not specified, fetches all.'
    )
    parser.add_argument(
        '--season',
        default='2024-2025',
        help='Season for incremental mode (default: 2024-2025)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/raw',
        help='Output directory (default: data/raw)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'historical':
        fetcher = HistoricalDataFetcher(seasons=args.seasons)
        fetcher.fetch_all_data(output_dir=args.output_dir)
    else:
        fetcher = IncrementalDataFetcher(season=args.season)
        fetcher.fetch_new_matches(output_dir=args.output_dir)


if __name__ == '__main__':
    main()