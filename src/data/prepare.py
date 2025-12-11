"""
Data Preparation for Modeling
Splits data into train/validation/test sets and handles preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class DataPreparator:
    """
    Prépare les données pour le modeling :
    - Split train/val/test
    - Gestion des valeurs manquantes
    - Feature selection
    - Feature scaling
    """
    
    def __init__(self, test_size=0.15, val_size=0.15, random_state=42):
        """
        Args:
            test_size: Proportion pour le test set
            val_size: Proportion pour le validation set
            random_state: Seed pour reproductibilité
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_stats = {}
    
    def prepare_data(self, features_path, output_dir='data/processed/v1',
                    min_gameweek=10, max_missing_pct=30):
        """
        Prépare les données complètes pour le modeling
        
        Args:
            features_path: Chemin vers le fichier de features
            output_dir: Dossier de sortie
            min_gameweek: Gameweek minimum pour les prédictions (avoir assez d'historique)
            max_missing_pct: % maximum de valeurs manquantes acceptées par feature
        """
        print(f"\n{'='*70}")
        print("DATA PREPARATION FOR MODELING")
        print(f"{'='*70}\n")
        
        # 1. Charger les données
        print("📥 Loading features...")
        df = pd.read_parquet(features_path)
        print(f"   Loaded: {len(df):,} records, {len(df.columns)} features")
        
        # 2. Filtrer par gameweek minimum
        print(f"\n🔍 Filtering gameweek >= {min_gameweek}...")
        df_filtered = df[df['gameweek'] >= min_gameweek].copy()
        print(f"   Retained: {len(df_filtered):,} records ({len(df_filtered)/len(df)*100:.1f}%)")
        
        # 3. Gérer les valeurs manquantes
        print(f"\n🧹 Handling missing values...")
        df_clean, dropped_features = self._handle_missing_values(
            df_filtered, max_missing_pct
        )
        print(f"   Dropped {len(dropped_features)} features with >{max_missing_pct}% missing")
        print(f"   Remaining features: {len(df_clean.columns)}")
        
        # 4. Séparer features, metadata et target
        print(f"\n📊 Separating features, metadata and target...")
        metadata_cols = ['season', 'team', 'gameweek']
        target_col = 'target_final_points'
        
        feature_cols = [col for col in df_clean.columns 
                       if col not in metadata_cols + [target_col, 'target_final_position']]
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        metadata = df_clean[metadata_cols]
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Target: {target_col}")
        print(f"   Samples: {len(X):,}")
        
        # 5. Split par saison (time-based split)
        print(f"\n✂️  Splitting data (time-based)...")
        X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = \
            self._split_by_season(X, y, metadata)
        
        print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # 6. Imputation des valeurs manquantes restantes
        print(f"\n🔧 Imputing remaining missing values...")
        X_train_imp, X_val_imp, X_test_imp = self._impute_missing(
            X_train, X_val, X_test
        )
        
        # 7. Feature scaling
        print(f"\n📏 Scaling features...")
        X_train_scaled, X_val_scaled, X_test_scaled = self._scale_features(
            X_train_imp, X_val_imp, X_test_imp
        )
        
        # 8. Sauvegarder les datasets
        print(f"\n💾 Saving datasets...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les données préparées
        self._save_datasets(
            output_path,
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            meta_train, meta_val, meta_test,
            feature_cols
        )
        
        # Sauvegarder le scaler
        scaler_path = output_path / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        print(f"   ✓ Scaler saved to {scaler_path}")
        
        # Sauvegarder les métadonnées de préparation
        self._save_preparation_metadata(
            output_path, feature_cols, dropped_features,
            len(X_train), len(X_val), len(X_test)
        )
        
        print(f"\n{'='*70}")
        print("✅ DATA PREPARATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Output directory: {output_path}")
        print(f"Files created:")
        print(f"  - train.parquet, val.parquet, test.parquet")
        print(f"  - scaler.joblib")
        print(f"  - preparation_metadata.json")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_cols
        }
    
    def _handle_missing_values(self, df, max_missing_pct):
        """Supprime les features avec trop de valeurs manquantes"""
        missing_pct = (df.isnull().sum() / len(df) * 100)
        features_to_drop = missing_pct[missing_pct > max_missing_pct].index.tolist()
        
        # Ne pas supprimer les colonnes metadata et target
        protected_cols = ['season', 'team', 'gameweek', 'target_final_points', 'target_final_position']
        features_to_drop = [f for f in features_to_drop if f not in protected_cols]
        
        df_clean = df.drop(columns=features_to_drop)
        
        return df_clean, features_to_drop
    
    def _split_by_season(self, X, y, metadata):
        """
        Split time-based : 
        - Train: saisons anciennes
        - Val: saison intermédiaire
        - Test: saison la plus récente
        """
        # Combiner pour le split
        df_combined = pd.concat([metadata, X, y], axis=1)
        
        # Obtenir les saisons uniques triées
        seasons = sorted(df_combined['season'].unique())
        
        print(f"   Available seasons: {seasons}")
        
        # Stratégie de split :
        # - Test: dernière saison
        # - Val: avant-dernière saison
        # - Train: toutes les autres
        test_season = seasons[-1]
        val_season = seasons[-2]
        train_seasons = seasons[:-2]
        
        print(f"   Train seasons: {train_seasons}")
        print(f"   Val season: {val_season}")
        print(f"   Test season: {test_season}")
        
        # Séparer les données
        train_mask = df_combined['season'].isin(train_seasons)
        val_mask = df_combined['season'] == val_season
        test_mask = df_combined['season'] == test_season
        
        df_train = df_combined[train_mask]
        df_val = df_combined[val_mask]
        df_test = df_combined[test_mask]
        
        # Extraire X, y, metadata
        metadata_cols = ['season', 'team', 'gameweek']
        target_col = 'target_final_points'
        feature_cols = [col for col in df_combined.columns 
                       if col not in metadata_cols + [target_col, 'target_final_position']]
        
        X_train = df_train[feature_cols]
        X_val = df_val[feature_cols]
        X_test = df_test[feature_cols]
        
        y_train = df_train[target_col]
        y_val = df_val[target_col]
        y_test = df_test[target_col]
        
        meta_train = df_train[metadata_cols]
        meta_val = df_val[metadata_cols]
        meta_test = df_test[metadata_cols]
        
        return X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test
    
    def _impute_missing(self, X_train, X_val, X_test):
        """Impute les valeurs manquantes avec la médiane du train set"""
        # Calculer les médianes sur le train set
        medians = X_train.median()
        
        # Imputer
        X_train_imp = X_train.fillna(medians)
        X_val_imp = X_val.fillna(medians)
        X_test_imp = X_test.fillna(medians)
        
        # Vérifier
        remaining_na_train = X_train_imp.isnull().sum().sum()
        remaining_na_val = X_val_imp.isnull().sum().sum()
        remaining_na_test = X_test_imp.isnull().sum().sum()
        
        print(f"   Remaining NaN - Train: {remaining_na_train}, Val: {remaining_na_val}, Test: {remaining_na_test}")
        
        # Si encore des NaN, remplir avec 0
        if remaining_na_train > 0 or remaining_na_val > 0 or remaining_na_test > 0:
            X_train_imp = X_train_imp.fillna(0)
            X_val_imp = X_val_imp.fillna(0)
            X_test_imp = X_test_imp.fillna(0)
            print(f"   Filled remaining NaN with 0")
        
        return X_train_imp, X_val_imp, X_test_imp
    
    def _scale_features(self, X_train, X_val, X_test):
        """Scale les features avec StandardScaler"""
        # Fit sur train uniquement
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform val et test
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def _save_datasets(self, output_path, X_train, X_val, X_test,
                      y_train, y_val, y_test, meta_train, meta_val, meta_test,
                      feature_cols):
        """Sauvegarde les datasets train/val/test"""
        
        # Combiner features, target et metadata
        train_df = pd.concat([meta_train.reset_index(drop=True), 
                             X_train.reset_index(drop=True), 
                             y_train.reset_index(drop=True)], axis=1)
        val_df = pd.concat([meta_val.reset_index(drop=True), 
                           X_val.reset_index(drop=True), 
                           y_val.reset_index(drop=True)], axis=1)
        test_df = pd.concat([meta_test.reset_index(drop=True), 
                            X_test.reset_index(drop=True), 
                            y_test.reset_index(drop=True)], axis=1)
        
        # Sauvegarder en parquet
        train_df.to_parquet(output_path / 'train.parquet', index=False)
        val_df.to_parquet(output_path / 'val.parquet', index=False)
        test_df.to_parquet(output_path / 'test.parquet', index=False)
        
        print(f"   ✓ train.parquet: {len(train_df)} records")
        print(f"   ✓ val.parquet: {len(val_df)} records")
        print(f"   ✓ test.parquet: {len(test_df)} records")
    
    def _save_preparation_metadata(self, output_path, feature_cols, 
                                  dropped_features, n_train, n_val, n_test):
        """Sauvegarde les métadonnées de préparation"""
        from datetime import datetime
        
        metadata = {
            'created_at': datetime.now().isoformat(),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'dropped_features': dropped_features,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'train_pct': n_train / (n_train + n_val + n_test) * 100,
            'val_pct': n_val / (n_train + n_val + n_test) * 100,
            'test_pct': n_test / (n_train + n_val + n_test) * 100,
            'scaler': 'StandardScaler',
            'imputation_strategy': 'median'
        }
        
        with open(output_path / 'preparation_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✓ preparation_metadata.json")


def main():
    parser = argparse.ArgumentParser(description='Data Preparation for Modeling')
    parser.add_argument(
        '--features-path',
        default='data/processed/v1/features.parquet',
        help='Path to features file'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed/v1',
        help='Output directory'
    )
    parser.add_argument(
        '--min-gameweek',
        type=int,
        default=10,
        help='Minimum gameweek for predictions (to have enough history)'
    )
    parser.add_argument(
        '--max-missing-pct',
        type=float,
        default=30,
        help='Maximum missing percentage for features'
    )
    
    args = parser.parse_args()
    
    preparator = DataPreparator()
    preparator.prepare_data(
        features_path=args.features_path,
        output_dir=args.output_dir,
        min_gameweek=args.min_gameweek,
        max_missing_pct=args.max_missing_pct
    )


if __name__ == '__main__':
    main()