"""
DeepChecks Data & Model Validation
Version compatible avec DeepChecks 0.17+
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse
import logging

# DeepChecks imports - uniquement les checks disponibles
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import (
    full_suite,
    train_test_validation,
    data_integrity,
)

# Checks individuels disponibles
from deepchecks.tabular.checks import (
    # Data Integrity
    MixedNulls,
    StringMismatch,
    MixedDataTypes,
    IsSingleValue,
    DataDuplicates,
    
    # Train-Test Validation
    TrainTestFeatureDrift,
    TrainTestLabelDrift,
    DatasetsSizeComparison,
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DeepChecksValidator:
    """
    Validation avec DeepChecks (version compatible)
    - Data Integrity : Qualité des données
    - Train-Test Validation : Cohérence train/test
    """
    
    def __init__(self, reports_dir='reports/deepchecks'):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure le logging"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/deepchecks_validation.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_dataset(self, df: pd.DataFrame, label: str = 'target_final_points', 
                      features: list = None, cat_features: list = None) -> Dataset:
        """
        Crée un Dataset DeepChecks
        
        Args:
            df: DataFrame
            label: Nom de la colonne target
            features: Liste des features (optionnel, auto-détecté)
            cat_features: Features catégorielles
        """
        if features is None:
            # Auto-détection : toutes les colonnes sauf target et métadata
            exclude_cols = [label, 'season', 'team', 'gameweek', 'match_id']
            features = [col for col in df.columns if col not in exclude_cols]
        
        if cat_features is None:
            # Auto-détection des catégorielles
            cat_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.logger.info(f"Dataset créé : {len(df)} lignes, {len(features)} features, {len(cat_features)} catégorielles")
        
        return Dataset(
            df=df,
            label=label,
            features=features,
            cat_features=cat_features
        )
    
    # ============================================================================
    # 1. DATA INTEGRITY CHECKS
    # ============================================================================
    
    def check_data_integrity(self, df: pd.DataFrame, label: str = 'target_final_points') -> dict:
        """
        Vérifie l'intégrité des données
        
        Checks:
        - Mixed nulls
        - String mismatch
        - Mixed data types
        - Single value columns
        - Duplicates
        """
        self.logger.info("="*70)
        self.logger.info("[1/2] Running Data Integrity Checks...")
        self.logger.info("="*70)
        
        dataset = self.create_dataset(df, label)
        
        # Suite d'intégrité complète
        suite = data_integrity()
        result = suite.run(dataset)
        
        # Sauvegarder le rapport
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.reports_dir / f"data_integrity_{timestamp}.html"
        result.save_as_html(str(html_path))
        
        self.logger.info(f"   Report saved: {html_path}")
        
        # Extraire les résultats
        summary = self._extract_check_results(result)
        
        self.logger.info(f"   Checks passed: {summary['n_passed']}/{summary['n_checks']}")
        
        return {
            'check_type': 'data_integrity',
            'timestamp': datetime.now().isoformat(),
            'report_path': str(html_path),
            'summary': summary,
            'success': True
        }
    
    # ============================================================================
    # 2. TRAIN-TEST VALIDATION
    # ============================================================================
    
    def validate_train_test_split(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  label: str = 'target_final_points') -> dict:
        """
        Valide la cohérence entre train et test
        
        Checks:
        - Feature drift entre train et test
        - Label drift
        - Dataset size comparison
        """
        self.logger.info("="*70)
        self.logger.info("[2/2] Running Train-Test Validation...")
        self.logger.info("="*70)
        
        train_ds = self.create_dataset(train_df, label)
        test_ds = self.create_dataset(test_df, label)
        
        # Suite de validation train-test
        suite = train_test_validation()
        result = suite.run(train_dataset=train_ds, test_dataset=test_ds)
        
        # Sauvegarder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.reports_dir / f"train_test_validation_{timestamp}.html"
        result.save_as_html(str(html_path))
        
        self.logger.info(f"   Report saved: {html_path}")
        
        summary = self._extract_check_results(result)
        
        self.logger.info(f"   Checks passed: {summary['n_passed']}/{summary['n_checks']}")
        
        return {
            'check_type': 'train_test_validation',
            'timestamp': datetime.now().isoformat(),
            'report_path': str(html_path),
            'summary': summary,
            'success': True
        }
    
    # ============================================================================
    # 3. CHECKS INDIVIDUELS
    # ============================================================================
    
    def check_feature_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           label: str = 'target_final_points',
                           drift_threshold: float = 0.15) -> dict:
        """
        Vérifie le drift des features entre train et test
        """
        self.logger.info("Running Feature Drift Check...")
        
        train_ds = self.create_dataset(train_df, label)
        test_ds = self.create_dataset(test_df, label)
        
        # Check spécifique de drift
        check = TrainTestFeatureDrift()
        result = check.run(train_ds, test_ds)
        
        # Extraire les features qui driftent
        drift_score = result.value if result.value else {}
        
        if isinstance(drift_score, dict):
            drifted_features = [
                feature for feature, score in drift_score.items()
                if isinstance(score, (int, float)) and score > drift_threshold
            ]
        else:
            drifted_features = []
        
        self.logger.info(f"   Drifted features: {len(drifted_features)}")
        
        return {
            'check_type': 'feature_drift',
            'timestamp': datetime.now().isoformat(),
            'drift_threshold': float(drift_threshold),
            'drifted_features': drifted_features,
            'n_drifted': int(len(drifted_features)),
            'success': True
        }
    
    def check_label_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                         label: str = 'target_final_points') -> dict:
        """
        Vérifie le drift du target (label)
        """
        self.logger.info("Running Label Drift Check...")
        
        train_ds = self.create_dataset(train_df, label)
        test_ds = self.create_dataset(test_df, label)
        
        check = TrainTestLabelDrift()
        result = check.run(train_ds, test_ds)
        
        drift_score = result.value.get('Drift score', 0) if isinstance(result.value, dict) else 0
        drift_detected = bool(drift_score > 0.1)
        
        self.logger.info(f"   Label drift: {'DETECTED' if drift_detected else 'OK'}")
        
        return {
            'check_type': 'label_drift',
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'drift_score': float(drift_score),
            'success': True
        }
    
    # ============================================================================
    # 4. SUITE COMPLÈTE
    # ============================================================================
    
    def run_full_validation(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           label: str = 'target_final_points') -> dict:
        """
        Lance toutes les validations en une seule fois
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("RUNNING FULL DEEPCHECKS VALIDATION")
        self.logger.info("="*70)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'train_size': int(len(train_df)),
            'test_size': int(len(test_df)),
            'checks': []
        }
        
        # 1. Data Integrity
        try:
            integrity_result = self.check_data_integrity(train_df, label)
            results['checks'].append(integrity_result)
            self.logger.info("[OK] Data integrity check completed")
        except Exception as e:
            self.logger.error(f"[ERROR] Data Integrity failed: {e}")
            results['checks'].append({'check_type': 'data_integrity', 'success': False, 'error': str(e)})
        
        # 2. Train-Test Validation
        try:
            validation_result = self.validate_train_test_split(train_df, test_df, label)
            results['checks'].append(validation_result)
            self.logger.info("[OK] Train-test validation completed")
        except Exception as e:
            self.logger.error(f"[ERROR] Train-Test Validation failed: {e}")
            results['checks'].append({'check_type': 'train_test', 'success': False, 'error': str(e)})
        
        # 3. Feature Drift
        try:
            drift_result = self.check_feature_drift(train_df, test_df, label)
            results['checks'].append(drift_result)
            self.logger.info("[OK] Feature drift check completed")
        except Exception as e:
            self.logger.error(f"[ERROR] Feature Drift failed: {e}")
            results['checks'].append({'check_type': 'feature_drift', 'success': False, 'error': str(e)})
        
        # 4. Label Drift
        try:
            label_drift_result = self.check_label_drift(train_df, test_df, label)
            results['checks'].append(label_drift_result)
            self.logger.info("[OK] Label drift check completed")
        except Exception as e:
            self.logger.error(f"[ERROR] Label Drift failed: {e}")
            results['checks'].append({'check_type': 'label_drift', 'success': False, 'error': str(e)})
        
        # Sauvegarder résumé avec NumpyEncoder
        summary_path = self.reports_dir / "validation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        self.logger.info("\n" + "="*70)
        self.logger.info(f"[SUCCESS] Full validation complete! Summary: {summary_path}")
        self.logger.info("="*70)
        
        return results
    
    # ============================================================================
    # HELPERS
    # ============================================================================
    
    def _extract_check_results(self, suite_result) -> dict:
        """
        Extrait un résumé des résultats d'une suite
        Gère les CheckFailure objects correctement
        """
        summary = {
            'n_checks': len(suite_result.results),
            'n_passed': 0,
            'n_failed': 0,
            'failed_checks': []
        }
        
        for check_result in suite_result.results:
            # Vérifier si c'est un CheckFailure
            if hasattr(check_result, 'check'):
                # C'est un résultat normal
                try:
                    if hasattr(check_result, 'passed_conditions') and check_result.passed_conditions():
                        summary['n_passed'] += 1
                    else:
                        summary['n_failed'] += 1
                        summary['failed_checks'].append(str(check_result.header))
                except AttributeError:
                    # Si passed_conditions() n'existe pas, considérer comme échec
                    summary['n_failed'] += 1
                    summary['failed_checks'].append(str(check_result.header))
            else:
                # C'est probablement un CheckFailure - compter comme échec
                summary['n_failed'] += 1
                summary['failed_checks'].append(str(getattr(check_result, 'header', 'Unknown Check')))
        
        summary['success_rate'] = float(summary['n_passed'] / summary['n_checks']) if summary['n_checks'] > 0 else 0.0
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='DeepChecks Validation')
    parser.add_argument('--data-dir', default='data/processed/v1', help='Data directory')
    parser.add_argument('--label', default='target_final_points', help='Target column')
    parser.add_argument('--mode', choices=['integrity', 'train-test', 'full'], 
                       default='full', help='Validation mode')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DEEPCHECKS VALIDATOR - STARTING")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Target column: {args.label}")
    print("="*70 + "\n")
    
    # Charger les données
    data_dir = Path(args.data_dir)
    
    train_path = data_dir / 'train.parquet'
    test_path = data_dir / 'test.parquet'
    
    if not train_path.exists():
        print(f"[ERROR] Train file not found: {train_path}")
        return
    
    if not test_path.exists():
        print(f"[ERROR] Test file not found: {test_path}")
        return
    
    print(f"Loading train data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"  {len(train_df):,} rows, {len(train_df.columns)} columns")
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_parquet(test_path)
    print(f"  {len(test_df):,} rows, {len(test_df.columns)} columns\n")
    
    # Créer le validateur
    validator = DeepChecksValidator()
    
    # Lancer la validation
    if args.mode == 'integrity':
        result = validator.check_data_integrity(train_df, args.label)
    elif args.mode == 'train-test':
        result = validator.validate_train_test_split(train_df, test_df, args.label)
    else:  # full
        result = validator.run_full_validation(train_df, test_df, label=args.label)
    
    print("\n" + "="*70)
    print("DEEPCHECKS VALIDATION COMPLETE")
    print("="*70)
    print(f"Check reports saved to: reports/deepchecks/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()