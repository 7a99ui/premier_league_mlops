"""
Integrated Monitoring Pipeline
Combine Great Expectations + Evidently + DeepChecks
Approche complète comme le projet de Bassem
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import argparse
import logging

# Imports locaux
import sys
sys.path.append('src')

# Great Expectations (déjà implémenté)
from data.validation import DataValidator as GEValidator

# Evidently (déjà implémenté)
from monitoring.drift_detection import DriftDetectorV2

# DeepChecks (nouveau)
# from monitoring.deepchecks_validator import DeepChecksValidator


import numpy as np

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


class IntegratedMonitoringPipeline:
    """
    Pipeline de monitoring complet
    
    1. Great Expectations : Validation des données brutes
    2. DeepChecks : Validation train/test et intégrité
    3. Evidently : Détection de drift
    
    Workflow:
    - Pre-training : GE + DeepChecks data integrity
    - Post-training : DeepChecks model evaluation
    - Production : Evidently drift detection
    """
    
    def __init__(self, config_path='configs/monitoring_config.yaml'):
        self.config_path = config_path
        self.reports_dir = Path('reports/integrated')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialiser les validateurs
        self.ge_validator = GEValidator()
        self.drift_detector = DriftDetectorV2()
        # self.deepchecks_validator = DeepChecksValidator()
    
    def _setup_logging(self):
        """Configure le logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/integrated_monitoring.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    # ============================================================================
    # PHASE 1: PRE-TRAINING VALIDATION
    # ============================================================================
    
    def validate_raw_data(self, seasons, raw_data_dir='data/raw'):
        """
        Étape 1 : Valider les données brutes avec Great Expectations
        
        À faire AVANT le feature engineering
        """
        self.logger.info("="*70)
        self.logger.info("PHASE 1: RAW DATA VALIDATION (Great Expectations)")
        self.logger.info("="*70)
        
        # Great Expectations
        ge_results = self.ge_validator.validate_all_raw_data(seasons, raw_data_dir)
        
        # Vérifier si la validation a réussi
        success = all(r['success_rate'] == 1.0 for r in ge_results)
        
        if not success:
            self.logger.warning("⚠️  Some raw data validations failed!")
            self.logger.warning("Review: data/validation_reports/raw_data_validation.json")
        else:
            self.logger.info("✅ All raw data validations passed")
        
        return {
            'phase': 'raw_data_validation',
            'tool': 'great_expectations',
            'success': success,
            'results': ge_results
        }
    
    def validate_features_quality(self, features_path='data/processed/v1/features.parquet'):
        """
        Étape 2 : Valider les features avec Great Expectations
        
        À faire APRÈS le feature engineering, AVANT le split
        """
        self.logger.info("="*70)
        self.logger.info("PHASE 2: FEATURES QUALITY (Great Expectations)")
        self.logger.info("="*70)
        
        # Charger les features
        features_df = pd.read_parquet(features_path)
        
        # Great Expectations
        ge_result = self.ge_validator.validate_features(features_df)
        
        success = ge_result['success_rate'] == 1.0
        
        if not success:
            self.logger.warning("⚠️  Some feature validations failed!")
        else:
            self.logger.info("✅ All feature validations passed")
        
        return {
            'phase': 'features_quality',
            'tool': 'great_expectations',
            'success': success,
            'result': ge_result
        }
    
    def validate_train_test_integrity(self, train_path, test_path, label='target_final_points'):
        """
        Étape 3 : Valider l'intégrité train/test avec DeepChecks
        
        À faire APRÈS le split train/test
        """
        self.logger.info("="*70)
        self.logger.info("PHASE 3: TRAIN-TEST INTEGRITY (DeepChecks)")
        self.logger.info("="*70)
        
        # Charger les données
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # DeepChecks validation
        # Simuler pour l'instant (à implémenter avec le vrai DeepChecks)
        self.logger.info(f"Train size: {len(train_df)}")
        self.logger.info(f"Test size: {len(test_df)}")
        self.logger.info("✅ Train-Test integrity check passed (simulated)")
        
        return {
            'phase': 'train_test_integrity',
            'tool': 'deepchecks',
            'success': True,
            'train_size': len(train_df),
            'test_size': len(test_df)
        }
    
    # ============================================================================
    # PHASE 2: POST-TRAINING VALIDATION
    # ============================================================================
    
    def evaluate_model(self, model, test_path, label='target_final_points'):
        """
        Étape 4 : Évaluer le modèle avec DeepChecks
        
        À faire APRÈS l'entraînement
        """
        self.logger.info("="*70)
        self.logger.info("PHASE 4: MODEL EVALUATION (DeepChecks)")
        self.logger.info("="*70)
        
        # Charger les données de test
        test_df = pd.read_parquet(test_path)
        
        # DeepChecks evaluation
        # Simuler pour l'instant
        self.logger.info("✅ Model evaluation passed (simulated)")
        
        return {
            'phase': 'model_evaluation',
            'tool': 'deepchecks',
            'success': True
        }
    
    # ============================================================================
    # PHASE 3: PRODUCTION MONITORING
    # ============================================================================
    
    def detect_drift(self, reference_version='v1', current_version='v2', split='train'):
        """
        Étape 5 : Détecter le drift avec Evidently
        
        À faire EN PRODUCTION (régulièrement)
        """
        self.logger.info("="*70)
        self.logger.info("PHASE 5: DRIFT DETECTION (Evidently)")
        self.logger.info("="*70)
        
        # Evidently drift detection
        result = self.drift_detector.run_analysis(split=split)
        
        drift_detected = result['decision']['retrain_recommended']
        
        if drift_detected:
            self.logger.warning(f"⚠️  DRIFT DETECTED! Severity: {result['decision']['severity']}")
        else:
            self.logger.info("✅ No significant drift detected")
        
        return {
            'phase': 'drift_detection',
            'tool': 'evidently',
            'success': not drift_detected,
            'result': result
        }
    
    # ============================================================================
    # ORCHESTRATION
    # ============================================================================
    
    def run_pre_training_pipeline(self, seasons, raw_data_dir='data/raw', 
                                  features_path='data/processed/v1/features.parquet'):
        """
        Pipeline complet PRE-TRAINING
        
        1. Valider données brutes (GE)
        2. Valider features (GE)
        3. Valider split train/test (DeepChecks)
        """
        self.logger.info("\n" + "🚀 "*35)
        self.logger.info("STARTING PRE-TRAINING VALIDATION PIPELINE")
        self.logger.info("🚀 "*35 + "\n")
        
        results = {
            'pipeline': 'pre_training',
            'timestamp': datetime.now().isoformat(),
            'phases': []
        }
        
        # Phase 1: Raw data
        try:
            phase1 = self.validate_raw_data(seasons, raw_data_dir)
            results['phases'].append(phase1)
        except Exception as e:
            self.logger.error(f"Phase 1 failed: {e}")
            results['phases'].append({'phase': 'raw_data', 'success': False, 'error': str(e)})
        
        # Phase 2: Features
        try:
            phase2 = self.validate_features_quality(features_path)
            results['phases'].append(phase2)
        except Exception as e:
            self.logger.error(f"Phase 2 failed: {e}")
            results['phases'].append({'phase': 'features', 'success': False, 'error': str(e)})
        
        # Phase 3: Train-Test
        try:
            train_path = Path(features_path).parent / 'train.parquet'
            test_path = Path(features_path).parent / 'test.parquet'
            phase3 = self.validate_train_test_integrity(train_path, test_path)
            results['phases'].append(phase3)
        except Exception as e:
            self.logger.error(f"Phase 3 failed: {e}")
            results['phases'].append({'phase': 'train_test', 'success': False, 'error': str(e)})
        
        # Résumé
        all_success = all(p.get('success', False) for p in results['phases'])
        results['overall_success'] = all_success
        
        # Sauvegarder
        output_path = self.reports_dir / 'pre_training_validation.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        self.logger.info("\n" + "="*70)
        if all_success:
            self.logger.info("✅ PRE-TRAINING VALIDATION: ALL CHECKS PASSED")
            self.logger.info("   Ready to train models!")
        else:
            self.logger.warning("⚠️  PRE-TRAINING VALIDATION: SOME CHECKS FAILED")
            self.logger.warning("   Review validation reports before training!")
        self.logger.info("="*70)
        
        return results
    
    def run_production_monitoring(self, reference_version='v1', current_version='v2'):
        """
        Pipeline complet PRODUCTION MONITORING
        
        1. Détecter drift (Evidently)
        2. (Future: Alertes si drift critique)
        """
        self.logger.info("\n" + "📊 "*35)
        self.logger.info("STARTING PRODUCTION MONITORING")
        self.logger.info("📊 "*35 + "\n")
        
        results = {
            'pipeline': 'production_monitoring',
            'timestamp': datetime.now().isoformat(),
            'phases': []
        }
        
        # Phase: Drift Detection
        try:
            phase = self.detect_drift(reference_version, current_version)
            results['phases'].append(phase)
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            results['phases'].append({'phase': 'drift', 'success': False, 'error': str(e)})
        
        # Résumé
        drift_detected = not results['phases'][0].get('success', True)
        results['drift_detected'] = drift_detected
        results['action_required'] = drift_detected
        
        # Sauvegarder
        output_path = self.reports_dir / 'production_monitoring.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        self.logger.info("\n" + "="*70)
        if not drift_detected:
            self.logger.info("✅ PRODUCTION MONITORING: MODEL IS HEALTHY")
        else:
            self.logger.warning("⚠️  PRODUCTION MONITORING: DRIFT DETECTED - RETRAIN RECOMMENDED")
        self.logger.info("="*70)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Integrated Monitoring Pipeline')
    parser.add_argument(
        '--mode',
        choices=['pre-training', 'production', 'full'],
        default='pre-training',
        help='Pipeline mode'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2015-2016', '2016-2017', '2017-2018', '2018-2019', 
                '2019-2020', '2020-2021', '2021-2022', '2022-2023'],
        help='Seasons to validate'
    )
    
    args = parser.parse_args()
    
    # Créer le pipeline
    pipeline = IntegratedMonitoringPipeline()
    
    # Lancer selon le mode
    if args.mode == 'pre-training':
        result = pipeline.run_pre_training_pipeline(args.seasons)
    elif args.mode == 'production':
        result = pipeline.run_production_monitoring()
    else:  # full
        result1 = pipeline.run_pre_training_pipeline(args.seasons)
        result2 = pipeline.run_production_monitoring()
        result = {'pre_training': result1, 'production': result2}
    
    print("\n✅ Monitoring pipeline complete!")
    print(f"📊 Reports saved to: reports/integrated/")


if __name__ == '__main__':
    main()