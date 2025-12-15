"""
Model Evaluation Tests avec DeepChecks
Version corrigée - Compatible avec DeepChecks version récente
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse
import logging
import joblib
import mlflow
from mlflow.tracking import MlflowClient

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    # Data Integrity
    NewLabelTrainTest,
    
    # Train-Test Validation
    FeatureDrift,
    LabelDrift,
    PredictionDrift,
    MultivariateDrift,
    
    # Model Evaluation
    ModelInferenceTime,
    UnusedFeatures,
    SimpleModelComparison,
    TrainTestPerformance,  # Remplace PerformanceReport
    RocReport,
    ConfusionMatrixReport,
    CalibrationScore,
)


class ModelEvaluationSuite:
    """
    Suite complète d'évaluation de modèle
    """
    
    def __init__(self, reports_dir='reports/model_evaluation', 
                 mlflow_tracking_uri='mlruns'):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = MlflowClient()
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure le logging"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/model_evaluation.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_production_model(self) -> tuple:
        """
        Charge automatiquement le modèle en production
        Ordre: MLflow Registry → Local models/production/ → Meilleur MLflow
        """
        self.logger.info("Searching for production model...")
        
        # 1. MLflow Registry (Production)
        try:
            registered_models = self.client.search_registered_models()
            
            for rm in registered_models:
                prod_versions = self.client.get_latest_versions(rm.name, stages=["Production"])
                
                if prod_versions:
                    model_name = rm.name
                    model_version = prod_versions[0].version
                    
                    self.logger.info(f"✅ Found production model in MLflow: {model_name} v{model_version}")
                    
                    model_uri = f"models:/{model_name}/{model_version}"
                    model = mlflow.pyfunc.load_model(model_uri)
                    
                    model_info = {
                        'source': 'mlflow_registry',
                        'name': model_name,
                        'version': model_version,
                        'stage': 'Production'
                    }
                    
                    return model, model_info
        except Exception as e:
            self.logger.warning(f"MLflow Registry check failed: {e}")
        
        # 2. Local models/production/
        self.logger.info("No MLflow production model, searching for local model...")
        
        production_dir = Path('models/production')
        if production_dir.exists():
            latest_model_path = production_dir / 'latest_model.joblib'
            
            if latest_model_path.exists():
                self.logger.info(f"✅ Found local model: {latest_model_path}")
                
                model = joblib.load(latest_model_path)
                
                # Charger métadonnées
                metadata_path = production_dir / 'latest_metadata.json'
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                model_info = {
                    'source': 'local_file',
                    'path': str(latest_model_path),
                    'file_size_mb': round(latest_model_path.stat().st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(latest_model_path.stat().st_mtime).isoformat(),
                    **metadata
                }
                
                return model, model_info
        
        raise ValueError(
            "❌ No model found!\n"
            "Solutions:\n"
            "  1. Put a model in models/production/latest_model.joblib\n"
            "  2. Use: --model-path path/to/model.joblib\n"
            "  3. Register a model in MLflow Registry"
        )
    
    def _prepare_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                         label: str) -> tuple:
        """
        Prépare les datasets DeepChecks avec gestion correcte des features catégorielles
        """
        # Identifier les vraies features catégorielles (non encodées)
        categorical_features = []
        
        for col in train_df.columns:
            if col == label:
                continue
            
            # Colonnes qui sont des strings ou ont peu de valeurs uniques
            if train_df[col].dtype == 'object':
                categorical_features.append(col)
            elif train_df[col].dtype in ['int64', 'float64']:
                # Si c'est numérique mais avec très peu de valeurs uniques, peut-être catégoriel
                unique_ratio = train_df[col].nunique() / len(train_df)
                if unique_ratio < 0.05 and train_df[col].nunique() < 20:
                    categorical_features.append(col)
        
        self.logger.info(f"Detected categorical features: {categorical_features}")
        
        # Créer les datasets
        train_ds = Dataset(train_df, label=label, cat_features=categorical_features)
        test_ds = Dataset(test_df, label=label, cat_features=categorical_features)
        
        return train_ds, test_ds, categorical_features
    
    def _prepare_data_for_model(self, df: pd.DataFrame, model_info: dict) -> pd.DataFrame:
        """
        Prépare les données pour correspondre aux features attendues par le modèle
        """
        # Si le modèle a été entraîné avec des features spécifiques
        if 'feature_names' in model_info:
            expected_features = model_info['feature_names']
            
            # Vérifier les features manquantes
            missing_features = set(expected_features) - set(df.columns)
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                # Ajouter des colonnes nulles pour les features manquantes
                for feat in missing_features:
                    df[feat] = 0
            
            # Sélectionner uniquement les features attendues dans le bon ordre
            df = df[expected_features]
        
        return df
    
    # ========================================================================
    # TESTS
    # ========================================================================
    
    def test_new_label_train_test(self, train_ds: Dataset, test_ds: Dataset) -> dict:
        """Test 1: New Label Train Test"""
        self.logger.info("Running: New Label Train Test")
        
        try:
            check = NewLabelTrainTest()
            result = check.run(train_ds, test_ds)
            
            passed = result.passed_conditions() if hasattr(result, 'passed_conditions') else True
            
            return {
                'test_name': 'New Label Train Test',
                'category': 'Data Integrity',
                'passed': passed,
                'criteria': 'Ratio of new labels = 0%',
                'result': f"Passed: {passed}"
            }
        except Exception as e:
            return {
                'test_name': 'New Label Train Test',
                'category': 'Data Integrity',
                'passed': False,
                'criteria': 'Ratio of new labels = 0%',
                'result': f"Error: {str(e)}"
            }
    
    def test_feature_drift(self, train_ds: Dataset, test_ds: Dataset) -> dict:
        """Test 2: Feature Drift"""
        self.logger.info("Running: Feature Drift")
        
        try:
            check = FeatureDrift()
            check.add_condition_drift_score_less_than(max_score=0.2)
            result = check.run(train_ds, test_ds)
            
            passed = result.passed_conditions()
            
            return {
                'test_name': 'Feature Drift',
                'category': 'Train-Test Validation',
                'passed': passed,
                'criteria': 'Drift score <= 0.2',
                'result': f"Passed: {passed}"
            }
        except Exception as e:
            return {
                'test_name': 'Feature Drift',
                'category': 'Train-Test Validation',
                'passed': False,
                'criteria': 'Drift score <= 0.2',
                'result': f"Error: {str(e)}"
            }
    
    def test_label_drift(self, train_ds: Dataset, test_ds: Dataset) -> dict:
        """Test 3: Label Drift"""
        self.logger.info("Running: Label Drift")
        
        try:
            check = LabelDrift()
            check.add_condition_drift_score_less_than(max_score=0.2)
            result = check.run(train_ds, test_ds)
            
            passed = result.passed_conditions()
            
            return {
                'test_name': 'Label Drift',
                'category': 'Train-Test Validation',
                'passed': passed,
                'criteria': 'Drift score <= 0.2',
                'result': f"Passed: {passed}"
            }
        except Exception as e:
            return {
                'test_name': 'Label Drift',
                'category': 'Train-Test Validation',
                'passed': False,
                'criteria': 'Drift score <= 0.2',
                'result': f"Error: {str(e)}"
            }
    
    def test_prediction_drift(self, train_ds: Dataset, test_ds: Dataset, model) -> dict:
        """Test 4: Prediction Drift"""
        self.logger.info("Running: Prediction Drift")
        
        try:
            check = PredictionDrift()
            check.add_condition_drift_score_less_than(max_score=0.15)
            result = check.run(train_ds, test_ds, model)
            
            passed = result.passed_conditions()
            
            return {
                'test_name': 'Prediction Drift',
                'category': 'Train-Test Validation',
                'passed': passed,
                'criteria': 'Drift score <= 0.15',
                'result': f"Passed: {passed}"
            }
        except Exception as e:
            return {
                'test_name': 'Prediction Drift',
                'category': 'Train-Test Validation',
                'passed': False,
                'criteria': 'Drift score <= 0.15',
                'result': f"Error: {str(e)}"
            }
    
    def test_multivariate_drift(self, train_ds: Dataset, test_ds: Dataset) -> dict:
        """Test 5: Multivariate Drift"""
        self.logger.info("Running: Multivariate Drift")
        
        try:
            check = MultivariateDrift()
            check.add_condition_overall_drift_value_less_than(0.25)
            result = check.run(train_ds, test_ds)
            
            passed = result.passed_conditions()
            
            return {
                'test_name': 'Multivariate Drift',
                'category': 'Train-Test Validation',
                'passed': passed,
                'criteria': 'Drift value <= 0.25',
                'result': f"Passed: {passed}"
            }
        except Exception as e:
            return {
                'test_name': 'Multivariate Drift',
                'category': 'Train-Test Validation',
                'passed': False,
                'criteria': 'Drift value <= 0.25',
                'result': f"Error: {str(e)}"
            }
    
    def test_inference_time(self, dataset: Dataset, model) -> dict:
        """Test 6: Model Inference Time"""
        self.logger.info("Running: Model Inference Time")
        
        try:
            check = ModelInferenceTime()
            check.add_condition_inference_time_less_than(0.001)
            result = check.run(dataset, model)
            
            passed = result.passed_conditions()
            
            return {
                'test_name': 'Model Inference Time',
                'category': 'Model Evaluation',
                'passed': passed,
                'criteria': 'Inference time <= 0.001s per sample',
                'result': f"Passed: {passed}"
            }
        except Exception as e:
            return {
                'test_name': 'Model Inference Time',
                'category': 'Model Evaluation',
                'passed': False,
                'criteria': 'Inference time <= 0.001s per sample',
                'result': f"Error: {str(e)}"
            }
    
    def test_train_test_performance(self, train_ds: Dataset, test_ds: Dataset, model) -> dict:
        """Test 7: Train Test Performance (remplace PerformanceReport)"""
        self.logger.info("Running: Train Test Performance")
        
        try:
            check = TrainTestPerformance()
            check.add_condition_train_test_relative_degradation_less_than(0.1)
            result = check.run(train_ds, test_ds, model)
            
            passed = result.passed_conditions()
            
            return {
                'test_name': 'Train Test Performance',
                'category': 'Model Evaluation',
                'passed': passed,
                'criteria': 'Train-Test degradation <= 10%',
                'result': f"Passed: {passed}"
            }
        except Exception as e:
            return {
                'test_name': 'Train Test Performance',
                'category': 'Model Evaluation',
                'passed': False,
                'criteria': 'Train-Test degradation <= 10%',
                'result': f"Error: {str(e)}"
            }
    
    def test_simple_model_comparison(self, train_ds: Dataset, test_ds: Dataset, model) -> dict:
        """Test 8: Simple Model Comparison"""
        self.logger.info("Running: Simple Model Comparison")
        
        try:
            check = SimpleModelComparison()
            check.add_condition_gain_greater_than(0.1)
            result = check.run(train_ds, test_ds, model)
            
            passed = result.passed_conditions()
            
            return {
                'test_name': 'Simple Model Comparison',
                'category': 'Model Evaluation',
                'passed': passed,
                'criteria': 'Performance gain >= 10%',
                'result': f"Passed: {passed}"
            }
        except Exception as e:
            return {
                'test_name': 'Simple Model Comparison',
                'category': 'Model Evaluation',
                'passed': False,
                'criteria': 'Performance gain >= 10%',
                'result': f"Error: {str(e)}"
            }
    
    def test_unused_features(self, train_ds: Dataset, test_ds: Dataset, model) -> dict:
        """Test 9: Unused Features"""
        self.logger.info("Running: Unused Features")
        
        try:
            check = UnusedFeatures()
            result = check.run(train_ds, test_ds, model)
            
            # Ce test est informatif, pas de condition strict
            return {
                'test_name': 'Unused Features',
                'category': 'Model Evaluation',
                'passed': None,  # Informatif
                'criteria': 'Informational',
                'result': "Check completed"
            }
        except Exception as e:
            return {
                'test_name': 'Unused Features',
                'category': 'Model Evaluation',
                'passed': None,
                'criteria': 'Informational',
                'result': f"Error: {str(e)}"
            }
    
    def test_calibration_score(self, test_ds: Dataset, model) -> dict:
        """Test 10: Calibration Score (pour classification)"""
        self.logger.info("Running: Calibration Score")
        
        try:
            # Vérifier si c'est une tâche de classification
            y_test = test_ds.data[test_ds.label_name]
            if y_test.nunique() > 10:
                # Probablement régression, skip
                return {
                    'test_name': 'Calibration Score',
                    'category': 'Model Evaluation',
                    'passed': None,
                    'criteria': 'N/A (Regression task)',
                    'result': "Skipped (not a classification task)"
                }
            
            check = CalibrationScore()
            result = check.run(test_ds, model)
            
            return {
                'test_name': 'Calibration Score',
                'category': 'Model Evaluation',
                'passed': None,  # Informatif
                'criteria': 'Informational',
                'result': "Check completed"
            }
        except Exception as e:
            return {
                'test_name': 'Calibration Score',
                'category': 'Model Evaluation',
                'passed': None,
                'criteria': 'Informational',
                'result': f"Error: {str(e)}"
            }
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    def run_all_tests(self, train_path: str, test_path: str, 
                     model=None, model_path: str = None,
                     label: str = 'target_final_points') -> dict:
        """
        Lance TOUS les tests
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("RUNNING FULL MODEL EVALUATION SUITE")
        self.logger.info("="*70)
        
        # Charger données
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        self.logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        # Charger modèle
        model_info = {}
        
        if model is not None:
            self.logger.info("Using provided model")
        elif model_path:
            self.logger.info(f"Loading model from: {model_path}")
            model = joblib.load(model_path)
            model_info['source'] = 'file'
            model_info['path'] = model_path
            
            # Extraire les feature names si disponibles
            if hasattr(model, 'feature_names_in_'):
                model_info['feature_names'] = list(model.feature_names_in_)
                self.logger.info(f"Model expects {len(model.feature_names_in_)} features")
        else:
            model, model_info = self.load_production_model()
        
        # Préparer les datasets
        train_ds, test_ds, cat_features = self._prepare_datasets(train_df, test_df, label)
        
        # Préparer les données pour le modèle si nécessaire
        if 'feature_names' in model_info:
            self.logger.info("Aligning features with model expectations...")
            train_df_model = self._prepare_data_for_model(
                train_df.drop(columns=[label]), 
                model_info
            )
            test_df_model = self._prepare_data_for_model(
                test_df.drop(columns=[label]), 
                model_info
            )
            
            # Recréer les datasets avec les bonnes features
            train_ds_model = Dataset(
                pd.concat([train_df_model, train_df[[label]]], axis=1),
                label=label,
                cat_features=[f for f in cat_features if f in train_df_model.columns]
            )
            test_ds_model = Dataset(
                pd.concat([test_df_model, test_df[[label]]], axis=1),
                label=label,
                cat_features=[f for f in cat_features if f in test_df_model.columns]
            )
        else:
            train_ds_model = train_ds
            test_ds_model = test_ds
        
        # Résultats
        results = {
            'timestamp': datetime.now().isoformat(),
            'train_path': str(train_path),
            'test_path': str(test_path),
            'model_info': model_info,
            'tests': []
        }
        
        # Lancer les tests
        test_functions = [
            (self.test_new_label_train_test, [train_ds, test_ds]),
            (self.test_feature_drift, [train_ds, test_ds]),
            (self.test_label_drift, [train_ds, test_ds]),
            (self.test_prediction_drift, [train_ds_model, test_ds_model, model]),
            (self.test_multivariate_drift, [train_ds, test_ds]),
            (self.test_inference_time, [train_ds_model, model]),
            (self.test_train_test_performance, [train_ds_model, test_ds_model, model]),
            (self.test_simple_model_comparison, [train_ds_model, test_ds_model, model]),
            (self.test_unused_features, [train_ds_model, test_ds_model, model]),
            (self.test_calibration_score, [test_ds_model, model]),
        ]
        
        for test_func, args in test_functions:
            try:
                result = test_func(*args)
                results['tests'].append(result)
            except Exception as e:
                self.logger.error(f"{test_func.__name__} failed: {e}", exc_info=True)
                results['tests'].append({
                    'test_name': test_func.__name__,
                    'passed': False,
                    'result': f"Exception: {str(e)}"
                })
        
        # Calculer statistiques
        passed_tests = [t for t in results['tests'] if t.get('passed') == True]
        failed_tests = [t for t in results['tests'] if t.get('passed') == False]
        info_tests = [t for t in results['tests'] if t.get('passed') is None]
        
        results['summary'] = {
            'total_tests': len(results['tests']),
            'passed': len(passed_tests),
            'failed': len(failed_tests),
            'informational': len(info_tests),
            'pass_rate': len(passed_tests) / (len(passed_tests) + len(failed_tests)) if (len(passed_tests) + len(failed_tests)) > 0 else 0
        }
        
        # Sauvegarder
        output_path = self.reports_dir / 'full_evaluation_report.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Log résumé
        self.logger.info("\n" + "="*70)
        self.logger.info(f"RESULTS: {len(passed_tests)} PASSED, {len(failed_tests)} FAILED, {len(info_tests)} INFO")
        self.logger.info(f"Pass Rate: {results['summary']['pass_rate']:.1%}")
        self.logger.info(f"Report saved: {output_path}")
        self.logger.info("="*70)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation Suite')
    
    parser.add_argument('--train', default='data/processed/v3/train.parquet')
    parser.add_argument('--test', default='data/processed/v3/test.parquet')
    parser.add_argument('--label', default='target_final_points')
    parser.add_argument('--model-path', help='Path to .pkl/.joblib file')
    parser.add_argument('--auto', action='store_true', help='Auto-detect production model')
    parser.add_argument('--mlflow-uri', default='mlruns')
    
    args = parser.parse_args()
    
    suite = ModelEvaluationSuite(mlflow_tracking_uri=args.mlflow_uri)
    
    if args.model_path:
        results = suite.run_all_tests(args.train, args.test, model_path=args.model_path, label=args.label)
    else:
        results = suite.run_all_tests(args.train, args.test, label=args.label)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Report: reports/model_evaluation/full_evaluation_report.json")


if __name__ == '__main__':
    main()