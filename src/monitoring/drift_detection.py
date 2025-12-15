"""
Data Drift Detection avec Configuration Centralisée
Version améliorée avec YAML config et support UTF-8
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import argparse
import logging

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric


class DriftDetectorV2:
    """Version améliorée avec configuration YAML"""
    
    def __init__(self, config_path: str = "configs/drift_config.yaml"):
        # Charger la configuration
        self.config = self._load_config(config_path)
        
        # Paths
        self.base_path = Path(self.config['paths']['data_base'])
        self.reports_path = Path(self.config['paths']['reports'])
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        # Versions
        self.reference_version = self.config['versions']['reference']
        self.current_version = self.config['versions']['current']
        
        # Paramètres
        self.drift_threshold = self.config['detection']['drift_threshold']
        self.target_col = self.config['detection']['target_column']
        
        # Setup logging
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Charge la configuration depuis YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Config non trouvée: {config_path}")
            print("📝 Utilisation des valeurs par défaut")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Configuration par défaut si YAML manquant"""
        return {
            'paths': {'data_base': 'data/processed', 'reports': 'reports/drift'},
            'versions': {'reference': 'v1', 'current': 'v2'},
            'detection': {
                'drift_threshold': 0.3,
                'target_column': 'target_final_points',
                'splits': ['train', 'val', 'test']
            },
            'decision': {
                'severity_thresholds': {'high': 0.5, 'medium': 0.3, 'low': 0.0},
                'force_retrain_on_target_drift': True
            },
            'reports': {
                'generate_html': True,
                'generate_json': True,
                'keep_history': True
            }
        }
    
    def _setup_logging(self):
        """Configure le logging avec support UTF-8"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_file = self.config.get('logging', {}).get('file', 'logs/drift_detection.log')
        
        # Créer le dossier logs
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Créer les handlers avec encodage UTF-8
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # StreamHandler avec gestion d'erreurs pour les émojis
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(getattr(logging, log_level))
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Configurer le logger root
        logging.basicConfig(
            level=getattr(logging, log_level),
            handlers=[file_handler, stream_handler]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, version: str, split: str = "train") -> pd.DataFrame:
        """Charge les données"""
        data_path = self.base_path / version / f"{split}.parquet"
        
        if not data_path.exists():
            self.logger.error(f"Données introuvables : {data_path}")
            raise FileNotFoundError(f"Données introuvables : {data_path}")
        
        self.logger.info(f"Chargement: {data_path}")
        return pd.read_parquet(data_path)
    
    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict:
        """Détecte le drift avec Evidently"""
        # Colonnes communes
        common_cols = list(set(reference_df.columns) & set(current_df.columns))
        ref_data = reference_df[common_cols].copy()
        curr_data = current_df[common_cols].copy()
        
        self.logger.info(f"Analyse de drift sur {len(common_cols)} colonnes")
        
        # Configuration Evidently
        column_mapping = ColumnMapping()
        column_mapping.target = self.target_col
        
        # Créer le rapport
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ])
        
        # Exécuter
        report.run(reference_data=ref_data, current_data=curr_data, column_mapping=column_mapping)
        
        # Sauvegarder si configuré
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config['reports']['generate_html']:
            html_path = self.reports_path / f"drift_report_{self.reference_version}_vs_{self.current_version}_{timestamp}.html"
            report.save_html(str(html_path))
            self.logger.info(f"Rapport HTML: {html_path}")
        
        # Extraire métriques
        report_dict = report.as_dict()
        metrics = self._extract_metrics(report_dict)
        
        if self.config['reports']['generate_json']:
            json_path = self.reports_path / f"drift_metrics_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            self.logger.info(f"Métriques JSON: {json_path}")
        
        return metrics
    
    def _extract_metrics(self, report_dict: Dict) -> Dict:
        """Extrait les métriques clés"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "reference_version": self.reference_version,
            "current_version": self.current_version,
            "drift_detected": False,
            "dataset_drift": {},
            "drifted_features": [],
            "target_drift": {},
        }
        
        for metric in report_dict.get('metrics', []):
            metric_type = metric.get('metric')
            result = metric.get('result', {})
            
            if metric_type == "DatasetDriftMetric":
                metrics["drift_detected"] = result.get('dataset_drift', False)
                metrics["dataset_drift"] = {
                    "drift_share": result.get('drift_share', 0),
                    "number_of_drifted_columns": result.get('number_of_drifted_columns', 0),
                    "number_of_columns": result.get('number_of_columns', 0),
                }
                drift_by_columns = result.get('drift_by_columns', {})
                metrics["drifted_features"] = [
                    col for col, info in drift_by_columns.items() 
                    if info.get('drift_detected', False)
                ]
            
            elif metric_type == "ColumnDriftMetric":
                column_name = result.get('column_name')
                if column_name == self.target_col:
                    metrics["target_drift"] = {
                        "drift_detected": result.get('drift_detected', False),
                        "drift_score": result.get('drift_score'),
                        "stattest_name": result.get('stattest_name'),
                    }
        
        return metrics
    
    def should_retrain(self, metrics: Dict) -> Dict:
        """Décide du réentraînement"""
        decision = {
            "retrain_recommended": False,
            "reasons": [],
            "severity": "low",
        }
        
        # Calculer le drift réel
        drifted_features = metrics.get("drifted_features", [])
        num_total = metrics.get("dataset_drift", {}).get("number_of_columns", 1)
        actual_drift_rate = len(drifted_features) / num_total if num_total > 0 else 0
        
        # Vérifier target drift
        target_drift = metrics.get("target_drift", {}).get("drift_detected", False)
        force_retrain = self.config['decision']['force_retrain_on_target_drift']
        
        if target_drift and force_retrain:
            decision["retrain_recommended"] = True
            decision["reasons"].append(f"Target drift détecté sur '{self.target_col}'")
            decision["severity"] = "high"
        
        # Vérifier dataset drift
        thresholds = self.config['decision']['severity_thresholds']
        
        if actual_drift_rate > self.drift_threshold:
            decision["retrain_recommended"] = True
            decision["reasons"].append(
                f"Dataset drift: {actual_drift_rate:.1%} des features (seuil: {self.drift_threshold:.1%})"
            )
        
        # Déterminer sévérité
        if actual_drift_rate > thresholds['high']:
            decision["severity"] = "high"
        elif actual_drift_rate > thresholds['medium']:
            decision["severity"] = "medium"
        
        decision["actual_drift_rate"] = actual_drift_rate
        decision["num_drifted_features"] = len(drifted_features)
        decision["num_total_features"] = num_total
        
        return decision
    
    def run_analysis(self, split: str = "train") -> Dict:
        """Analyse complète"""
        self.logger.info(f"Début analyse: {self.reference_version} vs {self.current_version} ({split})")
        
        # Charger données
        reference_df = self.load_data(self.reference_version, split)
        current_df = self.load_data(self.current_version, split)
        
        # Détecter drift
        metrics = self.detect_drift(reference_df, current_df)
        
        # Décision
        decision = self.should_retrain(metrics)
        
        # Log résultat (sans émojis pour éviter problèmes Windows)
        if decision['retrain_recommended']:
            self.logger.warning(f"[ALERT] RÉENTRAÎNEMENT RECOMMANDÉ ({decision['severity']})")
        else:
            self.logger.info(f"[OK] MODÈLE OK ({decision['severity']})")
        
        result = {"metrics": metrics, "decision": decision}
        
        # Sauvegarder décision
        output_path = self.reports_path / "latest_decision.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Drift Detection avec Config")
    parser.add_argument("--config", type=str, default="configs/drift_config.yaml", help="Config file")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    
    args = parser.parse_args()
    
    # Créer détecteur
    detector = DriftDetectorV2(config_path=args.config)
    
    # Analyser
    result = detector.run_analysis(split=args.split)
    
    # Affichage avec gestion des émojis Windows
    try:
        print("\n✅ Analyse terminée !")
        print(f"📊 Drift rate: {result['decision']['actual_drift_rate']:.1%}")
        print(f"🤖 Recommandation: {'🔴 RÉENTRAÎNER' if result['decision']['retrain_recommended'] else '🟢 OK'}")
    except UnicodeEncodeError:
        # Fallback sans émojis pour Windows CMD
        print("\n[OK] Analyse terminée !")
        print(f"Drift rate: {result['decision']['actual_drift_rate']:.1%}")
        print(f"Recommandation: {'[!] RÉENTRAÎNER' if result['decision']['retrain_recommended'] else '[OK] MODÈLE OK'}")


if __name__ == "__main__":
    main()