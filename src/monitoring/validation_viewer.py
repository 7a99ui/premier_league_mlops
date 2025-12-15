"""
DeepChecks Results Viewer
Affiche les résultats de validation de manière lisible
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import argparse


class ValidationResultsViewer:
    """Visualise les résultats de validation DeepChecks"""
    
    def __init__(self):
        self.console = Console()
    
    def load_results(self, json_path: str = "reports/deepchecks/validation_summary.json") -> dict:
        """Charge les résultats depuis le JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def display_summary(self, results: dict):
        """Affiche le résumé général"""
        self.console.print("\n")
        self.console.print("="*80, style="bold blue")
        self.console.print("  DEEPCHECKS VALIDATION RESULTS SUMMARY", style="bold blue")
        self.console.print("="*80, style="bold blue")
        
        # Info générale
        timestamp = datetime.fromisoformat(results['timestamp'])
        self.console.print(f"\n⏰ Validation Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        self.console.print(f"📊 Train Size: {results['train_size']:,} rows")
        self.console.print(f"📊 Test Size: {results['test_size']:,} rows")
        self.console.print(f"🔍 Total Checks: {len(results['checks'])}")
    
    def display_check_details(self, results: dict):
        """Affiche les détails de chaque check"""
        
        for i, check in enumerate(results['checks'], 1):
            self.console.print(f"\n{'='*80}")
            
            check_type = check['check_type'].upper().replace('_', ' ')
            
            # Couleur selon le succès
            if check.get('success', False):
                status = "✅ SUCCESS"
                style = "green"
            else:
                status = "❌ FAILED"
                style = "red"
            
            self.console.print(f"\n[{style}]Check {i}: {check_type} - {status}[/{style}]", style="bold")
            
            # Détails selon le type
            if 'summary' in check:
                self._display_suite_summary(check['summary'])
            elif check['check_type'] == 'feature_drift':
                self._display_feature_drift(check)
            elif check['check_type'] == 'label_drift':
                self._display_label_drift(check)
    
    def _display_suite_summary(self, summary: dict):
        """Affiche le résumé d'une suite de checks"""
        n_checks = summary['n_checks']
        n_passed = summary['n_passed']
        n_failed = summary['n_failed']
        success_rate = summary['success_rate'] * 100
        
        # Table de résumé
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Checks", str(n_checks))
        table.add_row("✅ Passed", f"[green]{n_passed}[/green]")
        table.add_row("❌ Failed", f"[red]{n_failed}[/red]")
        table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        self.console.print(table)
        
        # Failed checks
        if n_failed > 0:
            self.console.print("\n🔴 Failed Checks:", style="bold red")
            for failed_check in summary['failed_checks']:
                self.console.print(f"  • {failed_check}", style="red")
    
    def _display_feature_drift(self, check: dict):
        """Affiche les résultats du feature drift"""
        n_drifted = check['n_drifted']
        threshold = check['drift_threshold']
        
        if n_drifted == 0:
            self.console.print(f"\n✅ No features drifted above threshold {threshold}", style="green")
        else:
            self.console.print(f"\n⚠️  {n_drifted} feature(s) drifted above threshold {threshold}", style="yellow")
            for feature in check['drifted_features']:
                self.console.print(f"  • {feature}", style="yellow")
    
    def _display_label_drift(self, check: dict):
        """Affiche les résultats du label drift"""
        drift_score = check['drift_score']
        drift_detected = check['drift_detected']
        
        if drift_detected:
            self.console.print(f"\n⚠️  Label drift detected!", style="yellow")
            self.console.print(f"   Drift Score: {drift_score:.4f}", style="yellow")
        else:
            self.console.print(f"\n✅ No significant label drift", style="green")
            self.console.print(f"   Drift Score: {drift_score:.4f}", style="green")
    
    def create_issues_report(self, results: dict) -> pd.DataFrame:
        """Crée un DataFrame avec tous les problèmes détectés"""
        issues = []
        
        for check in results['checks']:
            if 'summary' in check and check['summary']['n_failed'] > 0:
                for failed_check in check['summary']['failed_checks']:
                    issues.append({
                        'check_type': check['check_type'],
                        'issue': failed_check,
                        'severity': 'HIGH' if check['summary']['success_rate'] < 0.5 else 'MEDIUM'
                    })
            
            if check['check_type'] == 'feature_drift' and check['n_drifted'] > 0:
                for feature in check['drifted_features']:
                    issues.append({
                        'check_type': 'feature_drift',
                        'issue': f"Feature '{feature}' has drifted",
                        'severity': 'MEDIUM'
                    })
            
            if check['check_type'] == 'label_drift' and check['drift_detected']:
                issues.append({
                    'check_type': 'label_drift',
                    'issue': f"Label drift detected (score: {check['drift_score']:.4f})",
                    'severity': 'HIGH'
                })
        
        return pd.DataFrame(issues)
    
    def display_recommendations(self, results: dict):
        """Affiche les recommandations basées sur les résultats"""
        self.console.print(f"\n{'='*80}")
        self.console.print("\n💡 RECOMMENDATIONS", style="bold cyan")
        self.console.print("="*80)
        
        issues_df = self.create_issues_report(results)
        
        if len(issues_df) == 0:
            self.console.print("\n✅ All checks passed! Your data looks good.", style="green bold")
            return
        
        # Recommandations par type d'issue
        recommendations = {
            'data_integrity': [
                "📌 Review feature engineering to reduce multicollinearity",
                "📌 Investigate conflicting labels - may indicate data quality issues",
                "📌 Remove or encode identifier columns properly"
            ],
            'train_test_validation': [
                "📌 Check temporal data leakage - ensure proper time-based split",
                "📌 Remove duplicate records between train and test",
                "📌 Review feature distributions for significant drift",
                "📌 Consider stratified sampling to maintain label distribution"
            ],
            'feature_drift': [
                "📌 Investigate drifted features for data quality issues",
                "📌 Consider feature normalization or transformation",
                "📌 May need to retrain model with recent data"
            ],
            'label_drift': [
                "📌 Target distribution has changed - model may underperform",
                "📌 Consider retraining with more recent data",
                "📌 Implement monitoring for production predictions"
            ]
        }
        
        # Afficher les recommandations pertinentes
        check_types_with_issues = issues_df['check_type'].unique()
        
        for check_type in check_types_with_issues:
            if check_type in recommendations:
                self.console.print(f"\n🔧 {check_type.upper().replace('_', ' ')}:", style="bold yellow")
                for rec in recommendations[check_type]:
                    self.console.print(f"   {rec}")
    
    def run(self, json_path: str = "reports/deepchecks/validation_summary.json"):
        """Lance l'affichage complet"""
        try:
            results = self.load_results(json_path)
            
            self.display_summary(results)
            self.display_check_details(results)
            self.display_recommendations(results)
            
            # Export issues
            issues_df = self.create_issues_report(results)
            if len(issues_df) > 0:
                issues_path = Path(json_path).parent / "validation_issues.csv"
                issues_df.to_csv(issues_path, index=False)
                self.console.print(f"\n💾 Issues exported to: {issues_path}", style="cyan")
            
            self.console.print(f"\n{'='*80}\n")
            
        except FileNotFoundError:
            self.console.print(f"\n❌ Error: File not found: {json_path}", style="red bold")
        except Exception as e:
            self.console.print(f"\n❌ Error: {e}", style="red bold")


def main():
    parser = argparse.ArgumentParser(description='View DeepChecks Validation Results')
    parser.add_argument('--json', default='reports/deepchecks/validation_summary.json',
                       help='Path to validation_summary.json')
    
    args = parser.parse_args()
    
    viewer = ValidationResultsViewer()
    viewer.run(args.json)


if __name__ == '__main__':
    main()