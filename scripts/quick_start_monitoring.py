#!/usr/bin/env python3
"""
Quick Start Script - Monitoring System Test
Lance tous les checks de monitoring en une seule commande
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Exécute une commande et affiche le résultat"""
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ FAILED")
        if result.stderr:
            print(result.stderr)
        return False
    
    return True


def main():
    print("\n" + "🎯 "*35)
    print("PREMIER LEAGUE MLOPS - MONITORING QUICK START")
    print("🎯 "*35 + "\n")
    
    start_time = datetime.now()
    
    # Check des fichiers requis
    print("\n📋 Checking required files...")
    required_files = [
        'src/data/validation.py',
        'src/monitoring/drift_detection.py',
        'configs/drift_config.yaml',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"   ❌ Missing: {file_path}")
        else:
            print(f"   ✅ Found: {file_path}")
    
    if missing_files:
        print("\n⚠️  Some files are missing. Please set up the monitoring system first.")
        print("   See MONITORING_GUIDE.md for setup instructions.")
        return
    
    # Créer les dossiers nécessaires
    print("\n📁 Creating directories...")
    dirs = ['reports/deepchecks', 'reports/drift', 'reports/integrated', 'logs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {d}")
    
    results = []
    
    # 1. Great Expectations - Raw Data
    success = run_command(
        'python src/data/validation.py --mode raw',
        "Step 1: Great Expectations - Raw Data Validation"
    )
    results.append(('Great Expectations (Raw)', success))
    
    # 2. Great Expectations - Features
    success = run_command(
        'python src/data/validation.py --mode features --features-path data/processed/v1/features.parquet',
        "Step 2: Great Expectations - Features Validation"
    )
    results.append(('Great Expectations (Features)', success))
    
    # 3. Evidently - Drift Detection
    success = run_command(
        'python src/monitoring/drift_detection.py --config configs/drift_config.yaml --split train',
        "Step 3: Evidently - Drift Detection"
    )
    results.append(('Evidently Drift Detection', success))
    
    # 4. DeepChecks (si disponible)
    if Path('src/monitoring/deepchecks_validator.py').exists():
        success = run_command(
            'python src/monitoring/deepchecks_validator.py --mode full --data-dir data/processed/v1',
            "Step 4: DeepChecks - Full Validation"
        )
        results.append(('DeepChecks Validation', success))
    else:
        print("\n⚠️  DeepChecks validator not found. Skipping...")
        print("   To add: cp deepchecks_validator.py src/monitoring/")
    
    # Résumé final
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("📊 MONITORING QUICK START SUMMARY")
    print("="*70)
    
    for step, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {step}")
    
    total_success = sum(1 for _, s in results if s)
    total_steps = len(results)
    
    print(f"\n📈 Success Rate: {total_success}/{total_steps} ({total_success/total_steps*100:.0f}%)")
    print(f"⏱️  Duration: {duration:.1f}s")
    
    print("\n📁 Check reports in:")
    print("   - data/validation_reports/ (Great Expectations)")
    print("   - reports/drift/ (Evidently)")
    print("   - reports/deepchecks/ (DeepChecks)")
    
    if total_success == total_steps:
        print("\n✅ ALL MONITORING CHECKS PASSED!")
        print("   Your data and model pipeline is healthy.")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("   Review the reports above to identify issues.")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)