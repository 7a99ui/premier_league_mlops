# scripts/clean_predictions.py (version améliorée)
import os
import glob
import shutil
from datetime import datetime, timedelta
import argparse

def clean_predictions(keep_latest_only=True, days_to_keep=3, archive_old=False):
    """
    Nettoyer le dossier predictions
    
    Args:
        keep_latest_only: Garder seulement les fichiers 'latest'
        days_to_keep: Nombre de jours à garder pour les fichiers d'évolution
        archive_old: Archiver les anciens fichiers au lieu de les supprimer
    """
    predictions_dir = "predictions/"
    
    if not os.path.exists(predictions_dir):
        print(f"Le dossier {predictions_dir} n'existe pas.")
        return
    
    # 1. Nettoyer les CSV (garder seulement latest)
    if keep_latest_only:
        csv_files = glob.glob(f"{predictions_dir}/*.csv")
        for csv_file in csv_files:
            if "latest" not in os.path.basename(csv_file):
                if archive_old:
                    archive_file(csv_file)
                else:
                    os.remove(csv_file)
                    print(f"Supprimé CSV: {csv_file}")
    
    # 2. Nettoyer les JSON d'évolution
    evolution_dir = os.path.join(predictions_dir, "evolution")
    if os.path.exists(evolution_dir):
        now = datetime.now()
        json_files = glob.glob(f"{evolution_dir}/*.json")
        
        # Garder les 3 plus récents
        json_files.sort(key=os.path.getmtime, reverse=True)
        files_to_keep = json_files[:3]
        
        for json_file in json_files:
            if json_file not in files_to_keep:
                if archive_old:
                    archive_file(json_file)
                else:
                    os.remove(json_file)
                    print(f"Supprimé JSON: {json_file}")
    
    # 3. Nettoyer analysis/
    analysis_dir = os.path.join(predictions_dir, "analysis")
    if os.path.exists(analysis_dir):
        analysis_files = glob.glob(f"{analysis_dir}/*.csv")
        for analysis_file in analysis_files:
            if archive_old:
                archive_file(analysis_file)
            else:
                os.remove(analysis_file)
                print(f"Supprimé analyse: {analysis_file}")

def archive_file(filepath):
    """Archiver un fichier au lieu de le supprimer"""
    archive_dir = "predictions/archived/"
    os.makedirs(archive_dir, exist_ok=True)
    
    filename = os.path.basename(filepath)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(archive_dir, f"{filename}.{timestamp}")
    
    shutil.move(filepath, archive_path)
    print(f"Archivé: {filepath} -> {archive_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nettoyer le dossier predictions")
    parser.add_argument("--keep-all", action="store_true", help="Ne pas supprimer, juste archiver")
    parser.add_argument("--dry-run", action="store_true", help="Simuler sans supprimer")
    parser.add_argument("--days", type=int, default=3, help="Jours à garder pour évolution")
    
    args = parser.parse_args()
    
    print("=== Nettoyage des prédictions ===")
    print(f"Mode dry-run: {args.dry_run}")
    print(f"Archiver au lieu de supprimer: {args.keep_all}")
    
    if args.dry_run:
        print("\nFichiers qui seraient nettoyés:")
        # Logique de simulation ici
    else:
        clean_predictions(
            keep_latest_only=True,
            days_to_keep=args.days,
            archive_old=args.keep_all
        )
    
    print("\n=== Nettoyage terminé ===")