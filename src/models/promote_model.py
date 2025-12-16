"""
Script pour promouvoir un modèle vers Production dans MLflow Registry
Usage:
    python scripts/promote_model.py --version 1
    python scripts/promote_model.py --auto  # Prend la dernière version avec tag deployment_status=production
"""
import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import argparse

def promote_model(model_name="PremierLeagueModel", version=None, auto=False):
    """
    Promote un modèle vers le stage Production dans MLflow Registry
    
    Args:
        model_name: Nom du modèle dans le Registry
        version: Numéro de version spécifique à promouvoir
        auto: Si True, cherche automatiquement la version avec tag deployment_status=production
    """
    # Configuration DagsHub
    dagshub.init(repo_owner="7a99ui", repo_name="premier_league_mlops", mlflow=True)
    
    client = MlflowClient()
    
    try:
        # Mode automatique : chercher la version avec le bon tag
        if auto:
            print("🔍 Recherche de la version à promouvoir...")
            versions = client.search_model_versions(f"name='{model_name}'")
            
            if not versions:
                print(f"❌ Aucune version du modèle '{model_name}' trouvée")
                return False
            
            # Chercher la version avec deployment_status=production
            target_version = None
            for v in versions:
                if v.tags.get('deployment_status') == 'production':
                    target_version = v.version
                    break
            
            if not target_version:
                # Prendre la dernière version
                target_version = versions[0].version
                print(f"⚠️ Aucune version avec tag 'deployment_status=production', utilisation de la v{target_version}")
            else:
                print(f"✅ Version {target_version} trouvée avec tag 'deployment_status=production'")
            
            version = target_version
        
        if not version:
            print("❌ Aucune version spécifiée")
            return False
        
        print(f"\n{'='*70}")
        print(f"🚀 PROMOTION DU MODÈLE")
        print(f"{'='*70}")
        print(f"Modèle: {model_name}")
        print(f"Version: {version}")
        
        # Vérifier que la version existe
        try:
            model_version = client.get_model_version(model_name, version)
            print(f"\n📊 Informations sur la version {version}:")
            print(f"  Run ID: {model_version.run_id}")
            print(f"  Stage actuel: {model_version.current_stage}")
            if model_version.tags:
                print(f"  Tags:")
                for key, value in model_version.tags.items():
                    print(f"    - {key}: {value}")
        except Exception as e:
            print(f"❌ Version {version} introuvable: {e}")
            return False
        
        # Archiver les modèles en Production actuels
        print(f"\n📦 Archivage des modèles en Production actuels...")
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        for prod_model in production_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=prod_model.version,
                stage="Archived"
            )
            print(f"  ✓ Version {prod_model.version} archivée")
        
        # Promouvoir la nouvelle version
        print(f"\n🎯 Promotion de la version {version} vers Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCÈS !")
        print(f"{'='*70}")
        print(f"Le modèle {model_name} v{version} est maintenant en Production")
        print(f"\n💡 Pour charger ce modèle:")
        print(f"  model = mlflow.sklearn.load_model('models:/PremierLeagueModel/Production')")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la promotion: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Promouvoir un modèle vers Production dans MLflow Registry'
    )
    parser.add_argument(
        '--model-name',
        default='PremierLeagueModel',
        help='Nom du modèle dans MLflow Registry'
    )
    parser.add_argument(
        '--version',
        type=str,
        help='Numéro de version à promouvoir (ex: 1, 2, 3...)'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Promouvoir automatiquement la version avec tag deployment_status=production'
    )
    
    args = parser.parse_args()
    
    if not args.version and not args.auto:
        parser.error("Vous devez spécifier --version ou --auto")
    
    success = promote_model(
        model_name=args.model_name,
        version=args.version,
        auto=args.auto
    )
    
    if not success:
        exit(1)

if __name__ == '__main__':
    main()