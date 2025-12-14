#!/usr/bin/env python3
"""
Script pour nettoyer les modèles MLflow
Utilise avec précaution - les suppressions sont DÉFINITIVES !
"""

import mlflow
from mlflow.tracking import MlflowClient
import argparse
import dagshub

dagshub.init(repo_owner="7a99ui", repo_name="premier_league_mlops", mlflow=True)


def list_all_models(client):
    """Liste tous les modèles enregistrés"""
    print("\n" + "="*70)
    print("📋 MODÈLES ENREGISTRÉS")
    print("="*70)
    
    registered_models = client.search_registered_models()
    
    for rm in registered_models:
        print(f"\n🏷️  Modèle: {rm.name}")
        versions = client.search_model_versions(f"name='{rm.name}'")
        
        for v in versions:
            print(f"   Version {v.version}: {v.current_stage}")
            print(f"      Run ID: {v.run_id}")
            
            # Vérifier si feature_names.json existe
            try:
                artifacts = client.list_artifacts(v.run_id, "features")
                has_features = any('feature_names.json' in a.path for a in artifacts)
                status = "✅" if has_features else "❌"
                print(f"      {status} feature_names.json: {has_features}")
            except Exception as e:
                print(f"      ⚠️  Impossible de vérifier les artifacts: {e}")


def delete_model_version(client, model_name, version):
    """Supprime une version spécifique d'un modèle"""
    try:
        # D'abord, archiver si en Production/Staging
        model_version = client.get_model_version(model_name, version)
        if model_version.current_stage in ["Production", "Staging"]:
            print(f"   📦 Archivage de la version {version}...")
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Archived"
            )
        
        # Ensuite, supprimer
        print(f"   🗑️  Suppression de la version {version}...")
        client.delete_model_version(
            name=model_name,
            version=version
        )
        print(f"   ✅ Version {version} supprimée")
        return True
    except Exception as e:
        print(f"   ❌ Erreur lors de la suppression: {e}")
        return False


def delete_all_versions(client, model_name):
    """Supprime toutes les versions d'un modèle"""
    versions = client.search_model_versions(f"name='{model_name}'")
    
    print(f"\n🗑️  Suppression de toutes les versions de '{model_name}'...")
    
    for v in versions:
        delete_model_version(client, model_name, v.version)
    
    # Supprimer le modèle enregistré lui-même
    try:
        client.delete_registered_model(model_name)
        print(f"✅ Modèle '{model_name}' complètement supprimé")
    except Exception as e:
        print(f"❌ Erreur lors de la suppression du modèle: {e}")


def delete_versions_without_features(client, model_name):
    """Supprime uniquement les versions sans feature_names.json"""
    versions = client.search_model_versions(f"name='{model_name}'")
    
    print(f"\n🧹 Nettoyage des versions sans feature_names.json...")
    
    deleted = 0
    for v in versions:
        try:
            artifacts = client.list_artifacts(v.run_id, "features")
            has_features = any('feature_names.json' in a.path for a in artifacts)
            
            if not has_features:
                print(f"\n❌ Version {v.version} (stage: {v.current_stage}) - PAS de feature_names.json")
                delete_model_version(client, model_name, v.version)
                deleted += 1
            else:
                print(f"✅ Version {v.version} (stage: {v.current_stage}) - feature_names.json OK")
        except Exception as e:
            print(f"⚠️  Version {v.version}: erreur lors de la vérification - {e}")
    
    print(f"\n📊 Résumé: {deleted} version(s) supprimée(s)")


def delete_runs_by_experiment(client, experiment_name):
    """Supprime toutes les runs d'une expérience"""
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"❌ Expérience '{experiment_name}' non trouvée")
        return
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    print(f"\n🗑️  Suppression de {len(runs)} runs de l'expérience '{experiment_name}'...")
    
    for run in runs:
        try:
            client.delete_run(run.info.run_id)
            print(f"   ✅ Run {run.info.run_id} supprimée")
        except Exception as e:
            print(f"   ❌ Erreur pour run {run.info.run_id}: {e}")
    
    print(f"✅ Toutes les runs supprimées")


def archive_production_models(client, model_name):
    """Archive les modèles en Production"""
    prod_versions = client.get_latest_versions(
        name=model_name,
        stages=["Production"]
    )
    
    if not prod_versions:
        print("ℹ️  Aucun modèle en Production")
        return
    
    print(f"\n📦 Archivage des modèles en Production...")
    
    for v in prod_versions:
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived"
            )
            print(f"   ✅ Version {v.version} archivée")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")


def main():
    parser = argparse.ArgumentParser(description='Nettoyage des modèles MLflow')
    parser.add_argument('--action', 
                       choices=['list', 'delete-all', 'delete-without-features', 
                               'archive-production', 'delete-runs'],
                       required=True,
                       help='Action à effectuer')
    parser.add_argument('--model-name', 
                       default='PremierLeagueModel',
                       help='Nom du modèle (par défaut: PremierLeagueModel)')
    parser.add_argument('--experiment-name',
                       default='PremierLeague-Training',
                       help='Nom de l\'expérience (pour delete-runs)')
    parser.add_argument('--confirm',
                       action='store_true',
                       help='Confirmer la suppression (obligatoire pour les actions destructives)')
    
    args = parser.parse_args()
    
    # Configuration MLflow
    mlflow.set_tracking_uri("https://dagshub.com/7a99ui/premier_league_mlops.mlflow")
    client = MlflowClient()
    
    print("="*70)
    print("🧹 NETTOYAGE MLFLOW")
    print("="*70)
    
    if args.action == 'list':
        # Pas besoin de confirmation pour lister
        list_all_models(client)
    
    elif args.action == 'delete-all':
        if not args.confirm:
            print("\n⚠️  ATTENTION: Cette action supprimera TOUTES les versions du modèle!")
            print("   Utilise --confirm pour confirmer")
            return
        
        print(f"\n⚠️  Suppression de TOUTES les versions de '{args.model_name}'...")
        confirm = input("   Tapes 'YES' pour confirmer: ")
        if confirm == 'YES':
            delete_all_versions(client, args.model_name)
        else:
            print("   ❌ Annulé")
    
    elif args.action == 'delete-without-features':
        if not args.confirm:
            print("\n⚠️  Cette action supprimera les versions sans feature_names.json")
            print("   Utilise --confirm pour confirmer")
            return
        
        delete_versions_without_features(client, args.model_name)
    
    elif args.action == 'archive-production':
        archive_production_models(client, args.model_name)
    
    elif args.action == 'delete-runs':
        if not args.confirm:
            print("\n⚠️  Cette action supprimera TOUTES les runs de l'expérience!")
            print("   Utilise --confirm pour confirmer")
            return
        
        print(f"\n⚠️  Suppression de toutes les runs de '{args.experiment_name}'...")
        confirm = input("   Tapes 'YES' pour confirmer: ")
        if confirm == 'YES':
            delete_runs_by_experiment(client, args.experiment_name)
        else:
            print("   ❌ Annulé")
    
    print("\n✅ Opération terminée!")


if __name__ == '__main__':
    main()