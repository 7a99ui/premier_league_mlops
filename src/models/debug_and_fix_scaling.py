"""
Script: debug_and_fix_scaling.py
Objectif: Diagnostiquer et résoudre le problème d'incompatibilité
          entre les features du modèle v1 et les données v2.
"""

import pandas as pd
import joblib
import json
from pathlib import Path

def debug_and_fix():
    project_root = Path(__file__).parent.parent
    
    print("🔍 DIAGNOSTIC - Compatibilité Modèle (v1) ↔ Données (v2)")
    print("="*60)
    
    # 1. Charger la liste des features du modèle (v1)
    metadata_path = project_root / 'models' / 'production' / 'latest_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_features = metadata.get('feature_names', [])
    print(f"✅ Modèle chargé ({len(model_features)} features attendues)")
    print(f"   5 premières features: {model_features[:5]}")
    
    # 2. Charger les données v2
    v2_path = project_root / 'data' / 'processed' / 'v2' / 'features.parquet'
    df_v2 = pd.read_parquet(v2_path)
    
    # Extraire les noms de features des données v2
    v2_data_features = [col for col in df_v2.columns 
                       if col not in ['team', 'season', 'gameweek', 'target_final_points',
                                     'target_final_position', 'projected_points']]
    
    print(f"✅ Données v2 chargées ({len(v2_data_features)} features trouvées)")
    print(f"   5 premières features: {v2_data_features[:5]}")
    
    # 3. Analyser les différences
    print(f"\n📊 ANALYSE DES DIFFÉRENCES")
    print(f"   Features dans le modèle mais PAS dans v2:")
    missing_in_data = set(model_features) - set(v2_data_features)
    for feat in sorted(missing_in_data)[:5]:  # Afficher les 5 premiers
        print(f"     - {feat}")
    
    print(f"\n   Features dans v2 mais PAS dans le modèle:")
    extra_in_data = set(v2_data_features) - set(model_features)
    for feat in sorted(extra_in_data)[:5]:  # Afficher les 5 premiers
        print(f"     - {feat}")
    
    # 4. Charger le scaler v1
    v1_scaler_path = project_root / 'data' / 'processed' / 'v1' / 'scaler.joblib'
    scaler = joblib.load(v1_scaler_path)
    print(f"\n✅ Scaler v1 chargé ({len(scaler.mean_)} features scalées)")
    
    # 5. Résumé et solution
    print(f"\n🎯 DIAGNOSTIC TERMINÉ - RÉSUMÉ")
    print(f"   Le modèle attend {len(model_features)} features spécifiques.")
    print(f"   Les données v2 ont {len(v2_data_features)} features.")
    print(f"   {len(missing_in_data)} features manquent dans v2.")
    print(f"   {len(extra_in_data)} features sont en trop dans v2.")
    
    # Solution recommandée
    if missing_in_data:
        print(f"\n⚠️  SOLUTION REQUISE:")
        print(f"   1. Les features suivantes doivent être AJOUTÉES à vos données v2:")
        for feat in sorted(missing_in_data):
            print(f"      - {feat}")
        print(f"   2. Si elles n'existent pas, initialisez-les à 0 ou à la médiane.")
    
    # Créer un DataFrame ajusté pour test
    print(f"\n🧪 Création d'un DataFrame v2 ajusté pour test...")
    
    # Prendre un échantillon (par exemple, GW38 de la saison 2023-2024)
    sample_data = df_v2[
        (df_v2['season'] == '2023-2024') & 
        (df_v2['gameweek'] == 38)
    ].copy()
    
    # Créer un DataFrame avec toutes les features du modèle
    adjusted_df = pd.DataFrame()
    
    for feature in model_features:
        if feature in sample_data.columns:
            adjusted_df[feature] = sample_data[feature]
        else:
            print(f"   ⚠️  Initialisation de '{feature}' à 0 (valeur manquante)")
            adjusted_df[feature] = 0
    
    # Afficher la forme finale
    print(f"   ✅ DataFrame ajusté créé: {adjusted_df.shape}")
    print(f"   Colonnes: {list(adjusted_df.columns)[:5]}...")
    
    # 6. Tester le scaling sur l'échantillon ajusté
    print(f"\n🔧 TEST DU SCALING V1 SUR DONNÉES AJUSTÉES")
    try:
        scaled_sample = scaler.transform(adjusted_df)
        print(f"   ✅ Scaling réussi!")
        print(f"   Forme après scaling: {scaled_sample.shape}")
        print(f"   Moyenne (première feature): {scaled_sample[:, 0].mean():.3f}")
        print(f"   Std (première feature): {scaled_sample[:, 0].std():.3f}")
    except Exception as e:
        print(f"   ❌ Erreur de scaling: {e}")
    
    return model_features, v2_data_features, missing_in_data

if __name__ == '__main__':
    debug_and_fix()