"""
Feature Alignment Utility
Aligns new data features to match training data format
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

class FeatureAligner:
    """Aligns features between different data versions"""
    
    def __init__(self):
        project_root = Path(__file__).parent.parent.parent
        # Chemin vers scaler v1
        self.scaler_path = project_root / 'data' / 'processed' / 'v1' / 'scaler.joblib'
    
    def load_model_features(self, metadata_path):
        """Load feature list from model metadata"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.model_features = metadata.get('feature_names', [])
        print(f"✅ Loaded {len(self.model_features)} model features")
    
    def load_scaler(self, scaler_path):
        """Load scaler from v1"""
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Loaded scaler with {len(self.scaler.mean_)} features")
        else:
            print(f"⚠️  Scaler not found at {scaler_path}")
    
    def align_features(self, new_data_df, fill_missing=0, drop_extra=True):
        """
        Align new data features to match model features
        
        Args:
            new_data_df: DataFrame with new data (v2, v3, etc.)
            fill_missing: Value to fill missing features (default: 0)
            drop_extra: Whether to drop extra features (default: True)
        
        Returns:
            DataFrame with aligned features
        """
        if not self.model_features:
            raise ValueError("Model features not loaded")
        
        print(f"🔍 Aligning features...")
        print(f"   New data has: {len(new_data_df.columns)} columns")
        
        # Identify metadata columns
        metadata_cols = ['team', 'season', 'gameweek', 'target_final_points',
                        'target_final_position', 'projected_points']
        metadata_present = [col for col in metadata_cols if col in new_data_df.columns]
        
        # Separate metadata and features
        if metadata_present:
            metadata_df = new_data_df[metadata_present].copy()
            feature_df = new_data_df.drop(columns=metadata_present, errors='ignore')
        else:
            metadata_df = pd.DataFrame()
            feature_df = new_data_df.copy()
        
        # Create aligned feature DataFrame
        aligned_features = pd.DataFrame()
        
        for feature in self.model_features:
            if feature in feature_df.columns:
                aligned_features[feature] = feature_df[feature]
            else:
                print(f"   ⚠️  Adding missing feature '{feature}' with value {fill_missing}")
                aligned_features[feature] = fill_missing
        
        print(f"   ✅ Aligned features: {aligned_features.shape}")
        
        # Handle extra features
        extra_features = set(feature_df.columns) - set(self.model_features)
        if extra_features and not drop_extra:
            print(f"   ⚠️  Keeping {len(extra_features)} extra features")
            for feat in extra_features:
                aligned_features[feat] = feature_df[feat]
        
        # Recombine with metadata
        if not metadata_df.empty:
            aligned_data = pd.concat([metadata_df.reset_index(drop=True), 
                                     aligned_features.reset_index(drop=True)], axis=1)
        else:
            aligned_data = aligned_features
        
        return aligned_data
    
    def transform_with_scaler(self, aligned_data):
        """
        Apply v1 scaler to aligned data
        
        Args:
            aligned_data: DataFrame with aligned features
        
        Returns:
            Scaled data array
        """
        if self.scaler is None:
            raise ValueError("Scaler not loaded")
        
        # Extract just the features (excluding metadata)
        feature_cols = [col for col in aligned_data.columns 
                       if col not in ['team', 'season', 'gameweek', 'target_final_points',
                                     'target_final_position', 'projected_points']]
        
        X = aligned_data[feature_cols]
        
        print(f"🔧 Applying scaler to {X.shape[1]} features...")
        X_scaled = self.scaler.transform(X)
        
        print(f"   Before scaling - Mean: {X.mean().mean():.3f}")
        print(f"   After scaling  - Mean: {X_scaled.mean():.3f}")
        
        return X_scaled

# Usage example
if __name__ == '__main__':
    # Example usage
    aligner = FeatureAligner()
    aligner.load_model_features('models/production/latest_metadata.json')
    aligner.load_scaler('data/processed/v1/scaler.joblib')
    
    # Load new data (v2, v3, etc.)
    new_data = pd.read_parquet('data/processed/v2/features.parquet')
    
    # Align features
    aligned = aligner.align_features(new_data)
    
    # Apply scaling
    scaled = aligner.transform_with_scaler(aligned)
    
    print(f"🎉 Ready for prediction!")