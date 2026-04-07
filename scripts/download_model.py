#!/usr/bin/env python3
"""
Download TimesFM 2.5 model from HuggingFace
"""

import os
from huggingface_hub import hf_hub_download, snapshot_download

def download_model():
    """Download TimesFM 2.5 200M model"""
    
    print("Downloading TimesFM 2.5 200M model...")
    print("This may take a few minutes depending on your connection.")
    
    model_name = "google/timesfm-2.5-200m"
    
    try:
        # Download the full model
        local_path = snapshot_download(
            repo_id=model_name,
            local_dir="./models/timesfm-2.5-200m",
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model downloaded successfully to: {local_path}")
        print(f"   Size: ~800MB")
        
        return local_path
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("   The server will use mock predictions instead.")
        return None

if __name__ == "__main__":
    download_model()
