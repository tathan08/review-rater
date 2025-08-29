#!/usr/bin/env python3
"""
Pipeline Status Test
Tests the core functionality without importing problematic libraries
"""
import sys
import os
sys.path.append('src')

def test_basic_functionality():
    """Test basic pipeline functionality"""
    print("=== PIPELINE STATUS CHECK ===")
    print()
    
    # Test 1: Configuration system
    print("1. Testing configuration system...")
    try:
        from src.config.pipeline_config import config
        print(f"   ✅ Config loaded successfully")
        print(f"   ✅ Model config: {config.model.ollama_model}")
        print(f"   ✅ Data config: {config.data.sample_reviews_file}")
        print(f"   ✅ Output config: {config.output.predictions_dir}")
    except Exception as e:
        print(f"   ❌ Config failed: {e}")
    
    # Test 2: Data utilities  
    print()
    print("2. Testing data utilities...")
    try:
        # Simple sample data loading without demo_helpers
        sample_file = "data/sample/sample_reviews.csv"
        import pandas as pd
        df = pd.read_csv(sample_file)
        print(f"   ✅ Sample data loaded: {len(df)} reviews")
        print(f"   ✅ Columns: {list(df.columns)}")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
    
    # Test 3: Core constants
    print()
    print("3. Testing core constants...")
    try:
        from src.core.constants import POLICY_CATEGORIES, LABELS
        print(f"   ✅ Policy categories: {list(POLICY_CATEGORIES.keys())}")
        print(f"   ✅ Labels: {list(LABELS.keys())}")
    except Exception as e:
        print(f"   ❌ Constants failed: {e}")
    
    # Test 4: Pseudo-labeling structure
    print()
    print("4. Testing pseudo-labeling structure...")
    try:
        from src.pseudo_labelling.gpt_labeller import GPTPseudoLabeler
        labeler = GPTPseudoLabeler(config)
        print("   ✅ Pseudo-labeling system initialized")
        print("   ✅ Policy systems configured for:", list(labeler.policy_systems.keys()))
    except Exception as e:
        print(f"   ❌ Pseudo-labeling failed: {e}")
    
    # Test 5: Prompt system
    print()
    print("5. Testing prompt system...")
    try:
        from prompts.policy_prompts import build_prompt, NO_ADS_SYSTEM, FEW_SHOTS_NO_ADS
        prompt = build_prompt(NO_ADS_SYSTEM, "Test review", FEW_SHOTS_NO_ADS)
        print("   ✅ Prompt building system working")
        print(f"   ✅ Generated prompt length: {len(prompt)} characters")
    except Exception as e:
        print(f"   ❌ Prompt system failed: {e}")
    
    # Test 6: File structure
    print()
    print("6. Testing file structure...")
    directories = [
        "data/sample",
        "data/raw", 
        "data/processed",
        "results/predictions",
        "results/evaluations",
        "results/reports",
        "logs/pipeline_logs",
        "models/saved_models",
        "models/cache"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"   ✅ {directory}")
        else:
            print(f"   ❌ {directory}")
    
    # Test 7: Sample data file
    print()
    print("7. Testing sample data file...")
    sample_file = "data/sample/sample_reviews.csv"
    if os.path.exists(sample_file):
        import pandas as pd
        df = pd.read_csv(sample_file)
        print(f"   ✅ {sample_file} exists with {len(df)} reviews")
    else:
        print(f"   ❌ {sample_file} not found")

if __name__ == "__main__":
    test_basic_functionality()
