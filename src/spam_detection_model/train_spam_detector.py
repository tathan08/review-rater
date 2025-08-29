#!/usr/bin/env python3
"""
Train and evaluate the unified spam detection system.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime
sys.path.append('src')
from spam_detection import (
    UnifiedSpamDetector,
    load_training_data, 
    evaluate_detector
)

def main():
    parser = argparse.ArgumentParser(description='Train unified spam detector')
    parser.add_argument('--train_data', default='../../data/sample_reviews.csv',
                       help='Training data CSV file')
    parser.add_argument('--test_data', default=None,
                       help='Test data CSV file (if different from train)')
    parser.add_argument('--output_dir', default='outputs',
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Spam detection threshold')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Loading training data...")
    texts, labels, categories = load_training_data(args.train_data)
    
    print(f"Loaded {len(texts)} samples:")
    print(f"  - APPROVE: {labels.count('APPROVE')}")
    print(f"  - REJECT: {labels.count('REJECT')}")
    
    # Initialize and train unified detector
    print("\nInitializing unified spam detector...")
    detector = UnifiedSpamDetector(spam_threshold=args.threshold)
    
    print("Training unified detector...")
    detector.fit(texts, labels, categories)
    
    # Evaluate on training data (or test data if provided)
    test_texts, test_labels, test_categories = texts, labels, categories
    if args.test_data:
        print("Loading test data...")
        test_texts, test_labels, test_categories = load_training_data(args.test_data)
    
    print("\nEvaluating detector...")
    eval_results = evaluate_detector(
        detector, test_texts, test_labels, test_categories
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nBinary Classification (APPROVE/REJECT):")
    binary_results = eval_results['binary_classification']
    print(f"  Accuracy:  {binary_results['accuracy']:.3f}")
    print(f"  Precision: {binary_results['precision']:.3f}")
    print(f"  Recall:    {binary_results['recall']:.3f}")
    print(f"  F1-Score:  {binary_results['f1']:.3f}")
    
    print("\nConfidence Score Statistics:")
    conf_stats = eval_results['confidence_stats']
    print(f"  Mean: {conf_stats['mean']:.3f}")
    print(f"  Std:  {conf_stats['std']:.3f}")
    print(f"  Min:  {conf_stats['min']:.3f}")
    print(f"  Max:  {conf_stats['max']:.3f}")
    
    # Show feature importance
    print("\nTop Important Features:")
    try:
        feature_importance = detector.get_feature_importance(15)
        for i, (feature, importance) in enumerate(feature_importance, 1):
            print(f"  {i:2d}. {feature[:50]:50} {importance:8.4f}")
    except Exception as e:
        print(f"  Could not extract feature importance: {e}")
    
    # Save detailed results
    output_file = output_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Generate predictions on full dataset
    print("\nGenerating predictions...")
    results = detector.predict(test_texts)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'text': [text[:200] + '...' if len(text) > 200 else text for text in test_texts],
        'true_label': test_labels,
        'true_category': test_categories,
        'pred_label': [r.label for r in results],
        'pred_category': [r.category for r in results],
        'confidence': [r.confidence for r in results],
        'spam_probability': [r.features['spam_probability'] for r in results],
        'ci_lower': [r.confidence_interval[0] for r in results],
        'ci_upper': [r.confidence_interval[1] for r in results]
    })
    
    # Add pattern feature analysis
    for i, result in enumerate(results):
        features = result.features['pattern_features']
        predictions_df.loc[i, 'repetition_ratio'] = features['repetition_ratio']
        predictions_df.loc[i, 'phrase_repetition_score'] = features['phrase_repetition_score']
        predictions_df.loc[i, 'repeated_ngrams_ratio'] = features['repeated_ngrams_ratio']
        predictions_df.loc[i, 'word_diversity_ratio'] = features['word_diversity_ratio']
    
    # Save predictions
    pred_file = output_dir / 'predictions_unified.csv'
    predictions_df.to_csv(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")
    
    # Show some example predictions
    print("\nSample Predictions:")
    print("-" * 120)
    sample_size = min(10, len(predictions_df))
    for _, row in predictions_df.head(sample_size).iterrows():
        print(f"Text: {row['text']}")
        print(f"True: {row['true_label']} ({row['true_category']})")
        print(f"Pred: {row['pred_label']} ({row['pred_category']}) - Confidence: {row['confidence']:.3f}")
        print(f"Pattern Features: Repetition={row['repetition_ratio']:.2f}, "
              f"Phrase_Rep={row['phrase_repetition_score']:.2f}, "
              f"NGram_Rep={row['repeated_ngrams_ratio']:.2f}")
        print()
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"Training samples: {len(texts)}")
    print(f"Model type: Unified ML + Pattern Analysis")
    print(f"Detection threshold: {args.threshold}")
    print(f"Overall F1-Score: {binary_results['f1']:.3f}")
    print(f"\nTraining and evaluation complete!")
    print(f"Results saved in: {output_dir}")

if __name__ == '__main__':
    main()