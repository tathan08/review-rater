#!/usr/bin/env python3
"""
Interactive Spam Checker with Unified ML + Pattern Analysis
Enter text and get instant spam detection results.
"""

import sys
import os
import pandas as pd
# Add the current directory and parent directory to path
sys.path.append('.')
sys.path.append('..')

def initialize_detector():
    """Initialize and train the unified spam detector."""
    try:
        from spam_detection import UnifiedSpamDetector, load_training_data
        
        # Check if training data exists (adjust path for new location)
        data_path = '../../data/sample_reviews.csv'
        if not os.path.exists(data_path):
            print("❌ Error: Training data not found")
            print("Please ensure you're running from the correct directory")
            return None
            
        print("🔄 Loading unified spam detector...")
        texts, labels, categories = load_training_data(data_path)
        detector = UnifiedSpamDetector()
        detector.fit(texts, labels, categories)
        print("✅ Unified spam detector ready!")
        return detector
        
    except ImportError:
        print("❌ Error: Cannot import spam detection modules")
        print("Make sure you're in the correct directory and dependencies are installed")
        return None
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return None

def check_text(detector, text):
    """Check if text is spam and display detailed results."""
    try:
        results = detector.predict([text])
        result = results[0]
        
        print(f"\n📝 Text: \"{text}\"")
        print("-" * 60)
        
        # Main result
        if result.label == 'REJECT':
            print("🚨 RESULT: SPAM DETECTED")
            print(f"📂 Category: {result.category}")
        else:
            print("✅ RESULT: CLEAN TEXT")
            
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"📊 Spam Probability: {result.features['spam_probability']:.3f}")
        print(f"🔒 Confidence Interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
        
        # Pattern analysis breakdown
        pattern_features = result.features['pattern_features']
        print("\n🔍 Pattern Analysis:")
        
        # Repetition analysis
        if pattern_features['repetition_ratio'] > 0.3:
            print(f"   🔄 Word Repetition: {pattern_features['repetition_ratio']:.3f} {'🚨' if pattern_features['repetition_ratio'] > 0.5 else '⚠️'}")
        
        if pattern_features['phrase_repetition_score'] > 0.3:
            print(f"   📝 Phrase Patterns: {pattern_features['phrase_repetition_score']:.3f} {'🚨' if pattern_features['phrase_repetition_score'] > 0.5 else '⚠️'}")
        
        if pattern_features['repeated_ngrams_ratio'] > 0.3:
            print(f"   🔗 N-gram Repetition: {pattern_features['repeated_ngrams_ratio']:.3f} {'🚨' if pattern_features['repeated_ngrams_ratio'] > 0.4 else '⚠️'}")
        
        # Text quality indicators
        print(f"   📊 Word Diversity: {pattern_features['word_diversity_ratio']:.3f}")
        print(f"   📖 Readability Score: {pattern_features['readability_score']:.1f}")
        
        # Style indicators
        if pattern_features['caps_ratio'] > 0.2:
            print(f"   🔠 Excessive Caps: {pattern_features['caps_ratio']:.3f} ⚠️")
        if pattern_features['template_score'] > 0.4:
            print(f"   📋 Template-like: {pattern_features['template_score']:.3f} ⚠️")
        
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error analyzing text: {e}")

def filter_clean_text_from_csv(text_input, detector=None):
    """
    Check if a single text input is clean (not spam).
    
    Args:
        text_input: Single text string to check
        detector: Optional pre-initialized detector (for efficiency in loops)
    
    Returns:
        The text if clean, None if spam
    """
    if detector is None:
        detector = initialize_detector()
        if not detector:
            return None
    
    if not text_input or not text_input.strip():
        return None
        
    results = detector.predict([text_input])
    if results[0].label == 'APPROVE':
        return text_input
    else:
        return None

def interactive_mode():
    """Interactive mode for checking individual texts."""
    # Initialize detector
    detector = initialize_detector()
    if not detector:
        return
    
    print("\n💡 Try examples like:")
    print("  • 'Food is good food is great food is nice'  (repetitive pattern)")
    print("  • 'hi hi hi hi hi hi hi hi'  (extreme repetition)")
    print("  • 'Great restaurant with excellent food!'  (normal review)")
    print("  • 'Amazing deal amazing price amazing food'  (phrase repetition)")
    print("\n" + "=" * 60)
    
    # Interactive loop
    while True:
        try:
            text = input("\n🔍 Enter text to check: ").strip()
            
            # Check for quit commands
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            # Skip empty input
            if not text:
                continue
                
            # Check the text
            check_text(detector, text)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def csv_filter_mode():
    """CSV filtering mode."""
    detector = initialize_detector()
    if not detector:
        return
    
    print("\n📂 CSV Filter Mode")
    print("Enter text lines to filter. Only clean (non-spam) text will be returned.")
    print("Type 'done' to finish.\n")
    
    clean_texts = []
    
    while True:
        try:
            text = input("Enter text: ").strip()
            
            if text.lower() == 'done':
                break
                
            if not text:
                continue
                
            clean_text = filter_clean_text_from_csv(text, detector)
            if clean_text:
                clean_texts.append(clean_text)
                print(f"✅ CLEAN: {clean_text}")
            else:
                print(f"🚨 SPAM: {text}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n📊 Summary: {len(clean_texts)} clean texts collected")
    if clean_texts:
        print("\nClean texts:")
        for i, text in enumerate(clean_texts, 1):
            print(f"{i}. {text}")

def main():
    """Main function with mode selection."""
    print("🛡️  UNIFIED SPAM DETECTION TOOL")
    print("=" * 60)
    print("ML + Pattern Analysis for Advanced Spam Detection")
    print("Detects repetitive patterns, phrase repetition, and learned spam signals")
    print("=" * 60)
    
    print("\nSelect mode:")
    print("1. Interactive Mode - Check individual texts")
    print("2. CSV Filter Mode - Filter multiple texts")
    
    while True:
        try:
            choice = input("\nEnter choice (1 or 2): ").strip()
            
            if choice == '1':
                interactive_mode()
                break
            elif choice == '2':
                csv_filter_mode()
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

if __name__ == '__main__':
    main()