"""
Unified spam detection system combining machine learning with pattern analysis.
Assumes data has already passed through toxicity/hate speech filtering pipeline.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix

import textstat
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DetectionResult:
    """Container for detection results."""
    label: str  # 'APPROVE' or 'REJECT'
    confidence: float  # 0.0 to 1.0
    category: str  # violation category or 'None'
    features: Dict  # method-specific features
    confidence_interval: Tuple[float, float]

class PatternFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract pattern-based features for ML model."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract pattern features from texts."""
        features = []
        
        for text in X:
            text_features = self._extract_pattern_features(text)
            
            # Convert to feature vector
            feature_vector = [
                text_features['repetition_ratio'],
                text_features['word_diversity_ratio'],
                text_features['repeated_ngrams_ratio'],
                text_features['caps_ratio'],
                text_features['punct_ratio'],
                text_features['readability_score'] / 100.0,  # Normalize
                text_features['avg_sentence_length'] / 50.0,  # Normalize
                text_features['word_count'] / 100.0,  # Normalize
                text_features['template_score'],
                text_features['phrase_repetition_score'],
                text_features['local_repetition_score']
            ]
            features.append(feature_vector)
            
        return np.array(features)
    
    def _extract_pattern_features(self, text: str) -> Dict:
        """Extract comprehensive pattern features from text."""
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Basic repetition analysis
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        repetition_ratio = 0.0
        if word_count > 0:
            max_repetition = max(word_freq.values()) if word_freq else 1
            repetition_ratio = max_repetition / word_count
        
        # Word diversity (unique words / total words)
        unique_words = len(set(word.lower() for word in words))
        word_diversity_ratio = unique_words / max(word_count, 1)
        
        # N-gram repetition detection (for "food is good food is great food is nice")
        repeated_ngrams_ratio = self._detect_repeated_ngrams(words)
        
        # Phrase repetition score (detects patterns like "food is X" repeating)
        phrase_repetition_score = self._detect_phrase_patterns(words)
        
        # Local repetition score (detects repetition in specific text segments)
        local_repetition_score = self._detect_local_repetition(words)
        
        # Template detection (common patterns)
        template_indicators = [
            r'\b(excellent|amazing|great|good|bad|terrible)\b.*\b(food|service|place|restaurant)\b',
            r'\b(recommend|suggest|try|visit)\b.*\b(place|restaurant|here)\b',
            r'\b(will|would).*\b(come|go|visit).*\b(again|back)\b'
        ]
        
        template_matches = sum(1 for pattern in template_indicators 
                             if re.search(pattern, text, re.IGNORECASE))
        template_score = template_matches / len(template_indicators)
        
        # Readability (using textstat)
        try:
            # Use textstat if available (ignore lint warnings)
            readability_score = textstat.flesch_reading_ease(text)  # type: ignore
        except Exception as e:
            # Fallback readability calculation
            words = len(text.split())
            sentences = len(re.split(r'[.!?]+', text))
            if sentences > 0 and words > 0:
                avg_words_per_sentence = words / sentences
                # Simple readability approximation (lower is harder to read)
                readability_score = max(0, min(100, 120 - avg_words_per_sentence * 2))
            else:
                readability_score = 50.0
        
        # Character-level features
        caps_count = sum(1 for c in text if c.isupper())
        punct_count = sum(1 for c in text if c in '!?.,;:')
        
        caps_ratio = caps_count / max(char_count, 1)
        punct_ratio = punct_count / max(char_count, 1)
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'repetition_ratio': repetition_ratio,
            'word_diversity_ratio': word_diversity_ratio,
            'repeated_ngrams_ratio': repeated_ngrams_ratio,
            'phrase_repetition_score': phrase_repetition_score,
            'local_repetition_score': local_repetition_score,
            'template_score': template_score,
            'readability_score': readability_score,
            'caps_ratio': caps_ratio,
            'punct_ratio': punct_ratio,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length
        }
    
    def _detect_repeated_ngrams(self, words: List[str], n: int = 2) -> float:
        """Detect repeated n-grams that indicate spam patterns."""
        if len(words) < n * 2:  # Need at least 2 n-grams to compare
            return 0.0
        
        # Generate n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n]).lower()
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        # Count n-gram frequencies
        ngram_freq = {}
        for ngram in ngrams:
            ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1
        
        # Calculate repetition score
        total_ngrams = len(ngrams)
        repeated_ngrams = sum(1 for freq in ngram_freq.values() if freq > 1)
        
        return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    def _detect_phrase_patterns(self, words: List[str]) -> float:
        """Detect repeating phrase patterns like 'food is X food is Y food is Z'."""
        if len(words) < 6:  # Need enough words for pattern detection
            return 0.0
        
        # Look for patterns where same 2-word prefix repeats
        pattern_scores = []
        
        # Check 2-word patterns
        for i in range(len(words) - 3):
            prefix = f"{words[i]} {words[i+1]}".lower()
            
            # Count how many times this prefix appears
            prefix_count = 0
            for j in range(i, len(words) - 1):
                if j + 1 < len(words):
                    candidate = f"{words[j]} {words[j+1]}".lower()
                    if candidate == prefix:
                        prefix_count += 1
            
            if prefix_count > 1:
                # This prefix repeats - calculate pattern strength
                pattern_strength = prefix_count / (len(words) / 3)  # Normalize by text length
                pattern_scores.append(min(pattern_strength, 1.0))
        
        # Also check for local repetition at the end of text (common spam pattern)
        local_repetition_score = self._detect_local_repetition(words)
        
        return max(max(pattern_scores) if pattern_scores else 0.0, local_repetition_score)
    
    def _detect_local_repetition(self, words: List[str]) -> float:
        """Detect repetitive patterns in specific sections of text (e.g., end of text)."""
        if len(words) < 6:
            return 0.0
        
        max_score = 0.0
        
        # Check multiple segments: end of text and any suspicious consecutive patterns
        segments_to_check = []
        
        # 1. End segment (common spam pattern)
        end_segment_size = min(10, max(6, len(words) // 3))
        end_words = words[-end_segment_size:]
        segments_to_check.append(('end', end_words))
        
        # 2. Look for any consecutive repeated phrases throughout the text
        for start_pos in range(len(words) - 6):  # Sliding window
            window_size = min(8, len(words) - start_pos)
            window_words = words[start_pos:start_pos + window_size]
            segments_to_check.append(('window', window_words))
        
        for segment_type, segment_words in segments_to_check:
            # Look for exact phrase repetition in this segment
            for phrase_len in [2, 3, 4]:  # Check 2-word, 3-word, 4-word phrases
                if len(segment_words) < phrase_len * 2:  # Need at least 2 instances
                    continue
                    
                for i in range(len(segment_words) - phrase_len + 1):
                    phrase = ' '.join(segment_words[i:i+phrase_len]).lower()
                    
                    # Look for consecutive repetition (more suspicious than scattered)
                    consecutive_count = 1
                    next_pos = i + phrase_len
                    
                    while next_pos + phrase_len <= len(segment_words):
                        next_phrase = ' '.join(segment_words[next_pos:next_pos+phrase_len]).lower()
                        if next_phrase == phrase:
                            consecutive_count += 1
                            next_pos += phrase_len
                        else:
                            break
                    
                    if consecutive_count >= 2:  # Found consecutive repeated phrase
                        # Higher score for consecutive repetition
                        if segment_type == 'end':
                            # End segment repetition is more suspicious
                            local_score = (consecutive_count * phrase_len * 1.5) / len(segment_words)
                        else:
                            # General repetition
                            local_score = (consecutive_count * phrase_len) / len(segment_words)
                        
                        max_score = max(max_score, min(local_score, 1.0))
        
        return max_score


class UnifiedSpamDetector:
    """Unified spam detector combining TF-IDF and pattern analysis in single ML model."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 3), 
                 spam_threshold: float = 0.3):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.spam_threshold = spam_threshold
        self.pipeline = None
        self.calibrated_pipeline = None
        self.training_size = 0
        
    def fit(self, texts: List[str], labels: List[str], calibrate: bool = True):
        """
        Fit the unified model combining TF-IDF and pattern features.
        
        Args:
            texts: List of review texts
            labels: List of labels ('APPROVE' or 'REJECT')
            calibrate: Whether to calibrate probabilities
        """
        # Convert labels to binary
        y = [1 if label == 'REJECT' else 0 for label in labels]
        self.training_size = len(texts)
        
        # For small datasets, disable ML-based classification and rely on patterns only
        if len(texts) < 10:
            # With very small training data, ML is unreliable - use pattern-only mode
            self.effective_threshold = 0.9  # Very high threshold to disable ML classification
            self.pattern_only_mode = True
            print(f"⚠️  Very small training set ({len(texts)} samples). Using pattern-only detection mode.")
        elif len(texts) < 20:
            # Small dataset - be more conservative with ML threshold
            adjusted_threshold = min(0.6, self.spam_threshold + 0.2)
            print(f"⚠️  Small training set ({len(texts)} samples). Adjusting threshold: {self.spam_threshold:.2f} → {adjusted_threshold:.2f}")
            self.effective_threshold = adjusted_threshold
            self.pattern_only_mode = False
        else:
            self.effective_threshold = self.spam_threshold
            self.pattern_only_mode = False
        
        # Create unified pipeline with both TF-IDF and pattern features
        tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.95
        )
        
        pattern_extractor = PatternFeatureExtractor()
        
        # Combine features using FeatureUnion
        feature_union = FeatureUnion([  # type: ignore
            ('tfidf', tfidf_vectorizer),
            ('patterns', pattern_extractor)
        ])
        
        self.pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ))
        ])
        
        # Fit the pipeline
        print("Training unified ML + Pattern model...")
        self.pipeline.fit(texts, y)
        
        # Calibrate probabilities if requested
        if calibrate and len(texts) >= 10:
            min_class_size = min(sum(y), len(y) - sum(y))
            cv_folds = min(3, max(2, min_class_size))
            
            self.calibrated_pipeline = CalibratedClassifierCV(
                self.pipeline, 
                method='isotonic',
                cv=cv_folds
            )
            self.calibrated_pipeline.fit(texts, y)  # type: ignore
        elif calibrate:
            print(f"Skipping calibration: need at least 10 samples, got {len(texts)}")
    
    def predict(self, texts: List[str], use_calibrated: bool = True) -> List[DetectionResult]:
        """Predict spam using unified model."""
        if self.pipeline is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        pipeline = self.calibrated_pipeline if (use_calibrated and self.calibrated_pipeline) else self.pipeline
        
        # Get predictions and probabilities
        proba = pipeline.predict_proba(texts)  # type: ignore
        spam_proba = proba[:, 1]  # Probability of spam (REJECT)
        
        results = []
        for i, prob in enumerate(spam_proba):
            # Extract pattern features for analysis
            pattern_extractor = PatternFeatureExtractor()
            text_features = pattern_extractor._extract_pattern_features(texts[i])
            
            # Check if we're in pattern-only mode (small training data)
            pattern_only_mode = getattr(self, 'pattern_only_mode', False)
            threshold = getattr(self, 'effective_threshold', self.spam_threshold)
            
            if pattern_only_mode:
                # Pattern-only classification for small datasets
                word_count = text_features['word_count']
                
                # Define clear spam patterns
                is_obvious_spam = (
                    (text_features['repetition_ratio'] > 0.8) or  # Very high word repetition 
                    (text_features['phrase_repetition_score'] > 0.7) or  # Very high phrase repetition
                    (text_features['repeated_ngrams_ratio'] > 0.6) or  # Very high ngram repetition
                    (text_features['local_repetition_score'] > 0.4) or  # Local repetition (like "food is good food is good")
                    (word_count < 3 and text_features['repetition_ratio'] > 0.6)  # Short repetitive text
                )
                
                # Special cases for high-confidence spam detection
                is_phrase_spam = text_features['phrase_repetition_score'] > 0.9
                is_local_spam = text_features['local_repetition_score'] > 0.5
                
                # Additional filters to avoid false positives (but not for phrase spam)
                seems_legitimate = (
                    text_features['readability_score'] > 30 and 
                    word_count > 8 and 
                    text_features['word_diversity_ratio'] > 0.6 and
                    text_features['phrase_repetition_score'] < 0.8  # Allow phrase detection override
                )
                
                # Flag obvious spam patterns, high phrase repetition, or local repetition
                if is_phrase_spam or is_local_spam or (is_obvious_spam and not seems_legitimate):
                    label = 'REJECT'
                    confidence = max(
                        text_features['repetition_ratio'],
                        text_features['phrase_repetition_score'], 
                        text_features['repeated_ngrams_ratio'],
                        text_features['local_repetition_score']
                    )
                else:
                    label = 'APPROVE'
                    confidence = 1.0 - max(
                        text_features['repetition_ratio'] * 0.5,
                        text_features['phrase_repetition_score'] * 0.5,
                        text_features['repeated_ngrams_ratio'] * 0.5,
                        text_features['local_repetition_score'] * 0.5
                    )
                
                pattern_override = (label == 'REJECT')
                
            else:
                # Normal ML + pattern mode
                label = 'REJECT' if prob > threshold else 'APPROVE'
                
                # Pattern-based override for obvious spam
                word_count = text_features['word_count']
                is_very_repetitive = (
                    (text_features['repetition_ratio'] > 0.8) or  
                    (text_features['phrase_repetition_score'] > 0.7) or 
                    (text_features['repeated_ngrams_ratio'] > 0.6)
                )
                
                has_reasonable_diversity = text_features['word_diversity_ratio'] > 0.3 and word_count > 6
                is_likely_review = text_features['readability_score'] > 20 and word_count > 4
                
                if is_very_repetitive and not (has_reasonable_diversity and is_likely_review):
                    label = 'REJECT'
                    pattern_override = True
                    confidence = max(
                        text_features['repetition_ratio'],
                        text_features['phrase_repetition_score'], 
                        text_features['repeated_ngrams_ratio']
                    )
                else:
                    pattern_override = False
                    confidence = prob if label == 'REJECT' else (1 - prob)
            
            # Determine category based on dominant pattern
            category = 'None'
            if label == 'REJECT':
                if pattern_override:
                    if text_features['repetition_ratio'] > 0.6:
                        category = 'Repetitive_Spam'
                    elif text_features['local_repetition_score'] > 0.4:
                        category = 'Local_Repetition_Spam'
                    elif text_features['phrase_repetition_score'] > 0.5:
                        category = 'Phrase_Pattern_Spam'
                    elif text_features['repeated_ngrams_ratio'] > 0.4:
                        category = 'NGram_Pattern_Spam'
                    else:
                        category = 'Pattern_Spam'
                else:
                    # ML-detected spam
                    if text_features['repetition_ratio'] > 0.4 or text_features['phrase_repetition_score'] > 0.4:
                        category = 'Repetitive_Spam'
                    elif text_features['template_score'] > 0.6:
                        category = 'Template_Spam'
                    else:
                        category = 'ML_Detected_Spam'
            
            # Simple confidence interval calculation
            ci_margin = 0.1 * (1 - confidence)  # Smaller margin for higher confidence
            ci_lower = max(0.0, confidence - ci_margin)
            ci_upper = min(1.0, confidence + ci_margin)
            
            results.append(DetectionResult(
                label=label,
                confidence=float(confidence),
                category=category,
                features={
                    'spam_probability': float(prob),
                    'pattern_features': text_features,
                    'threshold_used': float(threshold)
                },
                confidence_interval=(float(ci_lower), float(ci_upper))
            ))
        
        return results
    
    def get_feature_importance(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get most important features from the unified model."""
        if self.pipeline is None:
            raise ValueError("Model not fitted.")
            
        try:
            classifier = self.pipeline.named_steps['classifier']
            feature_union = self.pipeline.named_steps['features']
            
            # Get feature names
            tfidf_vectorizer = feature_union.transformer_list[0][1]
            tfidf_names = list(tfidf_vectorizer.get_feature_names_out())
            pattern_names = [
                'repetition_ratio', 'word_diversity_ratio', 'repeated_ngrams_ratio',
                'caps_ratio', 'punct_ratio', 'readability_score', 
                'avg_sentence_length', 'word_count', 'template_score', 
                'phrase_repetition_score', 'local_repetition_score'
            ]
            
            all_feature_names = tfidf_names + pattern_names
            
            # Get coefficients
            coef = classifier.coef_[0]
            top_indices = np.argsort(np.abs(coef))[-top_k:][::-1]
            
            return [(all_feature_names[idx], float(coef[idx])) for idx in top_indices]
        except Exception as e:
            print(f"Warning: Could not extract feature importance: {e}")
            return []


def load_training_data(file_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Load training data from CSV file."""
    df = pd.read_csv(file_path)
    
    # Handle different column name variations
    text_col = None
    for col in ['text', 'review', 'content', 'text_clean']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(f"No text column found. Available columns: {list(df.columns)}")
    
    texts = df[text_col].astype(str).tolist()
    
    # Handle labels - check if column exists, if not create default
    if 'gold_label' in df.columns:
        labels = df['gold_label'].tolist()
    elif 'label' in df.columns:
        labels = df['label'].tolist()
    else:
        labels = ['APPROVE'] * len(texts)
    
    # Handle categories - check if column exists, if not create default  
    if 'gold_category' in df.columns:
        categories = df['gold_category'].tolist()
    elif 'category' in df.columns:
        categories = df['category'].tolist()
    else:
        categories = ['None'] * len(texts)
    
    return texts, labels, categories


def evaluate_detector(detector: UnifiedSpamDetector, 
                              texts: List[str], 
                              true_labels: List[str], 
                              true_categories: List[str]) -> Dict:
    """Evaluate the unified detector performance."""
    results = detector.predict(texts)
    
    # Extract predictions
    pred_labels = [r.label for r in results]
    pred_categories = [r.category for r in results]
    
    # Binary classification metrics
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', pos_label='REJECT'
    )
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Confidence analysis
    confidences = [r.confidence for r in results]
    
    evaluation_results = {
        'binary_classification': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'confidence_stats': {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
    }
    
    return evaluation_results