"""
GPT Pseudo-labeling System
Generates ground truth labels using GPT for policy violation detection
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.logger import LoggerFactory
from ..core.constants import POLICY_CATEGORIES, LABELS
from ..core.utils import create_standard_result, find_text_column, ensure_id_column

# Import prompts from the prompts module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from prompts.policy_prompts import (
    NO_ADS_SYSTEM, 
    IRRELEVANT_SYSTEM, 
    RANT_NO_VISIT_SYSTEM,
    build_prompt,
    FEW_SHOTS_NO_ADS,
    FEW_SHOTS_IRRELEVANT,
    FEW_SHOTS_RANT
)

class GPTPseudoLabeler:
    """GPT-based pseudo-labeling system for generating ground truth"""
    
    def __init__(self, config):
        self.config = config
        self.logger = LoggerFactory.create_gpt_logger(config)
        self.client = None
        self.total_calls = 0  # Simple call counter
        
        # Initialize OpenAI client if available and API key provided
        if OPENAI_AVAILABLE and hasattr(config, 'openai_api_key') and config.openai_api_key:
            self.client = OpenAI(api_key=config.openai_api_key)
            self.logger.info("OpenAI client initialized")
        else:
            self.logger.warning("OpenAI client not available or API key not provided")
        
        # Use the consolidated prompt systems from policy_prompts.py
        self.policy_systems = {
            POLICY_CATEGORIES['NO_ADS']: (NO_ADS_SYSTEM, FEW_SHOTS_NO_ADS),
            POLICY_CATEGORIES['IRRELEVANT']: (IRRELEVANT_SYSTEM, FEW_SHOTS_IRRELEVANT), 
            POLICY_CATEGORIES['RANT_NO_VISIT']: (RANT_NO_VISIT_SYSTEM, FEW_SHOTS_RANT)
        }
    
    def _call_gpt(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Make API call to GPT with simple retry logic"""
        if not self.client:
            self.logger.error("OpenAI client not initialized")
            return None
        
        for attempt in range(max_retries):
            try:
                model = getattr(self.config, 'gpt_model', 'gpt-3.5-turbo')
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes Google reviews for policy violations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.1  # Low temperature for consistent results
                )
                
                # Simple call tracking
                self.total_calls += 1
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                self.logger.warning(f"GPT API call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        return None
    
    def label_single_review(self, review_text: str, business_name: str = "Unknown Business") -> Dict[str, Any]:
        """Label a single review for all policy violations using the consolidated approach"""
        results = {}
        
        for policy_category, (system_prompt, few_shots) in self.policy_systems.items():
            # Build the complete prompt using the consolidated function
            prompt = build_prompt(system_prompt, review_text[:1000], few_shots)
            
            response = self._call_gpt(prompt)
            
            if response:
                try:
                    # Parse JSON response
                    import json
                    response_data = json.loads(response.strip())
                    
                    results[policy_category] = {
                        'violation': response_data.get('label') == LABELS['REJECT'],
                        'confidence': float(response_data.get('confidence', 0.0)),
                        'explanation': response_data.get('rationale', ''),
                        'category': response_data.get('category', POLICY_CATEGORIES['NONE'])
                    }
                        
                except Exception as e:
                    self.logger.warning(f"Error parsing GPT response for {policy_category}: {str(e)}")
                    # Use default values on parsing error
                    results[policy_category] = {
                        'violation': False, 
                        'confidence': 0.0, 
                        'explanation': 'Parse error',
                        'category': POLICY_CATEGORIES['NONE']
                    }
            else:
                # Default values when API call fails
                results[policy_category] = {
                    'violation': False, 
                    'confidence': 0.0, 
                    'explanation': 'API call failed',
                    'category': POLICY_CATEGORIES['NONE']
                }
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def _create_diverse_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Create a diverse sample of reviews for labeling"""
        if len(df) <= sample_size:
            return df
        
        # Try to get a diverse sample by text length and other factors
        df_copy = df.copy()
        text_col = find_text_column(df_copy)
        df_copy['text_length'] = df_copy[text_col].str.len()
        
        # Sample across different text lengths
        try:
            # Create quartiles based on text length
            df_copy['length_quartile'] = pd.qcut(df_copy['text_length'], q=4, labels=False, duplicates='drop')
            sampled = df_copy.groupby('length_quartile').apply(
                lambda x: x.sample(min(len(x), sample_size // 4), random_state=42)
            ).reset_index(drop=True)
            
            # If we don't have enough, sample more randomly
            if len(sampled) < sample_size:
                remaining = sample_size - len(sampled)
                excluded = df_copy[~df_copy.index.isin(sampled.index)]
                if len(excluded) > 0:
                    additional = excluded.sample(min(remaining, len(excluded)), random_state=42)
                    sampled = pd.concat([sampled, additional]).reset_index(drop=True)
            
            return sampled.drop(['text_length', 'length_quartile'], axis=1)
        except:
            # Fallback to random sampling
            return df_copy.sample(min(sample_size, len(df_copy)), random_state=42).drop('text_length', axis=1)
    
    def generate_pseudo_labels(self, df: pd.DataFrame, sample_size: int = 500, 
                             save_progress: bool = True) -> pd.DataFrame:
        """Generate pseudo labels for a sample of reviews"""
        self.logger.info(f"Starting pseudo-labeling for {min(sample_size, len(df))} reviews")
        
        if not self.client:
            self.logger.error("Cannot generate pseudo-labels: OpenAI client not initialized")
            return pd.DataFrame()
        
        # Ensure proper column structure
        df = ensure_id_column(df)
        text_col = find_text_column(df)
        
        # Sample reviews strategically
        if len(df) > sample_size:
            sampled_df = self._create_diverse_sample(df, sample_size)
        else:
            sampled_df = df.copy()
        
        results = []
        
        # Save progress every N reviews
        progress_interval = 25
        
        for idx, (orig_idx, row) in enumerate(tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Pseudo-labeling")):
            review_text = str(row[text_col])
            business_name = row.get('business_name', row.get('name', 'Unknown Business'))
            
            # Get pseudo labels for this review
            policy_results = self.label_single_review(review_text, business_name)
            
            # Determine overall label and category
            violations = [policy for policy, result in policy_results.items() if result['violation']]
            
            if violations:
                # Choose the violation with highest confidence
                best_policy = max(violations, key=lambda p: policy_results[p]['confidence'])
                final_label = LABELS['REJECT']
                final_category = best_policy
                confidence = policy_results[best_policy]['confidence']
                explanation = policy_results[best_policy]['explanation']
            else:
                final_label = LABELS['APPROVE']
                final_category = POLICY_CATEGORIES['NONE']
                confidence = 0.8  # Default confidence for approval
                explanation = "No policy violations detected"
            
            result = create_standard_result(
                review_id=row['id'],
                text=review_text,
                label=final_label,
                category=final_category,
                confidence=confidence,
                rationale=explanation
            )
            
            # Add detailed policy results
            result['policy_details'] = policy_results
            results.append(result)
            
            # Save progress periodically
            if save_progress and (idx + 1) % progress_interval == 0:
                temp_df = pd.DataFrame(results)
                temp_file = f"pseudo_labels_progress_{idx+1}.csv"
                temp_df.to_csv(temp_file, index=False)
                self.logger.info(f"Progress saved: {temp_file}")
        
        results_df = pd.DataFrame(results)
        self.logger.info(f"Pseudo-labeling completed. Generated {len(results_df)} labels")
        self.logger.info(f"Total API calls made: {self.total_calls}")
        
        return results_df