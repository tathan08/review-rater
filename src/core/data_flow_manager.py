"""
Data Flow Management
Handles data flow, validation, and caching between pipeline components
"""

import pandas as pd
import pickle
import os
import hashlib
from typing import Any, Optional, List
from ..core.logger import LoggerFactory

class DataFlowManager:
    """Manages data flow between pipeline components"""
    
    def __init__(self, config):
        self.config = config
        self.logger = LoggerFactory.create_pipeline_logger(config, "DataFlow")
        self.cache = {}
        
        # Create directories if they don't exist
        os.makedirs(config.data_path, exist_ok=True)
        os.makedirs(config.model_cache_path, exist_ok=True)
        os.makedirs(config.results_path, exist_ok=True)
    
    def save_intermediate_results(self, data: Any, filename: str, stage: str) -> bool:
        """Save intermediate results for pipeline recovery"""
        if not self.config.save_intermediate_results:
            return True
            
        try:
            filepath = os.path.join(self.config.data_path, f"{stage}_{filename}")
            
            if isinstance(data, pd.DataFrame):
                data.to_pickle(filepath + ".pkl")
                # Also save as CSV for inspection
                data.to_csv(filepath + ".csv", index=False)
            else:
                with open(filepath + ".pkl", 'wb') as f:
                    pickle.dump(data, f)
            
            self.logger.info(f"Saved {stage} results to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving {stage} results: {str(e)}")
            return False
    
    def load_intermediate_results(self, filename: str, stage: str) -> Optional[Any]:
        """Load intermediate results for pipeline recovery"""
        try:
            filepath = os.path.join(self.config.data_path, f"{stage}_{filename}")
            
            if os.path.exists(filepath + ".pkl"):
                if "dataframe" in filename.lower():
                    return pd.read_pickle(filepath + ".pkl")
                else:
                    with open(filepath + ".pkl", 'rb') as f:
                        return pickle.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Error loading {stage} results: {str(e)}")
            return None
    
    def validate_data(self, df: pd.DataFrame, stage: str, required_cols: List[str] = None) -> bool:
        """Validate data at each pipeline stage"""
        try:
            # Basic validation
            if df is None or df.empty:
                self.logger.error(f"{stage}: DataFrame is empty")
                return False
            
            # Default required columns for review data
            if required_cols is None:
                required_cols = ['review_text']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.error(f"{stage}: Missing columns: {missing_cols}")
                return False
            
            # Check for null values in critical columns
            for col in required_cols:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    self.logger.warning(f"{stage}: Found {null_count} null values in {col}")
            
            # Data type validation
            if 'review_text' in df.columns:
                non_string_count = sum(~df['review_text'].apply(lambda x: isinstance(x, str)))
                if non_string_count > 0:
                    self.logger.warning(f"{stage}: Found {non_string_count} non-string reviews")
            
            self.logger.log_data_info(df, stage)
            self.logger.info(f"{stage}: Data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error in {stage}: {str(e)}")
            return False
    
    def create_data_hash(self, data: Any) -> str:
        """Create hash for data caching"""
        if isinstance(data, pd.DataFrame):
            # Hash based on shape and first few rows
            hash_input = f"{data.shape}_{data.head().to_string()}"
        else:
            hash_input = str(data)
        
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    def cache_data(self, key: str, data: Any):
        """Cache data in memory"""
        self.cache[key] = data
        self.logger.debug(f"Cached data with key: {key}")
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data"""
        return self.cache.get(key)
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_size(self) -> int:
        """Get number of cached items"""
        return len(self.cache)
    
    def validate_file_path(self, file_path: str) -> bool:
        """Validate that file exists and is readable"""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File does not exist: {file_path}")
                return False
            
            if not os.access(file_path, os.R_OK):
                self.logger.error(f"File is not readable: {file_path}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"File validation error: {str(e)}")
            return False
    
    def estimate_memory_usage(self, df: pd.DataFrame) -> dict:
        """Estimate memory usage of dataframe"""
        try:
            memory_usage = df.memory_usage(deep=True)
            total_mb = memory_usage.sum() / 1024**2
            
            return {
                'total_mb': total_mb,
                'per_column_mb': (memory_usage / 1024**2).to_dict(),
                'rows': len(df),
                'columns': len(df.columns)
            }
        except Exception as e:
            self.logger.error(f"Memory estimation error: {str(e)}")
            return {'total_mb': 0}