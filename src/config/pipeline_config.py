"""
Pipeline Configuration Classes

This module contains configuration classes for various pipeline components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class ModelConfig:
    """Configuration for model settings"""
    ollama_model: str = "mistral:7b-instruct"
    hf_sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    hf_toxicity_model: str = "unitary/toxic-bert"
    hf_zero_shot_model: str = "facebook/bart-large-mnli"
    
    # Confidence thresholds
    sentiment_threshold: float = 0.7
    toxicity_threshold: float = 0.5
    zero_shot_threshold: float = 0.7
    ensemble_tau: float = 0.55


@dataclass
class DataConfig:
    """Configuration for data paths and settings"""
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/clean"  # Renamed from processed to clean
    sample_data_dir: str = "data/sample"
    pseudo_label_dir: str = "data/pseudo-label"  # New directory for pseudo-labels
    training_dir: str = "data/training"  # New directory for training data
    testing_dir: str = "data/testing"  # New directory for testing data
    
    # Default input file
    sample_reviews_file: str = "data/sample/sample_reviews.csv"
    
    
@dataclass
class OutputConfig:
    """Configuration for output paths"""
    results_dir: str = "results"
    predictions_dir: str = "results/predictions"
    evaluations_dir: str = "results/evaluations"
    reports_dir: str = "results/reports"
    
    # Default output files
    prompt_predictions: str = "results/predictions/predictions.csv"
    hf_predictions: str = "results/predictions/predictions_hf.csv"
    ensemble_predictions: str = "results/predictions/predictions_ens.csv"


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    logs_dir: str = "logs"
    pipeline_logs_dir: str = "logs/pipeline_logs"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all components"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Legacy compatibility
    batch_size: int = 32
    max_workers: int = 4
    cache_predictions: bool = True
    
    # Gemini configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    
    # Advanced settings
    enable_performance_monitoring: bool = True
    enable_error_recovery: bool = True
    save_intermediate_results: bool = True
    verbose_logging: bool = True
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        directories = [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.sample_data_dir,
            self.data.pseudo_label_dir,
            self.data.training_dir,
            self.data.testing_dir,
            self.output.predictions_dir,
            self.output.evaluations_dir,
            self.output.reports_dir,
            self.logging.pipeline_logs_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def create_demo_config(cls, gemini_api_key: str = "") -> 'PipelineConfig':
        """Create configuration optimized for demo/testing"""
        return cls(
            batch_size=16,
            gemini_api_key=gemini_api_key,
            cache_predictions=True,
            verbose_logging=True
        )
    
    @classmethod
    def create_production_config(cls, gemini_api_key: str = "") -> 'PipelineConfig':
        """Create configuration optimized for production use"""
        return cls(
            batch_size=64,
            gemini_api_key=gemini_api_key,
            cache_predictions=True,
            verbose_logging=False
        )
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        return True


# Global configuration instance
config = PipelineConfig()