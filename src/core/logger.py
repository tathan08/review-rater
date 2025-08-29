"""
Logging Utilities
Comprehensive logging system for the pipeline
"""

import logging
import os
from datetime import datetime
from typing import Optional

class PipelineLogger:
    """Enhanced logging utility for pipeline operations"""
    
    def __init__(self, name: str = 'SpamDetectionPipeline', log_level=logging.INFO, 
                 log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler (if specified)
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_stage_start(self, stage_name: str):
        """Log the start of a pipeline stage"""
        self.info(f"Starting stage: {stage_name}")
    
    def log_stage_end(self, stage_name: str, duration: float):
        """Log the end of a pipeline stage"""
        self.info(f"Completed stage: {stage_name} (Duration: {duration:.2f}s)")
    
    def log_data_info(self, df, stage_name: str):
        """Log information about dataframe"""
        self.info(f"{stage_name} - Data shape: {df.shape}")
        self.info(f"{stage_name} - Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    def log_performance_metrics(self, metrics: dict):
        """Log performance metrics"""
        self.info("Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")

class LoggerFactory:
    """Factory for creating different types of loggers"""
    
    @staticmethod
    def create_pipeline_logger(config, component_name: str = "pipeline") -> PipelineLogger:
        """Create a logger for pipeline components"""
        log_file = None
        if hasattr(config, 'logs_path') and config.logs_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(config.logs_path, f"{component_name}_{timestamp}.log")
        
        log_level = logging.DEBUG if getattr(config, 'verbose_logging', False) else logging.INFO
        
        return PipelineLogger(
            name=f"SpamDetection.{component_name}",
            log_level=log_level,
            log_file=log_file
        )
    
    @staticmethod
    def create_gpt_logger(config) -> PipelineLogger:
        """Create specialized logger for GPT operations"""
        return LoggerFactory.create_pipeline_logger(config, "GPT")
    
    @staticmethod
    def create_error_logger(config) -> PipelineLogger:
        """Create specialized logger for error handling"""
        return LoggerFactory.create_pipeline_logger(config, "ErrorHandler")