"""
Advanced NLP Sentiment Analysis Module

This module provides comprehensive natural language processing capabilities including:
- BERT-based sentiment analysis
- Named Entity Recognition (NER)
- Text summarization
- Topic modeling with LDA
- Multi-language support
"""

from .sentiment_analyzer import SentimentAnalyzer
from .entity_extractor import EntityExtractor
from .text_summarizer import TextSummarizer
from .topic_modeler import TopicModeler
from .multi_language_processor import MultiLanguageProcessor

__version__ = "1.0.0"
__author__ = "Senior Data Analyst"
__all__ = [
    "SentimentAnalyzer",
    "EntityExtractor",
    "TextSummarizer", 
    "TopicModeler",
    "MultiLanguageProcessor"
]
