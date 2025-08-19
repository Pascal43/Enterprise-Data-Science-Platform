"""
Advanced Computer Vision Module

This module provides comprehensive computer vision capabilities including:
- Document OCR with Tesseract
- Invoice data extraction
- Image classification
- Object detection
- Image preprocessing and enhancement
"""

from .document_processor import DocumentProcessor
from .image_classifier import ImageClassifier
from .object_detector import ObjectDetector
from .image_preprocessor import ImagePreprocessor
from .ocr_engine import OCREngine

__version__ = "1.0.0"
__author__ = "Senior Data Analyst"
__all__ = [
    "DocumentProcessor",
    "ImageClassifier",
    "ObjectDetector",
    "ImagePreprocessor",
    "OCREngine"
]
