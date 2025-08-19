"""
Advanced Financial Market Analysis Module

This module provides comprehensive financial analysis capabilities including:
- Real-time stock price prediction using LSTM networks
- Technical indicator analysis
- Portfolio optimization with Monte Carlo simulation
- Risk assessment models
- Market sentiment analysis
"""

from .stock_predictor import StockPredictor
from .portfolio_optimizer import PortfolioOptimizer
from .risk_analyzer import RiskAnalyzer
from .technical_indicators import TechnicalIndicators
from .market_sentiment import MarketSentimentAnalyzer

__version__ = "1.0.0"
__author__ = "Senior Data Analyst"
__all__ = [
    "StockPredictor",
    "PortfolioOptimizer", 
    "RiskAnalyzer",
    "TechnicalIndicators",
    "MarketSentimentAnalyzer"
]
