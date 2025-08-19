"""
Unit Tests for Financial Analysis Module

This module contains comprehensive tests for the financial analysis functionality
including stock prediction, technical indicators, and data processing.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_analysis.stock_predictor import StockPredictor

class TestStockPredictor(unittest.TestCase):
    """Test cases for StockPredictor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = StockPredictor(sequence_length=60, prediction_days=30)
        
        # Create sample stock data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(50, 150, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Add some trend to make data more realistic
        self.sample_data['Close'] = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
        self.sample_data['Open'] = self.sample_data['Close'] + np.random.normal(0, 2, 100)
        self.sample_data['High'] = self.sample_data['Close'] + np.random.uniform(0, 10, 100)
        self.sample_data['Low'] = self.sample_data['Close'] - np.random.uniform(0, 10, 100)
    
    def test_initialization(self):
        """Test StockPredictor initialization"""
        self.assertEqual(self.predictor.sequence_length, 60)
        self.assertEqual(self.predictor.prediction_days, 30)
        self.assertIsNone(self.predictor.model)
        self.assertFalse(self.predictor.is_trained)
    
    def test_add_technical_indicators(self):
        """Test technical indicators calculation"""
        data_with_indicators = self.predictor._add_technical_indicators(self.sample_data.copy())
        
        # Check that indicators were added
        expected_indicators = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 
                             'RSI', 'Stoch', 'Williams_R', 'BB_upper', 'BB_lower']
        
        for indicator in expected_indicators:
            self.assertIn(indicator, data_with_indicators.columns)
        
        # Check that data is not empty after processing
        self.assertGreater(len(data_with_indicators), 0)
        
        # Check that NaN values were removed
        self.assertFalse(data_with_indicators.isnull().any().any())
    
    def test_add_sentiment_features(self):
        """Test sentiment features calculation"""
        data_with_sentiment = self.predictor._add_sentiment_features(self.sample_data.copy())
        
        # Check that sentiment features were added
        expected_features = ['Volatility', 'Volatility_Ratio', 'ADX', 
                           'Support', 'Resistance', 'Price_Position']
        
        for feature in expected_features:
            self.assertIn(feature, data_with_sentiment.columns)
    
    def test_prepare_data(self):
        """Test data preparation for LSTM"""
        # Add technical indicators first
        data_with_indicators = self.predictor._add_technical_indicators(self.sample_data.copy())
        
        # Prepare data
        X_train, y_train, X_test, y_test = self.predictor.prepare_data(data_with_indicators)
        
        # Check shapes
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        
        # Check that X_train has correct shape
        self.assertEqual(X_train.shape[1], self.predictor.sequence_length)
        self.assertGreater(X_train.shape[2], 0)  # Number of features
    
    def test_build_model(self):
        """Test LSTM model building"""
        input_shape = (60, 25)  # sequence_length, num_features
        model = self.predictor.build_model(input_shape)
        
        # Check that model was created
        self.assertIsNotNone(model)
        
        # Check model summary
        self.assertIsNotNone(model.summary)
        
        # Check that model has expected layers
        layer_names = [layer.name for layer in model.layers]
        self.assertIn('bidirectional', layer_names[0])
        self.assertIn('dense', layer_names[-1])
    
    @patch('financial_analysis.stock_predictor.yf.Ticker')
    def test_fetch_stock_data(self, mock_ticker):
        """Test stock data fetching"""
        # Mock the yfinance response
        mock_stock = Mock()
        mock_stock.history.return_value = self.sample_data
        mock_ticker.return_value = mock_stock
        
        # Test data fetching
        result = self.predictor.fetch_stock_data('AAPL')
        
        # Check that data was fetched and processed
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        
        # Check that technical indicators were added
        self.assertIn('SMA_20', result.columns)
        self.assertIn('RSI', result.columns)
    
    @patch('financial_analysis.stock_predictor.yf.Ticker')
    def test_fetch_stock_data_empty(self, mock_ticker):
        """Test stock data fetching with empty data"""
        # Mock empty response
        mock_stock = Mock()
        mock_stock.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_stock
        
        # Test data fetching
        result = self.predictor.fetch_stock_data('INVALID')
        
        # Should return None for invalid symbol
        self.assertIsNone(result)
    
    def test_get_prediction_confidence(self):
        """Test prediction confidence calculation"""
        predictions = np.array([100, 101, 102, 103, 104])
        lower_bound, upper_bound = self.predictor.get_prediction_confidence(predictions)
        
        # Check that confidence intervals were calculated
        self.assertEqual(len(lower_bound), len(predictions))
        self.assertEqual(len(upper_bound), len(predictions))
        
        # Check that lower bound is less than upper bound
        self.assertTrue(np.all(lower_bound < upper_bound))
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        # Create a simple model first
        input_shape = (60, 25)
        model = self.predictor.build_model(input_shape)
        self.predictor.model = model
        self.predictor.is_trained = True
        
        # Test saving
        with patch('builtins.print') as mock_print:
            self.predictor.save_model("test_model.h5")
            mock_print.assert_called_with("Model saved to test_model.h5")
        
        # Test loading
        with patch('builtins.print') as mock_print:
            self.predictor.load_model("test_model.h5")
            mock_print.assert_called_with("Model loaded from test_model.h5")
            self.assertTrue(self.predictor.is_trained)
    
    def test_predict_future_not_trained(self):
        """Test prediction without trained model"""
        with self.assertRaises(ValueError):
            self.predictor.predict_future()
    
    def test_invalid_sequence_length(self):
        """Test initialization with invalid parameters"""
        with self.assertRaises(ValueError):
            StockPredictor(sequence_length=0, prediction_days=30)
        
        with self.assertRaises(ValueError):
            StockPredictor(sequence_length=60, prediction_days=0)

class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for technical indicators"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = StockPredictor()
        
        # Create sample price data
        self.prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        # Create data with known trend
        data = pd.DataFrame({
            'Close': self.prices,
            'High': self.prices + 2,
            'Low': self.prices - 2,
            'Volume': np.random.randint(1000000, 10000000, len(self.prices))
        })
        
        result = self.predictor._add_technical_indicators(data)
        
        # Check that RSI was calculated
        self.assertIn('RSI', result.columns)
        
        # RSI should be between 0 and 100
        rsi_values = result['RSI'].dropna()
        self.assertTrue(np.all((rsi_values >= 0) & (rsi_values <= 100)))
    
    def test_moving_averages(self):
        """Test moving average calculations"""
        data = pd.DataFrame({
            'Close': self.prices,
            'High': self.prices + 2,
            'Low': self.prices - 2,
            'Volume': np.random.randint(1000000, 10000000, len(self.prices))
        })
        
        result = self.predictor._add_technical_indicators(data)
        
        # Check that moving averages were calculated
        self.assertIn('SMA_20', result.columns)
        self.assertIn('SMA_50', result.columns)
        self.assertIn('EMA_12', result.columns)
        self.assertIn('EMA_26', result.columns)
        
        # Check that values are reasonable
        sma_20 = result['SMA_20'].dropna()
        self.assertTrue(len(sma_20) > 0)
        self.assertTrue(np.all(sma_20 > 0))

class TestDataValidation(unittest.TestCase):
    """Test cases for data validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = StockPredictor()
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            self.predictor.prepare_data(empty_df)
    
    def test_missing_columns(self):
        """Test handling of missing required columns"""
        incomplete_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107]
            # Missing 'Low', 'Close', 'Volume'
        })
        
        with self.assertRaises(KeyError):
            self.predictor.prepare_data(incomplete_df)
    
    def test_non_numeric_data(self):
        """Test handling of non-numeric data"""
        invalid_df = pd.DataFrame({
            'Open': ['a', 'b', 'c'],
            'High': ['d', 'e', 'f'],
            'Low': ['g', 'h', 'i'],
            'Close': ['j', 'k', 'l'],
            'Volume': ['m', 'n', 'o']
        })
        
        with self.assertRaises((ValueError, TypeError)):
            self.predictor.prepare_data(invalid_df)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
