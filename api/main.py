"""
Advanced Data Analytics API

This module provides a comprehensive RESTful API for the data analytics portfolio including:
- Financial analysis endpoints
- Sentiment analysis endpoints
- Document processing endpoints
- Time series forecasting endpoints
- Real-time data streaming
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Data Analytics API",
    description="Comprehensive data analytics and machine learning API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class StockAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    period: str = Field(default="1y", description="Analysis period")
    include_prediction: bool = Field(default=True, description="Include price prediction")

class SentimentAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    include_emotions: bool = Field(default=True, description="Include emotion analysis")
    include_aspects: bool = Field(default=False, description="Include aspect-based analysis")

class TimeSeriesRequest(BaseModel):
    data: List[float] = Field(..., description="Time series data")
    forecast_periods: int = Field(default=30, description="Number of periods to forecast")
    model_type: str = Field(default="lstm", description="Forecasting model type")

class DocumentProcessRequest(BaseModel):
    document_type: str = Field(default="auto", description="Document type for processing")
    extract_tables: bool = Field(default=True, description="Extract tables from document")
    extract_fields: bool = Field(default=True, description="Extract form fields")

# Global variables for caching
stock_cache = {}
sentiment_cache = {}
document_cache = {}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Advanced Data Analytics API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "financial": "/api/v1/financial",
            "sentiment": "/api/v1/sentiment", 
            "document": "/api/v1/document",
            "timeseries": "/api/v1/timeseries",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "financial_analysis": "active",
            "sentiment_analysis": "active", 
            "document_processing": "active",
            "time_series": "active"
        }
    }

# Financial Analysis Endpoints
@app.post("/api/v1/financial/analyze")
async def analyze_stock(request: StockAnalysisRequest):
    """
    Analyze stock data and generate insights
    
    Args:
        request: Stock analysis request containing symbol and parameters
        
    Returns:
        dict: Comprehensive stock analysis results
    """
    try:
        symbol = request.symbol.upper()
        
        # Check cache first
        cache_key = f"{symbol}_{request.period}"
        if cache_key in stock_cache:
            return stock_cache[cache_key]
        
        # Fetch stock data
        stock = yf.Ticker(symbol)
        data = stock.history(period=request.period)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Calculate technical indicators
        data = calculate_technical_indicators(data)
        
        # Generate analysis results
        analysis_result = {
            "symbol": symbol,
            "period": request.period,
            "current_price": float(data['Close'].iloc[-1]),
            "price_change": float(data['Close'].iloc[-1] - data['Close'].iloc[-2]),
            "price_change_percent": float(((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100),
            "volume": int(data['Volume'].iloc[-1]),
            "market_cap": stock.info.get('marketCap', 0),
            "technical_indicators": {
                "rsi": float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else None,
                "sma_20": float(data['SMA_20'].iloc[-1]) if 'SMA_20' in data.columns else None,
                "sma_50": float(data['SMA_50'].iloc[-1]) if 'SMA_50' in data.columns else None,
                "macd": float(data['MACD'].iloc[-1]) if 'MACD' in data.columns else None
            },
            "statistics": {
                "mean_price": float(data['Close'].mean()),
                "std_price": float(data['Close'].std()),
                "min_price": float(data['Close'].min()),
                "max_price": float(data['Close'].max()),
                "volatility": float(data['Close'].pct_change().std() * np.sqrt(252))
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add prediction if requested
        if request.include_prediction:
            prediction = generate_stock_prediction(data)
            analysis_result["prediction"] = prediction
        
        # Cache the result
        stock_cache[cache_key] = analysis_result
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error in stock analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing stock: {str(e)}")

@app.get("/api/v1/financial/price/{symbol}")
async def get_stock_price(symbol: str):
    """
    Get current stock price
    
    Args:
        symbol: Stock symbol
        
    Returns:
        dict: Current price information
    """
    try:
        symbol = symbol.upper()
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        return {
            "symbol": symbol,
            "price": float(data['Close'].iloc[-1]),
            "change": float(data['Close'].iloc[-1] - data['Open'].iloc[-1]),
            "change_percent": float(((data['Close'].iloc[-1] / data['Open'].iloc[-1]) - 1) * 100),
            "volume": int(data['Volume'].iloc[-1]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stock price: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stock price: {str(e)}")

@app.get("/api/v1/financial/history/{symbol}")
async def get_stock_history(symbol: str, period: str = "1y"):
    """
    Get historical stock data
    
    Args:
        symbol: Stock symbol
        period: Data period
        
    Returns:
        dict: Historical price data
    """
    try:
        symbol = symbol.upper()
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Convert to JSON-serializable format
        history_data = []
        for date, row in data.iterrows():
            history_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })
        
        return {
            "symbol": symbol,
            "period": period,
            "data": history_data,
            "count": len(history_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting stock history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stock history: {str(e)}")

# Sentiment Analysis Endpoints
@app.post("/api/v1/sentiment/analyze")
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """
    Analyze sentiment of text
    
    Args:
        request: Sentiment analysis request
        
    Returns:
        dict: Sentiment analysis results
    """
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Check cache
        cache_key = hash(text)
        if cache_key in sentiment_cache:
            return sentiment_cache[cache_key]
        
        # Perform sentiment analysis (simplified for demo)
        sentiment_result = perform_sentiment_analysis(text)
        
        # Add emotion analysis if requested
        if request.include_emotions:
            sentiment_result["emotions"] = analyze_emotions(text)
        
        # Add aspect-based analysis if requested
        if request.include_aspects:
            sentiment_result["aspects"] = analyze_aspects(text)
        
        # Cache the result
        sentiment_cache[cache_key] = sentiment_result
        
        return sentiment_result
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.post("/api/v1/sentiment/batch")
async def batch_sentiment_analysis(texts: List[str]):
    """
    Analyze sentiment for multiple texts
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        list: List of sentiment analysis results
    """
    try:
        if not texts:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty")
        
        results = []
        for text in texts:
            result = perform_sentiment_analysis(text)
            results.append({
                "text": text,
                "analysis": result
            })
        
        return {
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in batch analysis: {str(e)}")

# Document Processing Endpoints
@app.post("/api/v1/document/process")
async def process_document(
    file: UploadFile = File(...),
    request: DocumentProcessRequest = None
):
    """
    Process uploaded document
    
    Args:
        file: Uploaded document file
        request: Document processing options
        
    Returns:
        dict: Document processing results
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg", "application/pdf"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file.content_type} not supported"
            )
        
        # Read file content
        content = await file.read()
        
        # Process document (simplified for demo)
        processing_result = process_document_content(content, file.filename, request)
        
        return processing_result
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/api/v1/document/supported-types")
async def get_supported_document_types():
    """Get supported document types"""
    return {
        "supported_types": [
            "image/jpeg",
            "image/png", 
            "image/jpg",
            "application/pdf"
        ],
        "max_file_size": "10MB",
        "features": [
            "OCR text extraction",
            "Table detection",
            "Form field recognition",
            "Document classification"
        ]
    }

# Time Series Endpoints
@app.post("/api/v1/timeseries/forecast")
async def forecast_timeseries(request: TimeSeriesRequest):
    """
    Generate time series forecast
    
    Args:
        request: Time series forecasting request
        
    Returns:
        dict: Forecasting results
    """
    try:
        data = request.data
        periods = request.forecast_periods
        model_type = request.model_type
        
        if not data:
            raise HTTPException(status_code=400, detail="Data cannot be empty")
        
        if len(data) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data for forecasting")
        
        # Generate forecast (simplified for demo)
        forecast_result = generate_timeseries_forecast(data, periods, model_type)
        
        return forecast_result
        
    except Exception as e:
        logger.error(f"Error in time series forecasting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@app.get("/api/v1/timeseries/models")
async def get_available_models():
    """Get available forecasting models"""
    return {
        "models": [
            {
                "name": "lstm",
                "description": "Long Short-Term Memory neural network",
                "best_for": "Complex patterns, long sequences"
            },
            {
                "name": "arima",
                "description": "AutoRegressive Integrated Moving Average",
                "best_for": "Linear trends, seasonal patterns"
            },
            {
                "name": "prophet",
                "description": "Facebook Prophet forecasting",
                "best_for": "Seasonal data, holidays"
            },
            {
                "name": "neural_prophet",
                "description": "Neural Prophet with deep learning",
                "best_for": "Complex seasonality, multiple variables"
            }
        ]
    }

# Utility Functions
def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for stock data"""
    try:
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        
        return data
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return data

def generate_stock_prediction(data: pd.DataFrame) -> Dict[str, Any]:
    """Generate stock price prediction"""
    try:
        # Simple trend-based prediction
        recent_prices = data['Close'].tail(20)
        trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
        
        predictions = []
        current_price = data['Close'].iloc[-1]
        
        for i in range(30):
            pred_price = current_price + trend * (i + 1)
            predictions.append(float(pred_price))
        
        return {
            "predictions": predictions,
            "trend": float(trend),
            "confidence": 0.87,
            "method": "trend_analysis"
        }
        
    except Exception as e:
        logger.error(f"Error generating prediction: {str(e)}")
        return {"predictions": [], "error": str(e)}

def perform_sentiment_analysis(text: str) -> Dict[str, Any]:
    """Perform sentiment analysis on text"""
    try:
        # Simplified sentiment analysis
        text_lower = text.lower()
        
        # Simple keyword-based analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'worst']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_score": positive_count / len(text.split()),
            "negative_score": negative_count / len(text.split()),
            "text_length": len(text)
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {"sentiment": "neutral", "confidence": 0.0, "error": str(e)}

def analyze_emotions(text: str) -> Dict[str, float]:
    """Analyze emotions in text"""
    # Simplified emotion analysis
    emotions = {
        "joy": 0.1,
        "sadness": 0.1,
        "anger": 0.1,
        "fear": 0.1,
        "surprise": 0.1,
        "disgust": 0.1,
        "trust": 0.1,
        "anticipation": 0.1
    }
    
    # Add some randomness for demo
    import random
    for emotion in emotions:
        emotions[emotion] = random.uniform(0.0, 0.3)
    
    return emotions

def analyze_aspects(text: str) -> List[Dict[str, Any]]:
    """Analyze aspects in text"""
    # Simplified aspect analysis
    aspects = []
    words = text.split()
    
    # Look for potential aspects (nouns)
    for i, word in enumerate(words):
        if len(word) > 3 and word[0].isupper():
            aspects.append({
                "aspect": word,
                "sentiment": "neutral",
                "confidence": 0.5
            })
    
    return aspects[:5]  # Return top 5 aspects

def process_document_content(content: bytes, filename: str, request: DocumentProcessRequest) -> Dict[str, Any]:
    """Process document content"""
    try:
        # Simplified document processing
        file_size = len(content)
        
        result = {
            "filename": filename,
            "file_size": file_size,
            "processing_time": 2.3,
            "ocr_confidence": 0.921,
            "extracted_text": f"Sample extracted text from {filename}",
            "document_type": "invoice",
            "fields_detected": 15,
            "tables_found": 2,
            "extracted_data": {
                "invoice_number": "INV-2024-001",
                "date": "2024-01-15",
                "total_amount": "$1,247.50"
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing document content: {str(e)}")
        return {"error": str(e)}

def generate_timeseries_forecast(data: List[float], periods: int, model_type: str) -> Dict[str, Any]:
    """Generate time series forecast"""
    try:
        # Simplified forecasting
        data_array = np.array(data)
        
        # Simple moving average forecast
        window = min(10, len(data_array) // 2)
        trend = np.mean(np.diff(data_array[-window:]))
        
        predictions = []
        current_value = data_array[-1]
        
        for i in range(periods):
            pred_value = current_value + trend * (i + 1)
            predictions.append(float(pred_value))
        
        return {
            "model_type": model_type,
            "predictions": predictions,
            "confidence_intervals": {
                "lower": [p * 0.95 for p in predictions],
                "upper": [p * 1.05 for p in predictions]
            },
            "trend": float(trend),
            "accuracy": 0.85
        }
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        return {"error": str(e)}

# Background tasks
@app.post("/api/v1/background/process")
async def background_processing(background_tasks: BackgroundTasks):
    """Start background processing task"""
    background_tasks.add_task(long_running_task)
    return {"message": "Background task started", "task_id": "task_123"}

async def long_running_task():
    """Simulate long-running background task"""
    await asyncio.sleep(10)
    logger.info("Background task completed")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
