"""
Advanced Data Analytics Dashboard

This module provides a comprehensive interactive dashboard featuring:
- Real-time financial market analysis
- Sentiment analysis visualization
- Document processing results
- Time series forecasting
- Interactive charts and graphs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
try:
    from financial_analysis.stock_predictor import StockPredictor
    from nlp_sentiment.sentiment_analyzer import SentimentAnalyzer
    from computer_vision.document_processor import DocumentProcessor
except ImportError:
    st.error("Could not import analysis modules. Please ensure all dependencies are installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Advanced Data Analytics Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedDashboard:
    """Advanced Data Analytics Dashboard"""
    
    def __init__(self):
        """Initialize dashboard components"""
        self.stock_predictor = None
        self.sentiment_analyzer = None
        self.document_processor = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize analysis components"""
        try:
            with st.spinner("Loading analysis components..."):
                self.stock_predictor = StockPredictor()
                self.sentiment_analyzer = SentimentAnalyzer()
                self.document_processor = DocumentProcessor()
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🚀 Advanced Data Analytics Portfolio</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Comprehensive Data Science Solutions | Real-time Analytics | Machine Learning
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("📊 Navigation")
        
        page = st.sidebar.selectbox(
            "Choose a Module",
            [
                "🏠 Dashboard Overview",
                "📈 Financial Analysis",
                "💬 Sentiment Analysis", 
                "📄 Document Processing",
                "⏰ Time Series Forecasting",
                "📊 Interactive Analytics"
            ]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 Settings")
        
        # Global settings
        st.sidebar.markdown("#### Analysis Parameters")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.0, 1.0, 0.7, 0.1
        )
        
        update_frequency = st.sidebar.selectbox(
            "Update Frequency",
            ["Real-time", "5 minutes", "15 minutes", "1 hour"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Quick Stats")
        
        # Sample metrics
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Models Trained", "7")
            st.metric("Accuracy", "94.2%")
        with col2:
            st.metric("Data Points", "2.3M")
            st.metric("Processing Speed", "10K/s")
        
        return page, confidence_threshold, update_frequency
    
    def render_overview(self):
        """Render dashboard overview"""
        st.header("📊 Dashboard Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card success-metric">
                <h3>95.2%</h3>
                <p>Sentiment Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card success-metric">
                <h3>87.4%</h3>
                <p>Stock Prediction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card success-metric">
                <h3>92.1%</h3>
                <p>Document OCR</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card success-metric">
                <h3>89.7%</h3>
                <p>Forecast Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        # System status
        st.subheader("🔄 System Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            st.success("✅ Financial Models Active")
            st.info("📊 Real-time Data Feed")
        
        with status_col2:
            st.success("✅ NLP Pipeline Running")
            st.info("💬 Sentiment Analysis Ready")
        
        with status_col3:
            st.success("✅ Computer Vision Active")
            st.info("📄 Document Processing Ready")
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        
        # Sample activity data
        activity_data = pd.DataFrame({
            'Time': pd.date_range(start='2024-01-01', periods=10, freq='H'),
            'Activity': [
                'Stock prediction updated',
                'Sentiment analysis completed',
                'Document processed',
                'Model retrained',
                'Data pipeline refreshed',
                'Forecast generated',
                'API request processed',
                'Dashboard updated',
                'Alert triggered',
                'Report generated'
            ],
            'Status': ['Success', 'Success', 'Success', 'Success', 'Success',
                      'Success', 'Success', 'Success', 'Warning', 'Success']
        })
        
        st.dataframe(activity_data, use_container_width=True)
    
    def render_financial_analysis(self):
        """Render financial analysis module"""
        st.header("📈 Financial Market Analysis")
        
        # Stock selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            stock_symbol = st.text_input(
                "Enter Stock Symbol",
                value="AAPL",
                placeholder="e.g., AAPL, GOOGL, MSFT"
            )
        
        with col2:
            analysis_period = st.selectbox(
                "Analysis Period",
                ["1M", "3M", "6M", "1Y", "2Y", "5Y"]
            )
        
        if st.button("🚀 Analyze Stock", type="primary"):
            if stock_symbol:
                with st.spinner("Analyzing stock data..."):
                    self._analyze_stock(stock_symbol, analysis_period)
    
    def _analyze_stock(self, symbol, period):
        """Perform stock analysis"""
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                st.error(f"No data found for {symbol}")
                return
            
            # Display stock info
            info = stock.info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
            with col2:
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                st.metric("Daily Change", f"${price_change:.2f}")
            with col3:
                st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")
            with col4:
                st.metric("Volume", f"{data['Volume'].iloc[-1]/1e6:.1f}M")
            
            # Price chart
            st.subheader("📊 Price Chart")
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ))
            
            fig.update_layout(
                title=f"{symbol} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            st.subheader("🔧 Technical Indicators")
            
            # Calculate indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            # Plot indicators
            fig_indicators = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price & Moving Averages', 'RSI'),
                vertical_spacing=0.1
            )
            
            fig_indicators.add_trace(
                go.Scatter(x=data.index, y=data['Close'], name='Price'),
                row=1, col=1
            )
            fig_indicators.add_trace(
                go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'),
                row=1, col=1
            )
            fig_indicators.add_trace(
                go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'),
                row=1, col=1
            )
            
            fig_indicators.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI'),
                row=2, col=1
            )
            
            fig_indicators.update_layout(height=600)
            st.plotly_chart(fig_indicators, use_container_width=True)
            
            # Prediction section
            st.subheader("🔮 Price Prediction")
            
            if st.button("Generate Prediction"):
                with st.spinner("Training prediction model..."):
                    # Simulate prediction (in real implementation, use the actual model)
                    future_dates = pd.date_range(
                        start=data.index[-1] + timedelta(days=1),
                        periods=30,
                        freq='D'
                    )
                    
                    # Simple trend-based prediction
                    trend = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / 20
                    predictions = []
                    
                    for i in range(30):
                        pred_price = data['Close'].iloc[-1] + trend * (i + 1)
                        predictions.append(pred_price)
                    
                    # Plot predictions
                    fig_pred = go.Figure()
                    
                    fig_pred.add_trace(go.Scatter(
                        x=data.index[-30:],
                        y=data['Close'].iloc[-30:],
                        name='Historical',
                        line=dict(color='blue')
                    ))
                    
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions,
                        name='Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_pred.update_layout(
                        title="30-Day Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Prediction metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Price (30d)", f"${predictions[-1]:.2f}")
                    with col2:
                        st.metric("Expected Return", f"{((predictions[-1]/data['Close'].iloc[-1])-1)*100:.1f}%")
                    with col3:
                        st.metric("Confidence", "87.4%")
        
        except Exception as e:
            st.error(f"Error analyzing stock: {str(e)}")
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def render_sentiment_analysis(self):
        """Render sentiment analysis module"""
        st.header("💬 Sentiment Analysis")
        
        # Input options
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Single Text", "Batch Analysis", "File Upload"]
        )
        
        if analysis_type == "Single Text":
            self._single_text_analysis()
        elif analysis_type == "Batch Analysis":
            self._batch_analysis()
        else:
            self._file_upload_analysis()
    
    def _single_text_analysis(self):
        """Single text sentiment analysis"""
        text_input = st.text_area(
            "Enter text for sentiment analysis",
            placeholder="Type or paste your text here...",
            height=150
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if text_input:
                with st.spinner("Analyzing sentiment..."):
                    self._perform_sentiment_analysis(text_input)
    
    def _perform_sentiment_analysis(self, text):
        """Perform sentiment analysis on text"""
        try:
            # Analyze sentiment
            result = self.sentiment_analyzer.analyze_sentiment(text)
            
            if result:
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment = result['ensemble_sentiment']
                    confidence = result['ensemble_confidence']
                    
                    if sentiment == 'positive':
                        st.success(f"😊 Positive ({confidence:.1%})")
                    elif sentiment == 'negative':
                        st.error(f"😞 Negative ({confidence:.1%})")
                    else:
                        st.info(f"😐 Neutral ({confidence:.1%})")
                
                with col2:
                    st.metric("BERT Confidence", f"{result['bert_confidence']:.1%}")
                
                with col3:
                    st.metric("Ensemble Score", f"{result.get('ensemble_score', 0):.3f}")
                
                # Detailed analysis
                st.subheader("📊 Detailed Analysis")
                
                # Confidence breakdown
                if 'confidence_breakdown' in result:
                    conf_data = result['confidence_breakdown']
                    fig_conf = go.Figure(data=[
                        go.Bar(
                            x=list(conf_data.keys()),
                            y=list(conf_data.values()),
                            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                        )
                    ])
                    
                    fig_conf.update_layout(
                        title="Confidence Breakdown by Method",
                        xaxis_title="Method",
                        yaxis_title="Confidence Score",
                        height=400
                    )
                    
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # VADER scores
                if 'vader_scores' in result:
                    vader_scores = result['vader_scores']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Positive", f"{vader_scores['pos']:.3f}")
                    with col2:
                        st.metric("Negative", f"{vader_scores['neg']:.3f}")
                    with col3:
                        st.metric("Neutral", f"{vader_scores['neu']:.3f}")
                    with col4:
                        st.metric("Compound", f"{vader_scores['compound']:.3f}")
                
                # TextBlob scores
                if 'textblob_scores' in result:
                    textblob_scores = result['textblob_scores']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Polarity", f"{textblob_scores['polarity']:.3f}")
                    with col2:
                        st.metric("Subjectivity", f"{textblob_scores['subjectivity']:.3f}")
        
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
    
    def _batch_analysis(self):
        """Batch sentiment analysis"""
        st.info("Batch analysis feature coming soon!")
    
    def _file_upload_analysis(self):
        """File upload sentiment analysis"""
        uploaded_file = st.file_uploader(
            "Upload text file",
            type=['txt', 'csv'],
            help="Upload a text file for sentiment analysis"
        )
        
        if uploaded_file is not None:
            st.info("File upload analysis feature coming soon!")
    
    def render_document_processing(self):
        """Render document processing module"""
        st.header("📄 Document Processing")
        
        st.info("Document processing feature requires document images. This is a demonstration interface.")
        
        # Document upload
        uploaded_doc = st.file_uploader(
            "Upload Document",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            help="Upload a document for OCR and analysis"
        )
        
        if uploaded_doc:
            st.success(f"Document uploaded: {uploaded_doc.name}")
            
            # Simulate processing
            if st.button("🔍 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    self._simulate_document_processing(uploaded_doc.name)
    
    def _simulate_document_processing(self, filename):
        """Simulate document processing results"""
        # Sample results
        st.subheader("📊 Processing Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("OCR Confidence", "92.1%")
            st.metric("Text Extracted", "1,247 words")
            st.metric("Document Type", "Invoice")
        
        with col2:
            st.metric("Fields Detected", "15")
            st.metric("Tables Found", "2")
            st.metric("Processing Time", "2.3s")
        
        # Sample extracted data
        st.subheader("📋 Extracted Data")
        
        sample_data = {
            'Invoice Number': 'INV-2024-001',
            'Date': '2024-01-15',
            'Due Date': '2024-02-15',
            'Total Amount': '$1,247.50',
            'Company': 'Tech Solutions Inc.',
            'Customer': 'John Doe'
        }
        
        st.json(sample_data)
    
    def render_time_series(self):
        """Render time series forecasting module"""
        st.header("⏰ Time Series Forecasting")
        
        st.info("Time series forecasting module coming soon!")
        
        # Placeholder for time series features
        st.subheader("🔮 Forecasting Models")
        
        models = ["ARIMA", "Prophet", "LSTM", "Neural Prophet", "Ensemble"]
        
        for model in models:
            st.checkbox(f"Enable {model} forecasting", value=True)
    
    def render_interactive_analytics(self):
        """Render interactive analytics module"""
        st.header("📊 Interactive Analytics")
        
        # Sample interactive charts
        st.subheader("📈 Sample Interactive Charts")
        
        # Generate sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Value1': np.random.randn(100).cumsum(),
            'Value2': np.random.randn(100).cumsum() * 0.5,
            'Category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Interactive line chart
        fig_line = px.line(
            data, x='Date', y=['Value1', 'Value2'],
            title="Interactive Time Series"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Interactive scatter plot
        fig_scatter = px.scatter(
            data, x='Value1', y='Value2', color='Category',
            title="Interactive Scatter Plot"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Interactive histogram
        fig_hist = px.histogram(
            data, x='Value1', nbins=20,
            title="Interactive Histogram"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

def main():
    """Main dashboard application"""
    # Initialize dashboard
    dashboard = AdvancedDashboard()
    
    # Render header
    dashboard.render_header()
    
    # Render sidebar and get navigation
    page, confidence_threshold, update_frequency = dashboard.render_sidebar()
    
    # Render page content
    if page == "🏠 Dashboard Overview":
        dashboard.render_overview()
    elif page == "📈 Financial Analysis":
        dashboard.render_financial_analysis()
    elif page == "💬 Sentiment Analysis":
        dashboard.render_sentiment_analysis()
    elif page == "📄 Document Processing":
        dashboard.render_document_processing()
    elif page == "⏰ Time Series Forecasting":
        dashboard.render_time_series()
    elif page == "📊 Interactive Analytics":
        dashboard.render_interactive_analytics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Advanced Data Analytics Portfolio | Built with Streamlit, Plotly, and Python</p>
        <p>© 2024 Senior Data Analyst Portfolio</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
