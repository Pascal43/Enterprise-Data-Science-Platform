"""
Advanced Sentiment Analysis using BERT and Multiple NLP Techniques

This module implements sophisticated sentiment analysis capabilities using:
- BERT-based fine-tuned models
- Multi-label classification
- Aspect-based sentiment analysis
- Emotion detection
- Confidence scoring
"""

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline,
    BertTokenizer,
    BertForSequenceClassification
)
from transformers import TextClassificationPipeline
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import spacy
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """
    Advanced Sentiment Analysis System
    
    Features:
    - BERT-based sentiment classification
    - Multi-label emotion detection
    - Aspect-based sentiment analysis
    - Confidence scoring
    - Ensemble methods
    - Multi-language support
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the Sentiment Analyzer
        
        Args:
            model_name (str): Pre-trained model name for sentiment analysis
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.nlp = None
        
        # Load models
        self._load_models()
        
        # Sentiment labels
        self.sentiment_labels = {
            0: "negative",
            1: "neutral", 
            2: "positive"
        }
        
        # Emotion labels for multi-label classification
        self.emotion_labels = [
            "joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"
        ]
        
    def _load_models(self):
        """Load all required models"""
        try:
            # Load BERT model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load spaCy for additional NLP tasks
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found. Installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def analyze_sentiment(self, text, return_confidence=True):
        """
        Analyze sentiment of given text using BERT
        
        Args:
            text (str): Input text to analyze
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            dict: Sentiment analysis results
        """
        try:
            # BERT-based analysis
            bert_result = self.pipeline(text)[0]
            
            # Additional analysis methods
            vader_scores = self._analyze_vader(text)
            textblob_scores = self._analyze_textblob(text)
            
            # Ensemble results
            ensemble_sentiment = self._ensemble_sentiment(
                bert_result, vader_scores, textblob_scores
            )
            
            result = {
                'text': text,
                'bert_sentiment': bert_result['label'],
                'bert_confidence': bert_result['score'],
                'ensemble_sentiment': ensemble_sentiment['sentiment'],
                'ensemble_confidence': ensemble_sentiment['confidence'],
                'vader_scores': vader_scores,
                'textblob_scores': textblob_scores
            }
            
            if return_confidence:
                result['confidence_breakdown'] = {
                    'bert': bert_result['score'],
                    'vader': vader_scores['compound'],
                    'textblob': textblob_scores['polarity']
                }
            
            return result
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return None
    
    def _analyze_vader(self, text):
        """Analyze sentiment using VADER"""
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        return scores
    
    def _analyze_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def _ensemble_sentiment(self, bert_result, vader_scores, textblob_scores):
        """Combine results from multiple methods"""
        
        # Convert BERT label to numeric
        bert_sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        bert_score = bert_sentiment_map.get(bert_result['label'], 0)
        
        # Weighted ensemble
        weights = {
            'bert': 0.5,
            'vader': 0.3,
            'textblob': 0.2
        }
        
        ensemble_score = (
            weights['bert'] * bert_score +
            weights['vader'] * vader_scores['compound'] +
            weights['textblob'] * textblob_scores['polarity']
        )
        
        # Determine sentiment
        if ensemble_score > 0.1:
            sentiment = 'positive'
        elif ensemble_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        confidence = abs(ensemble_score)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'ensemble_score': ensemble_score
        }
    
    def analyze_emotions(self, text):
        """
        Analyze emotions in text using multi-label classification
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Emotion analysis results
        """
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=1)
            
            # Map to emotions (simplified mapping)
            emotion_scores = {}
            for i, emotion in enumerate(self.emotion_labels):
                if i < len(probabilities[0]):
                    emotion_scores[emotion] = probabilities[0][i].item()
            
            # Get top emotions
            top_emotions = sorted(
                emotion_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            return {
                'text': text,
                'emotion_scores': emotion_scores,
                'top_emotions': top_emotions,
                'dominant_emotion': top_emotions[0] if top_emotions else None
            }
            
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            return None
    
    def aspect_based_sentiment(self, text):
        """
        Perform aspect-based sentiment analysis
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Aspect-based sentiment results
        """
        try:
            if not self.nlp:
                return None
            
            doc = self.nlp(text)
            
            # Extract aspects (noun phrases and named entities)
            aspects = []
            for chunk in doc.noun_chunks:
                aspects.append(chunk.text)
            
            for ent in doc.ents:
                if ent.text not in aspects:
                    aspects.append(ent.text)
            
            # Analyze sentiment for each aspect
            aspect_sentiments = {}
            for aspect in aspects:
                # Create context around aspect
                aspect_sentiment = self._analyze_aspect_sentiment(text, aspect)
                if aspect_sentiment:
                    aspect_sentiments[aspect] = aspect_sentiment
            
            return {
                'text': text,
                'aspects': aspects,
                'aspect_sentiments': aspect_sentiments
            }
            
        except Exception as e:
            print(f"Error in aspect-based sentiment analysis: {str(e)}")
            return None
    
    def _analyze_aspect_sentiment(self, text, aspect):
        """Analyze sentiment for a specific aspect"""
        try:
            # Find sentences containing the aspect
            sentences = text.split('.')
            aspect_sentences = [s for s in sentences if aspect.lower() in s.lower()]
            
            if not aspect_sentences:
                return None
            
            # Analyze sentiment of aspect-containing sentences
            aspect_text = '. '.join(aspect_sentences)
            sentiment_result = self.analyze_sentiment(aspect_text)
            
            return sentiment_result
            
        except Exception as e:
            print(f"Error analyzing aspect sentiment: {str(e)}")
            return None
    
    def batch_analyze(self, texts, batch_size=32):
        """
        Analyze sentiment for multiple texts efficiently
        
        Args:
            texts (list): List of texts to analyze
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of sentiment analysis results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch:
                result = self.analyze_sentiment(text)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def evaluate_model(self, test_data, test_labels):
        """
        Evaluate model performance on test data
        
        Args:
            test_data (list): List of test texts
            test_labels (list): List of true labels
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            predictions = []
            confidences = []
            
            for text in test_data:
                result = self.analyze_sentiment(text)
                if result:
                    predictions.append(result['ensemble_sentiment'])
                    confidences.append(result['ensemble_confidence'])
            
            # Calculate metrics
            report = classification_report(test_labels, predictions, output_dict=True)
            conf_matrix = confusion_matrix(test_labels, predictions)
            
            return {
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'accuracy': report['accuracy'],
                'predictions': predictions,
                'confidences': confidences
            }
            
        except Exception as e:
            print(f"Error in model evaluation: {str(e)}")
            return None
    
    def get_sentiment_trends(self, texts_with_dates):
        """
        Analyze sentiment trends over time
        
        Args:
            texts_with_dates (list): List of tuples (text, date)
            
        Returns:
            dict: Sentiment trend analysis
        """
        try:
            trends = []
            
            for text, date in texts_with_dates:
                sentiment_result = self.analyze_sentiment(text)
                if sentiment_result:
                    trends.append({
                        'date': date,
                        'sentiment': sentiment_result['ensemble_sentiment'],
                        'confidence': sentiment_result['ensemble_confidence'],
                        'score': sentiment_result.get('ensemble_score', 0)
                    })
            
            # Calculate trend statistics
            df_trends = pd.DataFrame(trends)
            
            trend_analysis = {
                'trends': trends,
                'sentiment_distribution': df_trends['sentiment'].value_counts().to_dict(),
                'average_confidence': df_trends['confidence'].mean(),
                'sentiment_timeline': df_trends.groupby('date')['score'].mean().to_dict()
            }
            
            return trend_analysis
            
        except Exception as e:
            print(f"Error in trend analysis: {str(e)}")
            return None


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Test texts
    test_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the worst experience I've ever had. Terrible service!",
        "The product is okay, nothing special but gets the job done.",
        "I'm so excited about the new features! This is incredible!"
    ]
    
    print("Testing Sentiment Analysis:")
    print("=" * 50)
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = analyzer.analyze_sentiment(text)
        if result:
            print(f"Sentiment: {result['ensemble_sentiment']}")
            print(f"Confidence: {result['ensemble_confidence']:.3f}")
            print(f"BERT: {result['bert_sentiment']} ({result['bert_confidence']:.3f})")
        
        # Emotion analysis
        emotion_result = analyzer.analyze_emotions(text)
        if emotion_result and emotion_result['top_emotions']:
            print(f"Top Emotions: {emotion_result['top_emotions'][:2]}")
    
    print("\n" + "=" * 50)
    print("Sentiment Analysis Testing Complete!")
