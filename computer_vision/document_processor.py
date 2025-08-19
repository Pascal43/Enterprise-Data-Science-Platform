"""
Advanced Document Processing with OCR and Data Extraction

This module provides comprehensive document processing capabilities including:
- OCR text extraction using Tesseract
- Invoice data extraction and parsing
- Document classification
- Table extraction
- Form field detection
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import re
import json
from typing import Dict, List, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')

class DocumentProcessor:
    """
    Advanced Document Processing System
    
    Features:
    - Multi-format document support (PDF, images, scanned documents)
    - OCR text extraction with preprocessing
    - Invoice data extraction
    - Table detection and extraction
    - Form field recognition
    - Document classification
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize the Document Processor
        
        Args:
            tesseract_path (str): Path to Tesseract executable
        """
        self.tesseract_path = tesseract_path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Document type patterns
        self.document_patterns = {
            'invoice': [
                r'invoice',
                r'bill\s+to',
                r'amount\s+due',
                r'total\s+amount',
                r'payment\s+terms'
            ],
            'receipt': [
                r'receipt',
                r'purchase\s+date',
                r'merchant',
                r'card\s+ending'
            ],
            'contract': [
                r'contract',
                r'agreement',
                r'terms\s+and\s+conditions',
                r'effective\s+date'
            ]
        }
        
        # Invoice field patterns
        self.invoice_fields = {
            'invoice_number': r'invoice\s*#?\s*(\w+)',
            'date': r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            'due_date': r'due\s+date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            'total_amount': r'total\s*:?\s*\$?([\d,]+\.?\d*)',
            'subtotal': r'subtotal\s*:?\s*\$?([\d,]+\.?\d*)',
            'tax': r'tax\s*:?\s*\$?([\d,]+\.?\d*)',
            'company_name': r'company\s*:?\s*([A-Za-z\s&]+)',
            'customer_name': r'bill\s+to\s*:?\s*([A-Za-z\s]+)'
        }
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path (str): Path to the image file
            preprocess (bool): Whether to preprocess the image
            
        Returns:
            str: Extracted text
        """
        try:
            if preprocess:
                image = self.preprocess_image(image_path)
                if image is None:
                    return ""
            else:
                image = cv2.imread(image_path)
                if image is None:
                    return ""
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return ""
    
    def extract_text_with_confidence(self, image_path: str) -> Dict:
        """
        Extract text with confidence scores
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Text and confidence data
        """
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return {"text": "", "confidence": 0, "words": []}
            
            # Extract text with confidence
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Process results
            words = []
            total_confidence = 0
            valid_words = 0
            
            for i, conf in enumerate(data['conf']):
                if conf > 0:  # Valid confidence score
                    word = data['text'][i].strip()
                    if word:
                        words.append({
                            'text': word,
                            'confidence': conf,
                            'bbox': (data['left'][i], data['top'][i], 
                                   data['width'][i], data['height'][i])
                        })
                        total_confidence += conf
                        valid_words += 1
            
            avg_confidence = total_confidence / valid_words if valid_words > 0 else 0
            full_text = ' '.join([word['text'] for word in words])
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'words': words
            }
            
        except Exception as e:
            print(f"Error extracting text with confidence: {str(e)}")
            return {"text": "", "confidence": 0, "words": []}
    
    def classify_document(self, text: str) -> Dict:
        """
        Classify document type based on content
        
        Args:
            text (str): Extracted text from document
            
        Returns:
            dict: Document classification results
        """
        try:
            text_lower = text.lower()
            scores = {}
            
            for doc_type, patterns in self.document_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    score += len(matches)
                scores[doc_type] = score
            
            # Determine primary document type
            if scores:
                primary_type = max(scores, key=scores.get)
                confidence = scores[primary_type] / max(sum(scores.values()), 1)
            else:
                primary_type = "unknown"
                confidence = 0
            
            return {
                'document_type': primary_type,
                'confidence': confidence,
                'scores': scores
            }
            
        except Exception as e:
            print(f"Error classifying document: {str(e)}")
            return {'document_type': 'unknown', 'confidence': 0, 'scores': {}}
    
    def extract_invoice_data(self, text: str) -> Dict:
        """
        Extract structured data from invoice text
        
        Args:
            text (str): Invoice text
            
        Returns:
            dict: Extracted invoice data
        """
        try:
            extracted_data = {}
            
            for field_name, pattern in self.invoice_fields.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    extracted_data[field_name] = matches[0].strip()
                else:
                    extracted_data[field_name] = None
            
            # Extract line items (simplified)
            lines = text.split('\n')
            line_items = []
            
            for line in lines:
                # Look for patterns like "Item Description $XX.XX"
                item_match = re.search(r'([A-Za-z\s]+)\s+\$?([\d,]+\.?\d*)', line)
                if item_match:
                    line_items.append({
                        'description': item_match.group(1).strip(),
                        'amount': item_match.group(2)
                    })
            
            extracted_data['line_items'] = line_items
            
            return extracted_data
            
        except Exception as e:
            print(f"Error extracting invoice data: {str(e)}")
            return {}
    
    def extract_tables(self, image_path: str) -> List[pd.DataFrame]:
        """
        Extract tables from image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            list: List of extracted tables as DataFrames
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect table structure
            # This is a simplified approach - in practice, you might use
            # more sophisticated table detection algorithms
            
            # Extract table data using Tesseract
            table_data = pytesseract.image_to_data(
                gray, output_type=pytesseract.Output.DATAFRAME
            )
            
            # Filter valid text entries
            valid_data = table_data[table_data['conf'] > 0]
            
            if valid_data.empty:
                return []
            
            # Group by line to create table structure
            tables = []
            current_table = []
            current_line = -1
            
            for _, row in valid_data.iterrows():
                if row['text'].strip():
                    if row['top'] != current_line:
                        if current_table:
                            # Process current table
                            df = self._create_table_dataframe(current_table)
                            if not df.empty:
                                tables.append(df)
                        current_table = []
                        current_line = row['top']
                    
                    current_table.append({
                        'text': row['text'].strip(),
                        'left': row['left'],
                        'top': row['top']
                    })
            
            # Process last table
            if current_table:
                df = self._create_table_dataframe(current_table)
                if not df.empty:
                    tables.append(df)
            
            return tables
            
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []
    
    def _create_table_dataframe(self, table_data: List[Dict]) -> pd.DataFrame:
        """Create DataFrame from table data"""
        try:
            if not table_data:
                return pd.DataFrame()
            
            # Sort by position
            sorted_data = sorted(table_data, key=lambda x: (x['top'], x['left']))
            
            # Group into columns based on x-position
            columns = {}
            for item in sorted_data:
                col_key = item['left'] // 50  # Approximate column width
                if col_key not in columns:
                    columns[col_key] = []
                columns[col_key].append(item['text'])
            
            # Create DataFrame
            max_rows = max(len(col) for col in columns.values()) if columns else 0
            df_data = {}
            
            for col_idx, col_data in columns.items():
                col_name = f"Column_{col_idx}"
                df_data[col_name] = col_data + [''] * (max_rows - len(col_data))
            
            return pd.DataFrame(df_data)
            
        except Exception as e:
            print(f"Error creating table DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def detect_form_fields(self, image_path: str) -> List[Dict]:
        """
        Detect form fields in document
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            list: List of detected form fields
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect rectangles (potential form fields)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            form_fields = []
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's a rectangle
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by size (avoid very small or very large rectangles)
                    if 50 < w < 500 and 20 < h < 100:
                        form_fields.append({
                            'type': 'input_field',
                            'bbox': (x, y, w, h),
                            'area': w * h
                        })
            
            return form_fields
            
        except Exception as e:
            print(f"Error detecting form fields: {str(e)}")
            return []
    
    def process_document(self, image_path: str) -> Dict:
        """
        Complete document processing pipeline
        
        Args:
            image_path (str): Path to the document image
            
        Returns:
            dict: Complete processing results
        """
        try:
            # Extract text
            text_result = self.extract_text_with_confidence(image_path)
            
            # Classify document
            classification = self.classify_document(text_result['text'])
            
            # Extract specific data based on document type
            extracted_data = {}
            if classification['document_type'] == 'invoice':
                extracted_data = self.extract_invoice_data(text_result['text'])
            
            # Extract tables
            tables = self.extract_tables(image_path)
            
            # Detect form fields
            form_fields = self.detect_form_fields(image_path)
            
            return {
                'text': text_result['text'],
                'confidence': text_result['confidence'],
                'classification': classification,
                'extracted_data': extracted_data,
                'tables': [df.to_dict('records') for df in tables],
                'form_fields': form_fields,
                'processing_metadata': {
                    'image_path': image_path,
                    'text_length': len(text_result['text']),
                    'word_count': len(text_result['words'])
                }
            }
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return {}
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save processing results to file
        
        Args:
            results (dict): Processing results
            output_path (str): Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = DocumentProcessor()
    
    # Example usage
    print("Document Processing System")
    print("=" * 50)
    
    # Note: You would need actual document images to test
    # This is a demonstration of the API
    
    # Example processing pipeline
    # results = processor.process_document("sample_invoice.jpg")
    # processor.save_results(results, "output_results.json")
    
    print("Document processing system initialized successfully!")
    print("Use process_document() method to analyze documents.")
