"""
Image Parser with OCR Support

This parser handles PNG, JPG, JPEG, SVG, and other image formats.
It uses OCR (Optical Character Recognition) to extract text from images.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from .base_parser import BaseParser

# Set up logging
logger = logging.getLogger("ImageParser")

class ImageParser(BaseParser):
    """
    Image parser with OCR capabilities for extracting text from images
    Supports: PNG, JPG, JPEG, GIF, BMP, TIFF, SVG
    """
    
    def __init__(self, enable_ocr: bool = True, min_confidence: float = 0.6):
        """
        Initialize ImageParser
        
        Args:
            enable_ocr: Whether to enable OCR text extraction
            min_confidence: Minimum confidence score for OCR text
        """
        self.enable_ocr = enable_ocr
        self.min_confidence = min_confidence
        self.ocr_available = False
        
        # Try to import OCR libraries
        try:
            import pytesseract
            from PIL import Image
            self.pytesseract = pytesseract
            self.PIL_Image = Image
            self.ocr_available = True
            logger.info("OCR support available (pytesseract + PIL)")
        except ImportError:
            logger.warning("OCR libraries not available. Install: pip install pytesseract pillow")
            self.ocr_available = False
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse image file and extract text using OCR
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Get file info
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            extracted_text = ""
            metadata = {
                "file_type": "image",
                "file_extension": file_ext,
                "file_size": file_size,
                "ocr_enabled": self.enable_ocr and self.ocr_available
            }
            
            # Handle SVG files (text-based)
            if file_ext == '.svg':
                extracted_text = self._parse_svg(file_path)
                metadata["parsing_method"] = "svg_text_extraction"
            
            # Handle other image formats with OCR
            elif self.enable_ocr and self.ocr_available:
                extracted_text = self._extract_text_with_ocr(file_path)
                metadata["parsing_method"] = "ocr"
            else:
                # No OCR available - provide descriptive text
                extracted_text = f"[Image file: {os.path.basename(file_path)}]"
                metadata["parsing_method"] = "filename_only"
                logger.warning(f"No OCR available for {file_path}. Consider installing pytesseract and pillow.")
            
            return {
                "raw_text": extracted_text,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error parsing image file {file_path}: {e}")
            return {
                "raw_text": f"[Image parsing error: {e}]",
                "metadata": {"error": str(e)}
            }
    
    def _parse_svg(self, file_path: str) -> str:
        """Extract text content from SVG files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # Simple text extraction from SVG
            import re
            
            # Extract text from <text> tags
            text_matches = re.findall(r'<text[^>]*>(.*?)</text>', svg_content, re.DOTALL | re.IGNORECASE)
            
            # Extract text from <title> tags
            title_matches = re.findall(r'<title[^>]*>(.*?)</title>', svg_content, re.DOTALL | re.IGNORECASE)
            
            # Extract text from <desc> tags
            desc_matches = re.findall(r'<desc[^>]*>(.*?)</desc>', svg_content, re.DOTALL | re.IGNORECASE)
            
            # Combine all text
            all_text = []
            if title_matches:
                all_text.extend(title_matches)
            if desc_matches:
                all_text.extend(desc_matches)
            if text_matches:
                all_text.extend(text_matches)
            
            # Clean up the text
            cleaned_text = []
            for text in all_text:
                # Remove HTML tags and clean whitespace
                clean = re.sub(r'<[^>]+>', '', text)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if clean:
                    cleaned_text.append(clean)
            
            result = '\n'.join(cleaned_text) if cleaned_text else ""
            
            if result:
                logger.info(f"Extracted {len(result)} characters from SVG text elements")
            else:
                result = f"[SVG file: {os.path.basename(file_path)} - no text elements found]"
                
            return result
            
        except Exception as e:
            logger.error(f"Error parsing SVG file: {e}")
            return f"[SVG parsing error: {e}]"
    
    def _extract_text_with_ocr(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            # Open image
            image = self.PIL_Image.open(file_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using OCR
            extracted_text = self.pytesseract.image_to_string(image)
            
            # Get confidence scores if available
            try:
                data = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                if avg_confidence < self.min_confidence * 100:
                    logger.warning(f"Low OCR confidence ({avg_confidence:.1f}%) for {file_path}")
                    
            except Exception:
                # Confidence calculation failed, but we still have the text
                pass
            
            # Clean up the extracted text
            if extracted_text:
                # Remove excessive whitespace
                cleaned_text = re.sub(r'\s+', ' ', extracted_text).strip()
                
                if len(cleaned_text) > 10:  # Meaningful content threshold
                    logger.info(f"OCR extracted {len(cleaned_text)} characters from {file_path}")
                    return cleaned_text
                else:
                    logger.warning(f"OCR extracted minimal content from {file_path}")
                    return f"[Image: {os.path.basename(file_path)} - minimal text detected]"
            else:
                return f"[Image: {os.path.basename(file_path)} - no text detected]"
                
        except Exception as e:
            logger.error(f"OCR extraction failed for {file_path}: {e}")
            return f"[Image: {os.path.basename(file_path)} - OCR failed: {e}]"
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if the file format is supported"""
        supported_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.svg']
        ext = os.path.splitext(file_path)[1].lower()
        return ext in supported_extensions
