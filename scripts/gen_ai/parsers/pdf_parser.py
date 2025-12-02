import os
import string
import logging
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional, Tuple
from .base_parser import BaseParser
from PyPDF2 import PdfReader
import re
import tempfile
import nltk
import ssl

# Fix SSL certificate verification issue for NLTK downloads on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK resources
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception as e:
        logging.warning(f"Failed to download vader_lexicon: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PDFParser")

class PDFParser(BaseParser):
    """
    Enhanced PDF parser with advanced features:
    - Metadata extraction
    - Text structure preservation
    - Optional OCR for scanned documents
    - Image detection
    - Table detection
    - Smart error handling
    """

    def __init__(self, enable_ocr: bool = False,
                min_confidence: float = 0.75,
                extract_images: bool = False,
                extract_tables: bool = False):
        """
        Initialize the PDF parser with configurable options

        Args:
            enable_ocr: Whether to use OCR for text extraction when regular extraction fails
            min_confidence: Minimum confidence threshold for OCR results (0-1)
            extract_images: Whether to extract and analyze images from the PDF
            extract_tables: Whether to detect and extract tables from the PDF
        """
        self.enable_ocr = enable_ocr
        self.min_confidence = min_confidence
        self.extract_images = extract_images
        self.extract_tables = extract_tables

        # Initialize OCR if enabled
        if self.enable_ocr:
            try:
                import pytesseract
                from PIL import Image
                self.ocr_available = True
                logger.info("OCR support initialized")
            except ImportError:
                logger.warning("pytesseract or Pillow not installed. OCR features disabled.")
                self.ocr_available = False
        else:
            self.ocr_available = False

    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a PDF file and extract its content with enhanced structure

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing the extracted content and metadata
        """
        # Basic validation
        if not file_path.lower().endswith(".pdf"):
            return {"error": "Invalid file type: not a PDF", "success": False}
        if not os.path.exists(file_path):
            return {"error": "File not found", "success": False}
        if not os.access(file_path, os.R_OK):
            return {"error": "File not readable", "success": False}
        if os.path.getsize(file_path) == 0:
            return {"error": "File is empty", "success": False}

        result = {
            "success": True,
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "pages": [],
            "metadata": {},
            "text_content": "",
            "has_scanned_content": False
        }

        # Extract text and metadata using PyPDF2 first
        try:
            self._extract_with_pypdf2(file_path, result)
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            result["errors"] = [f"PyPDF2 extraction error: {str(e)}"]

        # If PyPDF2 failed or text is very limited, try PyMuPDF (fitz)
        if not result.get("text_content") or len(result["text_content"]) < 100:
            try:
                logger.info("Using PyMuPDF as fallback for better text extraction")
                self._extract_with_pymupdf(file_path, result)
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
                if not result.get("errors"):
                    result["errors"] = []
                result["errors"].append(f"PyMuPDF extraction error: {str(e)}")

        # If text extraction failed and OCR is enabled, try OCR
        if (not result["text_content"] or len(result["text_content"]) < 50) and self.enable_ocr and self.ocr_available:
            try:
                self._perform_ocr(file_path, result)
            except Exception as e:
                logger.warning(f"OCR extraction failed: {e}")
                if not result.get("errors"):
                    result["errors"] = []
                result["errors"].append(f"OCR extraction error: {str(e)}")

        # If text content is still empty after all attempts
        if not result["text_content"]:
            result["text_content"] = "[No text could be extracted from the PDF]"
            result["success"] = False

        # Extract special content if enabled
        if self.extract_images:
            try:
                self._extract_images(file_path, result)
            except Exception as e:
                logger.warning(f"Image extraction failed: {e}")

        if self.extract_tables:
            try:
                self._extract_tables(file_path, result)
            except Exception as e:
                logger.warning(f"Table extraction failed: {e}")

        # Include raw text for backward compatibility
        result["raw_text"] = result["text_content"]

        return result

    def _extract_with_pypdf2(self, file_path: str, result: Dict[str, Any]) -> None:
        """Extract text and metadata using PyPDF2"""
        reader = PdfReader(file_path)
        text = ""
        pages_content = []

        # Extract metadata
        if reader.metadata:
            metadata = reader.metadata
            result["metadata"] = {
                "title": metadata.get('/Title', ""),
                "author": metadata.get('/Author', ""),
                "subject": metadata.get('/Subject', ""),
                "creator": metadata.get('/Creator', ""),
                "producer": metadata.get('/Producer', ""),
                "creation_date": metadata.get('/CreationDate', ""),
                "modification_date": metadata.get('/ModDate', "")
            }

            # Clean up metadata values
            for key, value in result["metadata"].items():
                if isinstance(value, str):
                    result["metadata"][key] = value.strip()

        # Extract text from each page
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = self._clean_text(page_text)

            # Store page-specific information
            page_info = {
                "page_number": i + 1,
                "text": page_text,
                "is_empty": len(page_text.strip()) == 0
            }

            pages_content.append(page_info)
            text += page_text + "\n\n"

        result["pages"] = pages_content
        result["text_content"] = text.strip()
        result["total_pages"] = len(reader.pages)

    def _extract_with_pymupdf(self, file_path: str, result: Dict[str, Any]) -> None:
        """Extract text and document structure using PyMuPDF (fitz)"""
        doc = fitz.open(file_path)
        text = ""
        pages_content = []
        has_scanned_content = False

        # Extract metadata if not already present
        if not result.get("metadata"):
            metadata = doc.metadata
            result["metadata"] = {
                "title": metadata.get('title', ""),
                "author": metadata.get('author', ""),
                "subject": metadata.get('subject', ""),
                "creator": metadata.get('creator', ""),
                "producer": metadata.get('producer', ""),
                "creation_date": metadata.get('creationDate', ""),
                "modification_date": metadata.get('modDate', "")
            }

        # Process each page
        for i, page in enumerate(doc):
            # Extract text with more structure preservation
            page_text = page.get_text("text")
            blocks = page.get_text("blocks")

            # Check if this might be a scanned page (few text blocks but has images)
            if len(page_text.strip()) < 50 and page.get_images():
                has_scanned_content = True

            # Extract text with structure
            structured_text = ""
            for block in blocks:
                if block[6] == 0:  # Text block
                    structured_text += block[4] + "\n"

            # Clean the structured text
            structured_text = self._clean_text(structured_text)

            # Update or add page info
            page_info = next((p for p in result["pages"] if p["page_number"] == i + 1), None)
            if page_info:
                # Use the better text version
                if len(structured_text) > len(page_info["text"]):
                    page_info["text"] = structured_text
            else:
                page_info = {
                    "page_number": i + 1,
                    "text": structured_text,
                    "is_empty": len(structured_text.strip()) == 0
                }
                pages_content.append(page_info)

            text += structured_text + "\n\n"

        # Update result if we have better content
        if not result.get("text_content") or len(text) > len(result["text_content"]):
            result["text_content"] = text.strip()

        if pages_content and (not result.get("pages") or len(pages_content) > len(result["pages"])):
            result["pages"] = pages_content

        result["total_pages"] = len(doc)
        result["has_scanned_content"] = has_scanned_content

        doc.close()

    def _perform_ocr(self, file_path: str, result: Dict[str, Any]) -> None:
        """Perform OCR on the PDF pages that appear to be scanned"""
        import pytesseract
        from PIL import Image

        doc = fitz.open(file_path)
        text = result.get("text_content", "")
        ocr_applied = False

        for i, page in enumerate(doc):
            page_info = next((p for p in result["pages"] if p["page_number"] == i + 1), None)

            # Skip pages with enough text content
            if page_info and len(page_info["text"].strip()) > 50:
                continue

            # Convert page to image
            pix = page.get_pixmap(alpha=False)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                pix.save(tmp.name)
                img = Image.open(tmp.name)

                # Perform OCR
                ocr_text = pytesseract.image_to_string(img)
                ocr_text = self._clean_text(ocr_text)

                # Update page info
                if page_info:
                    page_info["text"] = ocr_text
                    page_info["ocr_applied"] = True
                else:
                    page_info = {
                        "page_number": i + 1,
                        "text": ocr_text,
                        "is_empty": len(ocr_text.strip()) == 0,
                        "ocr_applied": True
                    }
                    result["pages"].append(page_info)

                text += ocr_text + "\n\n"
                ocr_applied = True

        if ocr_applied:
            result["text_content"] = text.strip()
            result["ocr_applied"] = True

        doc.close()

    def _extract_images(self, file_path: str, result: Dict[str, Any]) -> None:
        """Extract information about images in the PDF"""
        doc = fitz.open(file_path)
        images = []

        for i, page in enumerate(doc):
            page_images = page.get_images(full=True)

            for img_index, img in enumerate(page_images):
                xref = img[0]
                base_img = doc.extract_image(xref)

                if base_img:
                    image_info = {
                        "page_number": i + 1,
                        "image_index": img_index + 1,
                        "width": base_img["width"],
                        "height": base_img["height"],
                        "size_bytes": len(base_img["image"]),
                        "format": base_img["ext"]
                    }

                    images.append(image_info)

        result["images"] = images
        result["image_count"] = len(images)

        doc.close()

    def _extract_tables(self, file_path: str, result: Dict[str, Any]) -> None:
        """Detect and extract tables from the PDF"""
        # Try to use tabula-py if available
        try:
            import tabula
            tables_data = []

            # Extract tables from PDF
            tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)

            for i, table in enumerate(tables):
                table_data = {
                    "table_index": i + 1,
                    "rows": table.shape[0],
                    "columns": table.shape[1],
                    "headers": list(table.columns),
                    "data": table.to_dict('records')
                }
                tables_data.append(table_data)

            result["tables"] = tables_data
            result["table_count"] = len(tables_data)

        except ImportError:
            # If tabula is not available, use regex-based table detection
            logger.warning("Tabula not installed, falling back to basic table detection")
            self._detect_tables_with_regex(result)

    def _detect_tables_with_regex(self, result: Dict[str, Any]) -> None:
        """Basic table detection using regex patterns in text content"""
        text_content = result.get("text_content", "")

        # Simple heuristic: look for patterns of whitespace-aligned columns
        table_patterns = [
            r'((?:\|[^\|]+)+\|)',  # Text tables with pipe separators
            r'((?:\+[-]+)+\+)',    # ASCII tables with plus and dash
            r'((?:\+-+)+\+)',      # Simplified ASCII tables
        ]

        table_matches = []

        for pattern in table_patterns:
            matches = re.findall(pattern, text_content)
            if matches:
                table_matches.extend(matches)

        result["possible_tables"] = len(table_matches)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""

        # Remove any leading/trailing whitespace
        text = text.strip()

        # Convert various Unicode whitespace characters to standard space
        text = re.sub(r'\s+', ' ', text)

        # Replace multiple newlines with double newline to preserve paragraphs
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove any non-printable characters
        text = ''.join(filter(lambda x: x.isprintable() or x == '\n', text))

        # Remove any non-UTF-8 characters
        text = text.encode('utf-8', 'ignore').decode('utf-8')

        return text

    def get_document_summary(self, file_path: str) -> Dict[str, Any]:
        """
        Get a concise summary of the document without full text extraction

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing document summary
        """
        result = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
        }

        try:
            doc = fitz.open(file_path)
            result.update({
                "total_pages": len(doc),
                "metadata": doc.metadata,
                "has_text": any(page.get_text().strip() for page in doc),
                "has_images": any(page.get_images() for page in doc),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", "")
            })
            doc.close()
        except Exception as e:
            result["error"] = f"Error getting document summary: {str(e)}"

        return result
