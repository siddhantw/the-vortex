"""
Enhanced Parser for Multiple Document Types and External Sources

This module provides comprehensive parsing capabilities for various document types
and external sources including Figma, Confluence, and web scraping.
"""

import os
import json
import yaml
import logging
import requests
import tempfile
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse, urljoin
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing parsers
try:
    from .text_parser import TextParser
    from .markdown_parser import MarkdownParser
    from .word_parser import WordParser
    from .pdf_parser import PDFParser
except ImportError:
    # Fallback for direct execution
    TextParser = None
    MarkdownParser = None
    WordParser = None
    PDFParser = None

# Optional imports for enhanced functionality
try:
    import xlrd
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logger.warning("Excel parsing not available. Install xlrd and openpyxl for Excel support.")

try:
    from bs4 import BeautifulSoup
    import lxml
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    logger.warning("Web scraping not available. Install beautifulsoup4 and lxml for web scraping.")

try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False

class BaseEnhancedParser(ABC):
    """Base class for enhanced parsers"""
    
    def __init__(self):
        self.file_size_limit = 200 * 1024 * 1024  # 200MB
        
    @abstractmethod
    def parse(self, source: str, **kwargs) -> Dict[str, Any]:
        """Parse the source and return structured data"""
        pass
    
    def validate_file_size(self, file_path: str) -> bool:
        """Validate file size is within limits"""
        try:
            size = os.path.getsize(file_path)
            if size > self.file_size_limit:
                logger.error(f"File {file_path} exceeds 200MB limit: {size/1024/1024:.2f}MB")
                return False
            return True
        except OSError:
            logger.error(f"Could not access file: {file_path}")
            return False

class ExcelParser(BaseEnhancedParser):
    """Parser for Excel files (.xlsx, .xls)"""
    
    def parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Parse Excel file and extract requirements"""
        if not EXCEL_AVAILABLE:
            return {"error": "Excel parsing not available", "raw_text": ""}
            
        if not self.validate_file_size(file_path):
            return {"error": "File size exceeds limit", "raw_text": ""}
        
        try:
            # Try different engines
            try:
                df = pd.read_excel(file_path, engine='openpyxl', sheet_name=None)
            except:
                df = pd.read_excel(file_path, engine='xlrd', sheet_name=None)
            
            all_text = ""
            requirements = []
            
            for sheet_name, sheet_data in df.items():
                all_text += f"\n=== Sheet: {sheet_name} ===\n"
                
                # Convert DataFrame to text
                sheet_text = sheet_data.to_string(index=False, na_rep='')
                all_text += sheet_text + "\n"
                
                # Look for requirement-like patterns
                for index, row in sheet_data.iterrows():
                    row_text = ' '.join([str(val) for val in row.values if pd.notna(val)])
                    if self._is_requirement_like(row_text):
                        requirements.append({
                            "id": f"excel_{sheet_name}_{index}",
                            "text": row_text,
                            "source": f"Sheet: {sheet_name}, Row: {index + 1}",
                            "type": "requirement"
                        })
            
            return {
                "raw_text": all_text,
                "requirements": requirements,
                "metadata": {
                    "source_type": "excel",
                    "sheets": list(df.keys()),
                    "total_sheets": len(df)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing Excel file {file_path}: {e}")
            return {"error": str(e), "raw_text": ""}
    
    def _is_requirement_like(self, text: str) -> bool:
        """Check if text looks like a requirement"""
        if len(text.strip()) < 10:
            return False
        
        requirement_keywords = [
            'shall', 'should', 'must', 'will', 'requirement', 'feature',
            'user story', 'acceptance criteria', 'given', 'when', 'then',
            'verify', 'validate', 'ensure', 'test case'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in requirement_keywords)

class CSVParser(BaseEnhancedParser):
    """Parser for CSV files"""
    
    def parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Parse CSV file and extract requirements"""
        if not self.validate_file_size(file_path):
            return {"error": "File size exceeds limit", "raw_text": ""}
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            
            all_text = df.to_string(index=False, na_rep='')
            requirements = []
            
            # Look for requirement columns
            req_columns = [col for col in df.columns if any(
                keyword in col.lower() for keyword in 
                ['requirement', 'description', 'story', 'criteria', 'test']
            )]
            
            for index, row in df.iterrows():
                for col in req_columns:
                    if pd.notna(row[col]) and len(str(row[col]).strip()) > 10:
                        requirements.append({
                            "id": f"csv_{index}_{col}",
                            "text": str(row[col]),
                            "source": f"Row: {index + 1}, Column: {col}",
                            "type": "requirement"
                        })
            
            return {
                "raw_text": all_text,
                "requirements": requirements,
                "metadata": {
                    "source_type": "csv",
                    "columns": list(df.columns),
                    "rows": len(df)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {e}")
            return {"error": str(e), "raw_text": ""}

class JSONParser(BaseEnhancedParser):
    """Parser for JSON files"""
    
    def parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Parse JSON file and extract requirements"""
        if not self.validate_file_size(file_path):
            return {"error": "File size exceeds limit", "raw_text": ""}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            all_text = json.dumps(data, indent=2)
            requirements = self._extract_requirements_from_json(data)
            
            return {
                "raw_text": all_text,
                "requirements": requirements,
                "metadata": {
                    "source_type": "json",
                    "structure": self._analyze_json_structure(data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            return {"error": str(e), "raw_text": ""}
    
    def _extract_requirements_from_json(self, data: Any, path: str = "") -> List[Dict[str, Any]]:
        """Recursively extract requirements from JSON data"""
        requirements = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if this looks like a requirement
                if isinstance(value, str) and self._is_requirement_like(value):
                    requirements.append({
                        "id": f"json_{current_path}",
                        "text": value,
                        "source": f"JSON path: {current_path}",
                        "type": "requirement"
                    })
                elif isinstance(value, (dict, list)):
                    requirements.extend(self._extract_requirements_from_json(value, current_path))
                    
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                if isinstance(item, str) and self._is_requirement_like(item):
                    requirements.append({
                        "id": f"json_{current_path}",
                        "text": item,
                        "source": f"JSON path: {current_path}",
                        "type": "requirement"
                    })
                elif isinstance(item, (dict, list)):
                    requirements.extend(self._extract_requirements_from_json(item, current_path))
        
        return requirements
    
    def _is_requirement_like(self, text: str) -> bool:
        """Check if text looks like a requirement"""
        if not isinstance(text, str) or len(text.strip()) < 15:
            return False
            
        requirement_keywords = [
            'shall', 'should', 'must', 'will', 'requirement', 'feature',
            'user story', 'acceptance criteria', 'given', 'when', 'then'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in requirement_keywords)
    
    def _analyze_json_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure for metadata"""
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys())[:10],  # First 10 keys
                "total_keys": len(data)
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "item_types": list(set(type(item).__name__ for item in data[:10]))
            }
        else:
            return {"type": type(data).__name__}

class YAMLParser(BaseEnhancedParser):
    """Parser for YAML files"""
    
    def parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Parse YAML file and extract requirements"""
        if not self.validate_file_size(file_path):
            return {"error": "File size exceeds limit", "raw_text": ""}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            all_text = yaml.dump(data, default_flow_style=False, indent=2)
            
            # Use JSON parser logic for YAML since structure is similar
            json_parser = JSONParser()
            requirements = json_parser._extract_requirements_from_json(data)
            
            return {
                "raw_text": all_text,
                "requirements": requirements,
                "metadata": {
                    "source_type": "yaml",
                    "structure": json_parser._analyze_json_structure(data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            return {"error": str(e), "raw_text": ""}

class HTMLParser(BaseEnhancedParser):
    """Parser for HTML files"""
    
    def parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Parse HTML file and extract text content"""
        if not WEB_SCRAPING_AVAILABLE:
            return {"error": "HTML parsing not available", "raw_text": ""}
            
        if not self.validate_file_size(file_path):
            return {"error": "File size exceeds limit", "raw_text": ""}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            all_text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract structured content
            requirements = []
            
            # Look for lists and headings that might contain requirements
            for i, elem in enumerate(soup.find_all(['li', 'p', 'div'], string=True)):
                text_content = elem.get_text().strip()
                if len(text_content) > 20 and self._is_requirement_like(text_content):
                    requirements.append({
                        "id": f"html_{elem.name}_{i}",
                        "text": text_content,
                        "source": f"HTML {elem.name} element",
                        "type": "requirement"
                    })
            
            return {
                "raw_text": all_text,
                "requirements": requirements,
                "metadata": {
                    "source_type": "html",
                    "title": soup.title.string if soup.title else "No title"
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing HTML file {file_path}: {e}")
            return {"error": str(e), "raw_text": ""}
    
    def _is_requirement_like(self, text: str) -> bool:
        """Check if text looks like a requirement"""
        if len(text.strip()) < 20:
            return False
            
        requirement_keywords = [
            'shall', 'should', 'must', 'will', 'requirement', 'feature',
            'user story', 'acceptance criteria', 'given', 'when', 'then'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in requirement_keywords)

class XMLParser(BaseEnhancedParser):
    """Parser for XML files"""
    
    def parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Parse XML file and extract content"""
        if not XML_AVAILABLE:
            return {"error": "XML parsing not available", "raw_text": ""}
            
        if not self.validate_file_size(file_path):
            return {"error": "File size exceeds limit", "raw_text": ""}
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            all_text = ""
            requirements = []
            
            # Extract all text content
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    all_text += f"{elem.tag}: {elem.text.strip()}\n"
                    
                    if self._is_requirement_like(elem.text.strip()):
                        requirements.append({
                            "id": f"xml_{elem.tag}_{len(requirements)}",
                            "text": elem.text.strip(),
                            "source": f"XML element: {elem.tag}",
                            "type": "requirement"
                        })
            
            return {
                "raw_text": all_text,
                "requirements": requirements,
                "metadata": {
                    "source_type": "xml",
                    "root_tag": root.tag,
                    "total_elements": len(list(root.iter()))
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {e}")
            return {"error": str(e), "raw_text": ""}
    
    def _is_requirement_like(self, text: str) -> bool:
        """Check if text looks like a requirement"""
        if len(text.strip()) < 15:
            return False
            
        requirement_keywords = [
            'shall', 'should', 'must', 'will', 'requirement', 'feature',
            'user story', 'acceptance criteria'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in requirement_keywords)

class WebScraper(BaseEnhancedParser):
    """Web scraper for extracting content from URLs"""
    
    def parse(self, url: str, **kwargs) -> Dict[str, Any]:
        """Scrape web page and extract content"""
        if not WEB_SCRAPING_AVAILABLE:
            return {"error": "Web scraping not available", "raw_text": ""}
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            all_text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract requirements
            requirements = []
            for i, elem in enumerate(soup.find_all(['li', 'p', 'div'], string=True)):
                text_content = elem.get_text().strip()
                if len(text_content) > 20 and self._is_requirement_like(text_content):
                    requirements.append({
                        "id": f"web_{i}",
                        "text": text_content,
                        "source": f"Web page: {url}",
                        "type": "requirement"
                    })
            
            return {
                "raw_text": all_text,
                "requirements": requirements,
                "metadata": {
                    "source_type": "web",
                    "url": url,
                    "title": soup.title.string if soup.title else "No title"
                }
            }
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            return {"error": str(e), "raw_text": ""}
    
    def _is_requirement_like(self, text: str) -> bool:
        """Check if text looks like a requirement"""
        if len(text.strip()) < 20:
            return False
            
        requirement_keywords = [
            'shall', 'should', 'must', 'will', 'requirement', 'feature',
            'user story', 'acceptance criteria', 'given', 'when', 'then'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in requirement_keywords)

class FigmaParser(BaseEnhancedParser):
    """Parser for Figma design files via API"""
    
    def parse(self, figma_url: str, access_token: str = None, **kwargs) -> Dict[str, Any]:
        """Extract design requirements from Figma file"""
        try:
            # Extract file ID from Figma URL
            file_id = self._extract_file_id(figma_url)
            if not file_id:
                return {"error": "Invalid Figma URL", "raw_text": ""}
            
            if not access_token:
                return {"error": "Figma access token required", "raw_text": ""}
            
            # Figma API endpoints
            api_base = "https://api.figma.com/v1"
            headers = {"X-Figma-Token": access_token}
            
            # Get file data
            file_response = requests.get(f"{api_base}/files/{file_id}", headers=headers)
            file_response.raise_for_status()
            file_data = file_response.json()
            
            # Extract text content and components
            all_text = ""
            requirements = []
            
            def extract_text_from_node(node, path=""):
                nonlocal all_text, requirements
                
                if 'characters' in node:
                    text = node['characters'].strip()
                    if text:
                        all_text += f"{path}: {text}\n"
                        if self._is_requirement_like(text):
                            requirements.append({
                                "id": f"figma_{node.get('id', 'unknown')}",
                                "text": text,
                                "source": f"Figma: {path}",
                                "type": "design_requirement"
                            })
                
                if 'children' in node:
                    for child in node['children']:
                        child_path = f"{path}/{child.get('name', 'unnamed')}"
                        extract_text_from_node(child, child_path)
            
            # Process document
            if 'document' in file_data:
                extract_text_from_node(file_data['document'], file_data['name'])
            
            return {
                "raw_text": all_text,
                "requirements": requirements,
                "metadata": {
                    "source_type": "figma",
                    "file_name": file_data.get('name', 'Unknown'),
                    "file_id": file_id,
                    "last_modified": file_data.get('lastModified', 'Unknown')
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing Figma file: {e}")
            return {"error": str(e), "raw_text": ""}
    
    def _extract_file_id(self, figma_url: str) -> Optional[str]:
        """Extract file ID from Figma URL"""
        try:
            # Figma URLs typically look like: https://www.figma.com/file/{file_id}/{file_name}
            parts = figma_url.split('/')
            if 'figma.com' in figma_url and 'file' in parts:
                file_index = parts.index('file')
                if file_index + 1 < len(parts):
                    return parts[file_index + 1]
            return None
        except:
            return None
    
    def _is_requirement_like(self, text: str) -> bool:
        """Check if text looks like a design requirement"""
        if len(text.strip()) < 10:
            return False
            
        design_keywords = [
            'user', 'button', 'click', 'display', 'show', 'hide', 'form',
            'input', 'dropdown', 'modal', 'popup', 'navigation', 'menu',
            'header', 'footer', 'sidebar', 'content', 'layout', 'responsive'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in design_keywords)

class ConfluenceParser(BaseEnhancedParser):
    """Parser for Confluence pages via API"""
    
    def parse(self, confluence_url: str, auth_token: str = None, **kwargs) -> Dict[str, Any]:
        """Extract content from Confluence page"""
        try:
            # Extract space and page info from URL
            page_info = self._extract_page_info(confluence_url)
            if not page_info:
                return {"error": "Invalid Confluence URL", "raw_text": ""}
            
            if not auth_token:
                return {"error": "Confluence authentication required", "raw_text": ""}
            
            # Confluence REST API
            base_url = page_info['base_url']
            headers = {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json"
            }
            
            # Get page content
            api_url = f"{base_url}/rest/api/content/{page_info['page_id']}?expand=body.storage,version"
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            page_data = response.json()
            
            # Extract content
            content_html = page_data['body']['storage']['value']
            soup = BeautifulSoup(content_html, 'html.parser')
            
            # Remove Confluence-specific elements
            for elem in soup(['ac:structured-macro', 'ac:parameter']):
                elem.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            all_text = '\n'.join(line for line in lines if line)
            
            # Extract requirements
            requirements = []
            for i, elem in enumerate(soup.find_all(['li', 'p', 'td'])):
                text_content = elem.get_text().strip()
                if len(text_content) > 15 and self._is_requirement_like(text_content):
                    requirements.append({
                        "id": f"confluence_{page_info['page_id']}_{i}",
                        "text": text_content,
                        "source": f"Confluence: {page_data.get('title', 'Unknown page')}",
                        "type": "requirement"
                    })
            
            return {
                "raw_text": all_text,
                "requirements": requirements,
                "metadata": {
                    "source_type": "confluence",
                    "page_title": page_data.get('title', 'Unknown'),
                    "page_id": page_info['page_id'],
                    "space": page_info.get('space', 'Unknown'),
                    "version": page_data.get('version', {}).get('number', 'Unknown')
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing Confluence page: {e}")
            return {"error": str(e), "raw_text": ""}
    
    def _extract_page_info(self, confluence_url: str) -> Optional[Dict[str, str]]:
        """Extract page information from Confluence URL"""
        try:
            # Confluence URLs: https://company.atlassian.net/wiki/spaces/SPACE/pages/PAGE_ID/Page+Title
            parsed = urlparse(confluence_url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            parts = parsed.path.split('/')
            if 'pages' in parts:
                page_index = parts.index('pages')
                if page_index + 1 < len(parts):
                    page_id = parts[page_index + 1]
                    space = None
                    if 'spaces' in parts:
                        space_index = parts.index('spaces')
                        if space_index + 1 < len(parts):
                            space = parts[space_index + 1]
                    
                    return {
                        'base_url': base_url,
                        'page_id': page_id,
                        'space': space
                    }
            return None
        except:
            return None
    
    def _is_requirement_like(self, text: str) -> bool:
        """Check if text looks like a requirement"""
        if len(text.strip()) < 15:
            return False
            
        requirement_keywords = [
            'shall', 'should', 'must', 'will', 'requirement', 'feature',
            'user story', 'acceptance criteria', 'given', 'when', 'then'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in requirement_keywords)

class EnhancedParserFactory:
    """Factory for creating enhanced parsers"""
    
    _parsers = {
        '.txt': TextParser,
        '.md': MarkdownParser,
        '.docx': WordParser,
        '.pdf': PDFParser,
        '.xlsx': ExcelParser,
        '.xls': ExcelParser,
        '.csv': CSVParser,
        '.json': JSONParser,
        '.yaml': YAMLParser,
        '.yml': YAMLParser,
        '.html': HTMLParser,
        '.htm': HTMLParser,
        '.xml': XMLParser,
        '.rtf': TextParser,  # RTF can be parsed as text for basic content
    }
    
    @classmethod
    def get_parser(cls, file_path: str) -> Optional[BaseEnhancedParser]:
        """Get appropriate parser for file"""
        ext = os.path.splitext(file_path)[1].lower()
        parser_class = cls._parsers.get(ext)
        
        if parser_class:
            try:
                return parser_class()
            except:
                # Fallback to text parser
                return TextParser() if TextParser else None
        
        return None
    
    @classmethod
    def get_web_parser(cls) -> WebScraper:
        """Get web scraper"""
        return WebScraper()
    
    @classmethod
    def get_figma_parser(cls) -> FigmaParser:
        """Get Figma parser"""
        return FigmaParser()
    
    @classmethod
    def get_confluence_parser(cls) -> ConfluenceParser:
        """Get Confluence parser"""
        return ConfluenceParser()

# Convenience function for backward compatibility
def get_enhanced_parser(file_path: str) -> Optional[BaseEnhancedParser]:
    """Get enhanced parser for file"""
    return EnhancedParserFactory.get_parser(file_path)
