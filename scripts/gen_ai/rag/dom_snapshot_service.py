"""
Smart DOM Snapshot Service for Enhanced Test Generation

This service captures comprehensive DOM snapshots and analyzes UI patterns
to generate more accurate and reliable automated test scripts.
"""

import json
import logging
import os
import time
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    webdriver = None  # type: ignore
    By = None  # type: ignore
    Options = None  # type: ignore
    WebDriverWait = None  # type: ignore
    EC = None  # type: ignore
    TimeoutException = Exception  # type: ignore
    NoSuchElementException = Exception  # type: ignore

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    COMPUTER_VISION_AVAILABLE = True
except ImportError:
    COMPUTER_VISION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ElementSnapshot:
    """Comprehensive snapshot of a UI element"""
    id: str
    tag_name: str
    text: str
    attributes: Dict[str, str]
    css_selectors: List[str]
    xpath_selectors: List[str]
    position: Dict[str, int]  # x, y, width, height
    screenshot_path: Optional[str]
    parent_context: Dict[str, Any]
    children_context: List[Dict[str, Any]]
    interaction_type: str  # button, input, link, etc.
    accessibility_info: Dict[str, Any]
    visual_features: Dict[str, Any]
    # New: additional alternate selectors (Playwright-first semantics)
    alt_selectors: List[str] = field(default_factory=list)

@dataclass
class PageSnapshot:
    """Complete snapshot of a web page"""
    url: str
    title: str
    timestamp: str
    elements: List[ElementSnapshot]
    page_structure: Dict[str, Any]
    forms: List[Dict[str, Any]]
    navigation_elements: List[Dict[str, Any]]
    interactive_elements: List[Dict[str, Any]]
    page_screenshot: Optional[str]
    performance_metrics: Dict[str, Any]
    accessibility_score: float

class SmartDOMAnalyzer:
    """Intelligent DOM analysis for test generation"""
    
    def __init__(self):
        self.element_patterns = {}
        self.interaction_patterns = {}
        self.selector_reliability_scores = {}
        
    def analyze_element_stability(self, element_history: List[ElementSnapshot]) -> Dict[str, Any]:
        """Analyze element stability across multiple snapshots"""
        stability_metrics = {
            'selector_stability': {},
            'position_stability': 0.0,
            'text_stability': 0.0,
            'attributes_stability': 0.0,
            'recommended_selectors': []
        }
        
        if len(element_history) < 2:
            return stability_metrics
            
        # Analyze selector stability
        for selector_type in ['css_selectors', 'xpath_selectors']:
            selectors = [getattr(elem, selector_type) for elem in element_history]
            common_selectors = set(selectors[0])
            for sel_list in selectors[1:]:
                common_selectors &= set(sel_list)
            
            stability_metrics['selector_stability'][selector_type] = {
                'stable_selectors': list(common_selectors),
                'stability_score': len(common_selectors) / len(selectors[0]) if selectors[0] else 0
            }
        
        # Analyze position stability
        positions = [elem.position for elem in element_history]
        position_variance = self._calculate_position_variance(positions)
        stability_metrics['position_stability'] = 1.0 - min(position_variance / 100, 1.0)
        
        # Generate recommended selectors based on stability
        stability_metrics['recommended_selectors'] = self._generate_stable_selectors(element_history)
        
        return stability_metrics
    
    def _calculate_position_variance(self, positions: List[Dict[str, int]]) -> float:
        """Calculate variance in element positions"""
        if len(positions) < 2:
            return 0.0
            
        coords = ['x', 'y', 'width', 'height']
        total_variance = 0
        
        for coord in coords:
            values = [pos.get(coord, 0) for pos in positions]
            if values:
                mean_val = sum(values) / len(values)
                variance = sum((val - mean_val) ** 2 for val in values) / len(values)
                total_variance += variance
                
        return total_variance / len(coords)
    
    def _generate_stable_selectors(self, element_history: List[ElementSnapshot]) -> List[Dict[str, Any]]:
        """Generate stable selectors based on element history"""
        stable_selectors = []
        
        # Prefer selectors that are consistent across snapshots
        if element_history:
            element = element_history[0]
            
            # ID selector (highest priority if stable)
            if element.attributes.get('id'):
                stable_selectors.append({
                    'type': 'id',
                    'selector': f"#{element.attributes['id']}",
                    'priority': 10,
                    'stability_score': 0.95
                })
            
            # Data attributes (high priority)
            for attr, value in element.attributes.items():
                if attr.startswith('data-testid') or attr.startswith('data-test'):
                    stable_selectors.append({
                        'type': 'attribute',
                        'selector': f"[{attr}='{value}']",
                        'priority': 9,
                        'stability_score': 0.9
                    })
            
            # ARIA labels (good for accessibility)
            if element.attributes.get('aria-label'):
                stable_selectors.append({
                    'type': 'aria',
                    'selector': f"[aria-label='{element.attributes['aria-label']}']",
                    'priority': 8,
                    'stability_score': 0.85
                })
        
        return sorted(stable_selectors, key=lambda x: x['priority'], reverse=True)

class DOMSnapshotService:
    """Service for capturing and analyzing DOM snapshots"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Defaults for speed and consistency
        self.config.setdefault('headless', True)
        self.config.setdefault('viewport', {'width': 1920, 'height': 1080})
        self.config.setdefault('incognito', True)
        self.config.setdefault('light_mode', True)  # skip heavy operations when True
        self.config.setdefault('wait_after_load_ms', 500)
        self.config.setdefault('page_load_state', 'domcontentloaded')
        # New: reduce UI motion (disable CSS animations/transitions) to minimize visual flicker
        self.config.setdefault('reduce_motion', True)
        self.driver = None
        self.playwright_browser = None
        self.playwright_context = None
        self.playwright_page = None
        self.analyzer = SmartDOMAnalyzer()
        self.snapshot_cache = {}
        
    async def initialize_playwright(self) -> bool:
        """Initialize Playwright browser"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available")
            return False
            
        try:
            from playwright.async_api import async_playwright
            self.playwright = await async_playwright().start()
            # Fast, deterministic launch
            viewport = self.config.get('viewport', {'width': 1920, 'height': 1080})
            vw, vh = int(viewport.get('width', 1920)), int(viewport.get('height', 1080))
            args = [
                "--disable-extensions",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                # Keep a fixed OS window size to avoid viewport oscillations
                f"--window-size={vw},{vh}",
            ]
            # Incognito hint (contexts are isolated but flag is harmless)
            if self.config.get('incognito', True):
                args.append("--incognito")
            self.playwright_browser = await self.playwright.chromium.launch(
                headless=self.config.get('headless', True),
                args=args
            )
            # Context config: in headful, disable Playwright viewport to match OS window exactly
            if self.config.get('headless', True):
                self.playwright_context = await self.playwright_browser.new_context(
                    viewport={'width': vw, 'height': vh},
                    device_scale_factor=1,
                    color_scheme='light',
                    java_script_enabled=True,
                )
            else:
                self.playwright_context = await self.playwright_browser.new_context(
                    viewport=None,
                    device_scale_factor=1,
                    color_scheme='light',
                    java_script_enabled=True,
                )
            self.playwright_page = await self.playwright_context.new_page()
            try:
                if not self.config.get('headless', True):
                    await self.playwright_page.bring_to_front()
                # Inject CSS to reduce motion if enabled
                if self.config.get('reduce_motion', True):
                    try:
                        await self.playwright_page.add_style_tag(content=(
                            "*{transition:none !important; animation:none !important;} html{scroll-behavior:auto !important;}"
                        ))
                    except Exception:
                        pass
            except Exception:
                pass
            logger.info("Playwright initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            return False
    
    def initialize_selenium(self) -> bool:
        """Initialize Selenium WebDriver"""
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available")
            return False
            
        try:
            options = Options()
            viewport = self.config.get('viewport', {'width': 1920, 'height': 1080})
            if self.config.get('headless', True):
                options.add_argument('--headless=new')  # Use new headless mode
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument(f"--window-size={int(viewport['width'])},{int(viewport['height'])}")
            # Open in incognito for clean sessions
            if self.config.get('incognito', True):
                options.add_argument('--incognito')

            # Try to use webdriver-manager for automatic ChromeDriver management
            try:
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
                logger.info("Selenium WebDriver initialized with webdriver-manager")
            except Exception as wdm_error:
                logger.warning(f"webdriver-manager failed: {wdm_error}, trying fallback...")
                self.driver = webdriver.Chrome(options=options)
                logger.info("Selenium WebDriver initialized with fallback method")

            try:
                # Force top-left origin and fixed viewport; avoid additional maximize to prevent flicker
                self.driver.set_window_position(0, 0)
                if not self.config.get('headless', True):
                    self.driver.set_window_rect(x=0, y=0, width=int(viewport['width']), height=int(viewport['height']))
                    # Do not call maximize_window after setting explicit rect
            except Exception:
                pass
            logger.info("Selenium WebDriver initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            return False
    
    async def capture_page_snapshot(self, url: str, wait_for_load: bool = True) -> PageSnapshot:
        """Capture comprehensive snapshot of a web page"""
        logger.info(f"Capturing snapshot for {url}")
        
        # Use Playwright if available, fallback to Selenium
        if self.playwright_page:
            return await self._capture_with_playwright(url, wait_for_load)
        elif self.driver:
            return self._capture_with_selenium(url, wait_for_load)
        else:
            raise RuntimeError("No browser automation tool available")

    async def capture_current_page_snapshot(self) -> PageSnapshot:
        """Capture snapshot of the currently loaded page without navigation"""
        if self.playwright_page:
            # Best effort settle
            try:
                await self.playwright_page.wait_for_load_state(self.config.get('page_load_state', 'domcontentloaded'))
                await self.playwright_page.wait_for_timeout(int(self.config.get('wait_after_load_ms', 300)))
            except Exception:
                pass
            title = await self.playwright_page.title()
            url = self.playwright_page.url
            screenshot_path = None
            if not self.config.get('light_mode', True):
                screenshot_path = self._get_screenshot_path(url)
                try:
                    await self.playwright_page.screenshot(path=screenshot_path, full_page=True)
                except Exception:
                    screenshot_path = None
            elements = await self._extract_elements_playwright()
            page_structure = await self._analyze_page_structure_playwright()
            forms = await self._extract_forms_playwright()
            performance_metrics = {}
            if not self.config.get('light_mode', True):
                try:
                    performance_metrics = await self._get_performance_metrics_playwright()
                except Exception:
                    performance_metrics = {}
            return PageSnapshot(
                url=url,
                title=title,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                elements=elements,
                page_structure=page_structure,
                forms=forms,
                navigation_elements=await self._extract_navigation_elements_playwright(),
                interactive_elements=await self._extract_interactive_elements_playwright(),
                page_screenshot=screenshot_path,
                performance_metrics=performance_metrics,
                accessibility_score=await self._calculate_accessibility_score_playwright()
            )
        elif self.driver:
            # Best effort settle
            try:
                WebDriverWait(self.driver, 3).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            except Exception:
                pass
            time.sleep(self.config.get('wait_after_load_ms', 300) / 1000.0)
            url = self.driver.current_url
            title = self.driver.title
            screenshot_path = None
            if not self.config.get('light_mode', True):
                screenshot_path = self._get_screenshot_path(url)
                try:
                    self.driver.save_screenshot(screenshot_path)
                except Exception:
                    screenshot_path = None
            elements = self._extract_elements_selenium()
            page_structure = self._analyze_page_structure_selenium()
            forms = self._extract_forms_selenium()
            return PageSnapshot(
                url=url,
                title=title,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                elements=elements,
                page_structure=page_structure,
                forms=forms,
                navigation_elements=self._extract_navigation_elements_selenium(),
                interactive_elements=self._extract_interactive_elements_selenium(),
                page_screenshot=screenshot_path,
                performance_metrics=self._get_performance_metrics_selenium() if not self.config.get('light_mode', True) else {},
                accessibility_score=self._calculate_accessibility_score_selenium()
            )
        else:
            raise RuntimeError("No browser automation tool available")

    async def _capture_with_playwright(self, url: str, wait_for_load: bool) -> PageSnapshot:
        """Capture snapshot using Playwright"""
        await self.playwright_page.goto(
            url,
            wait_until=self.config.get('page_load_state', 'domcontentloaded') if wait_for_load else 'domcontentloaded'
        )

        # Wait for dynamic content
        if wait_for_load:
            await self.playwright_page.wait_for_timeout(self.config.get('wait_after_load_ms', 500))

        # Get page info
        title = await self.playwright_page.title()
        
        # Capture page screenshot (skip in light mode)
        screenshot_path = None
        if not self.config.get('light_mode', True):
            screenshot_path = self._get_screenshot_path(url)
            try:
                await self.playwright_page.screenshot(path=screenshot_path, full_page=True)
            except Exception as e:
                logger.debug(f"Screenshot skipped: {e}")

        # Get all interactive elements
        elements = await self._extract_elements_playwright()
        
        # Analyze page structure
        page_structure = await self._analyze_page_structure_playwright()
        
        # Extract forms
        forms = await self._extract_forms_playwright()
        
        # Get performance metrics (skip in light mode)
        performance_metrics = {}
        if not self.config.get('light_mode', True):
            performance_metrics = await self._get_performance_metrics_playwright()

        return PageSnapshot(
            url=url,
            title=title,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            elements=elements,
            page_structure=page_structure,
            forms=forms,
            navigation_elements=await self._extract_navigation_elements_playwright(),
            interactive_elements=await self._extract_interactive_elements_playwright(),
            page_screenshot=screenshot_path,
            performance_metrics=performance_metrics,
            accessibility_score=await self._calculate_accessibility_score_playwright()
        )
    
    def _capture_with_selenium(self, url: str, wait_for_load: bool) -> PageSnapshot:
        """Capture snapshot using Selenium"""
        self.driver.get(url)
        
        if wait_for_load:
            try:
                WebDriverWait(self.driver, 6).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except Exception:
                pass
            time.sleep(self.config.get('wait_after_load_ms', 500) / 1000.0)  # Short wait for dynamic content

        # Get page info
        title = self.driver.title
        
        # Capture page screenshot (skip in light mode)
        screenshot_path = None
        if not self.config.get('light_mode', True):
            screenshot_path = self._get_screenshot_path(url)
            try:
                self.driver.save_screenshot(screenshot_path)
            except Exception:
                screenshot_path = None
        elements = self._extract_elements_selenium()
        page_structure = self._analyze_page_structure_selenium()
        forms = self._extract_forms_selenium()
        return PageSnapshot(
            url=url,
            title=title,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            elements=elements,
            page_structure=page_structure,
            forms=forms,
            navigation_elements=self._extract_navigation_elements_selenium(),
            interactive_elements=self._extract_interactive_elements_selenium(),
            page_screenshot=screenshot_path,
            performance_metrics=self._get_performance_metrics_selenium() if not self.config.get('light_mode', True) else {},
            accessibility_score=self._calculate_accessibility_score_selenium()
        )
    
    async def _extract_elements_playwright(self) -> List[ElementSnapshot]:
        """Extract all interactive elements using Playwright"""
        elements = []
        
        # Define selectors for interactive elements
        interactive_selectors = [
            'button', 'input', 'select', 'textarea', 'a[href]',
            '[onclick]', '[role="button"]', '[role="link"]',
            '[data-testid]', '[data-test]', '.btn', '.button'
        ]
        
        for selector in interactive_selectors:
            try:
                page_elements = await self.playwright_page.query_selector_all(selector)
                for element in page_elements:
                    element_data = await self._extract_element_data_playwright(element)
                    if element_data:
                        elements.append(element_data)
            except Exception as e:
                logger.warning(f"Error extracting elements with selector {selector}: {e}")
        
        return elements
    
    def _extract_elements_selenium(self) -> List[ElementSnapshot]:
        """Extract all interactive elements using Selenium"""
        elements = []
        
        # Define selectors for interactive elements
        interactive_selectors = [
            (By.TAG_NAME, 'button'),
            (By.TAG_NAME, 'input'),
            (By.TAG_NAME, 'select'),
            (By.TAG_NAME, 'textarea'),
            (By.CSS_SELECTOR, 'a[href]'),
            (By.CSS_SELECTOR, '[onclick]'),
            (By.CSS_SELECTOR, '[role="button"]'),
            (By.CSS_SELECTOR, '[data-testid]'),
            (By.CSS_SELECTOR, '.btn'),
            (By.CSS_SELECTOR, '.button')
        ]
        
        for by, selector in interactive_selectors:
            try:
                page_elements = self.driver.find_elements(by, selector)
                for element in page_elements:
                    element_data = self._extract_element_data_selenium(element)
                    if element_data:
                        elements.append(element_data)
            except Exception as e:
                logger.warning(f"Error extracting elements with selector {selector}: {e}")
        
        return elements
    
    async def _extract_element_data_playwright(self, element) -> Optional[ElementSnapshot]:
        """Extract comprehensive data from a Playwright element"""
        try:
            # Get basic element info
            tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
            text = (await element.text_content() or '').strip()
            
            # Get all attributes
            attributes = await element.evaluate('''
                el => {
                    const attrs = {};
                    for (let attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    return attrs;
                }
            ''')
            
            # Derive accessible label if present
            try:
                label_text = await element.evaluate(r'''
                    el => {
                        function byRef(id){ const n=document.getElementById(id); return n? n.innerText || n.textContent || '' : ''; }
                        let label = '';
                        if (el.labels && el.labels.length) { label = Array.from(el.labels).map(l => l.innerText || l.textContent || '').join(' ').trim(); }
                        if (!label && el.getAttribute('aria-label')) { label = el.getAttribute('aria-label'); }
                        if (!label && el.getAttribute('aria-labelledby')) { label = el.getAttribute('aria-labelledby').split(/\s+/).map(byRef).join(' ').trim(); }
                        return label;
                    }
                ''')
            except Exception:
                label_text = ''

            # Get element position
            bounding_box = await element.bounding_box()
            position = {
                'x': int(bounding_box['x']) if bounding_box else 0,
                'y': int(bounding_box['y']) if bounding_box else 0,
                'width': int(bounding_box['width']) if bounding_box else 0,
                'height': int(bounding_box['height']) if bounding_box else 0
            }
            
            # Generate selectors
            css_selectors = await self._generate_css_selectors_playwright(element)
            xpath_selectors = await self._generate_xpath_selectors_playwright(element)
            
            # Get accessibility info
            accessibility_info = await self._get_accessibility_info_playwright(element)
            
            # Determine interaction type
            interaction_type = self._determine_interaction_type(tag_name, attributes)
            
            # New: Alternate selectors tailored for Playwright's smart locators
            alt_selectors: List[str] = []
            if attributes.get('data-testid'):
                alt_selectors.append(f"testid={attributes['data-testid']}")
            if attributes.get('data-test'):
                alt_selectors.append(f"testid={attributes['data-test']}")
            if attributes.get('placeholder'):
                alt_selectors.append(f"placeholder={attributes['placeholder']}")
            if label_text:
                alt_selectors.append(f"label={label_text}")
            # Also surface common data attributes directly for CSS selection
            if attributes.get('data-element-label'):
                alt_selectors.append(f"[data-element-label='{attributes['data-element-label']}']")
            # Role-based selector using tag/role and visible name
            role = attributes.get('role')
            name_for_role = label_text or text
            if (tag_name in ['button', 'a'] or role == 'button') and name_for_role:
                alt_selectors.append(f"role=button[name='{name_for_role.strip()}']")
            if tag_name == 'a' and (text or attributes.get('title')):
                alt_selectors.append(f"role=link[name='{(text or attributes.get('title')).strip()}']")

            element_id = hashlib.md5(f"{css_selectors[0] if css_selectors else tag_name}_{position}".encode()).hexdigest()
            
            return ElementSnapshot(
                id=element_id,
                tag_name=tag_name,
                text=text,
                attributes=attributes,
                css_selectors=css_selectors,
                xpath_selectors=xpath_selectors,
                position=position,
                screenshot_path=None,  # Could capture individual element screenshots
                parent_context={},  # Could analyze parent elements
                children_context=[],  # Could analyze child elements
                interaction_type=interaction_type,
                accessibility_info=accessibility_info,
                visual_features={},  # Could add visual analysis
                alt_selectors=alt_selectors
            )
            
        except Exception as e:
            logger.warning(f"Error extracting element data: {e}")
            return None
    
    def _extract_element_data_selenium(self, element) -> Optional[ElementSnapshot]:
        """Extract comprehensive data from a Selenium element"""
        try:
            # Get basic element info
            tag_name = element.tag_name.lower()
            text = element.text.strip()
            
            # Get all attributes
            attributes = {}
            for attr in ['id', 'class', 'name', 'type', 'value', 'href', 'src', 'alt', 'title', 'aria-label', 'data-testid', 'data-test', 'placeholder', 'role']:
                try:
                    value = element.get_attribute(attr)
                    if value:
                        attributes[attr] = value
                except:
                    pass
            
            # Get element position
            location = element.location
            size = element.size
            position = {
                'x': location['x'],
                'y': location['y'],
                'width': size['width'],
                'height': size['height']
            }
            
            # Generate selectors
            css_selectors = self._generate_css_selectors_selenium(element)
            xpath_selectors = self._generate_xpath_selectors_selenium(element)
            
            # Determine interaction type
            interaction_type = self._determine_interaction_type(tag_name, attributes)
            
            # Alt selectors hints (non-CSS) for higher-level consumers
            alt_selectors: List[str] = []
            if attributes.get('data-testid'):
                alt_selectors.append(f"testid={attributes['data-testid']}")
            if attributes.get('data-test'):
                alt_selectors.append(f"testid={attributes['data-test']}")
            if attributes.get('placeholder'):
                alt_selectors.append(f"placeholder={attributes['placeholder']}")
            if attributes.get('aria-label'):
                alt_selectors.append(f"label={attributes['aria-label']}")
            if (tag_name == 'button' or attributes.get('type') == 'submit') and (text or attributes.get('aria-label')):
                alt_selectors.append(f"role=button[name='{(text or attributes.get('aria-label')).strip()}']")
            if tag_name == 'a' and (text or attributes.get('title')):
                alt_selectors.append(f"role=link[name='{(text or attributes.get('title')).strip()}']")

            element_id = hashlib.md5(f"{css_selectors[0] if css_selectors else tag_name}_{position}".encode()).hexdigest()
            
            return ElementSnapshot(
                id=element_id,
                tag_name=tag_name,
                text=text,
                attributes=attributes,
                css_selectors=css_selectors,
                xpath_selectors=xpath_selectors,
                position=position,
                screenshot_path=None,
                parent_context={},
                children_context=[],
                interaction_type=interaction_type,
                accessibility_info={},
                visual_features={},
                alt_selectors=alt_selectors
            )
            
        except Exception as e:
            logger.warning(f"Error extracting element data: {e}")
            return None
    
    def _determine_interaction_type(self, tag_name: str, attributes: Dict[str, str]) -> str:
        """Determine the type of interaction for an element"""
        if tag_name == 'button' or attributes.get('type') == 'submit':
            return 'button'
        elif tag_name == 'input':
            input_type = attributes.get('type', 'text')
            return f'input_{input_type}'
        elif tag_name == 'select':
            return 'dropdown'
        elif tag_name == 'textarea':
            return 'textarea'
        elif tag_name == 'a' and attributes.get('href'):
            return 'link'
        elif 'onclick' in attributes or 'role' in attributes:
            return 'clickable'
        else:
            return 'unknown'
    
    def _generate_css_selectors_selenium(self, element) -> List[str]:
        """Generate multiple CSS selectors for an element"""
        selectors: List[str] = []

        # ID selector
        element_id = element.get_attribute('id')
        if element_id:
            selectors.append(f"#{element_id}")
        
        # Class selectors
        class_name = element.get_attribute('class')
        if class_name:
            classes = class_name.split()
            if len(classes) == 1:
                selectors.append(f".{classes[0]}")
            elif len(classes) > 1:
                selectors.append(f".{'.'.join(classes)}")
        
        # Data attribute selectors
        for attr in ['data-testid', 'data-test', 'name', 'placeholder', 'aria-label', 'data-element-label']:
            value = element.get_attribute(attr)
            if value:
                selectors.append(f"[{attr}='{value}']")
        
        # Tag + attribute combinations
        tag_name = element.tag_name.lower()
        element_type = element.get_attribute('type')
        if element_type:
            selectors.append(f"{tag_name}[type='{element_type}']")
        
        # DO NOT add :contains() as it's not supported by Selenium CSS
        return selectors
    
    def _generate_xpath_selectors_selenium(self, element) -> List[str]:
        """Generate multiple XPath selectors for an element"""
        selectors: List[str] = []

        # Try to get XPath using JavaScript
        try:
            xpath = self.driver.execute_script("""
                function getXPath(element) {
                    if (element.id !== '') {
                        return '//*[@id="' + element.id + '"]';
                    }
                    if (element === document.body) {
                        return '/html/body';
                    }
                    
                    var ix = 0;
                    var siblings = element.parentNode.childNodes;
                    for (var i = 0; i < siblings.length; i++) {
                        var sibling = siblings[i];
                        if (sibling === element) {
                            return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                        }
                        if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                            ix++;
                        }
                    }
                }
                return getXPath(arguments[0]);
            """, element)
            if xpath:
                selectors.append(xpath)
        except:
            pass
        
        # Generate attribute-based XPath
        tag_name = element.tag_name.lower()
        
        # ID-based XPath
        element_id = element.get_attribute('id')
        if element_id:
            selectors.append(f"//{tag_name}[@id='{element_id}']")
        
        # Text-based XPath
        text = element.text.strip()
        if text:
            selectors.append(f"//{tag_name}[contains(normalize-space(.), '{text}')]")

        # Attribute-based XPath
        for attr in ['data-testid', 'data-test', 'name', 'class', 'placeholder', 'aria-label', 'data-element-label']:
            value = element.get_attribute(attr)
            if value:
                selectors.append(f"//{tag_name}[@{attr}='{value}']")
        
        return selectors
    
    async def _generate_css_selectors_playwright(self, element) -> List[str]:
        """Generate CSS selectors for Playwright element"""
        # Implementation similar to Selenium version but using Playwright API
        selectors: List[str] = []

        # Get element attributes
        attributes = await element.evaluate('''
            el => {
                const attrs = {};
                for (let attr of el.attributes) {
                    attrs[attr.name] = attr.value;
                }
                return attrs;
            }
        ''')
        
        tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
        
        # ID selector
        if attributes.get('id'):
            selectors.append(f"#{attributes['id']}")
        
        # Class selectors
        if attributes.get('class'):
            classes = attributes['class'].split()
            if len(classes) == 1:
                selectors.append(f".{classes[0]}")
            elif len(classes) > 1:
                selectors.append(f".{'.'.join(classes)}")
        
        # Data attribute selectors
        for attr in ['data-testid', 'data-test', 'name', 'placeholder', 'aria-label', 'data-element-label']:
            if attributes.get(attr):
                selectors.append(f"[{attr}='{attributes[attr]}']")
        
        return selectors
    
    async def _generate_xpath_selectors_playwright(self, element) -> List[str]:
        """Generate XPath selectors for Playwright element"""
        # Implementation similar to Selenium version but using Playwright API
        selectors: List[str] = []

        # Basic XPath generation using Playwright's evaluate
        try:
            xpath = await element.evaluate('''
                el => {
                    function getXPath(element) {
                        if (element.id !== '') {
                            return '//*[@id="' + element.id + '"]';
                        }
                        if (element === document.body) {
                            return '/html/body';
                        }
                        
                        var ix = 0;
                        var siblings = element.parentNode.childNodes;
                        for (var i = 0; i < siblings.length; i++) {
                            var sibling = siblings[i];
                            if (sibling === element) {
                                return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                            }
                            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                                ix++;
                            }
                        }
                    }
                    return getXPath(el);
                }
            ''')
            if xpath:
                selectors.append(xpath)
        except:
            pass
        
        return selectors
    
    def _get_screenshot_path(self, url: str) -> str:
        """Generate screenshot path for a URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        timestamp = int(time.time())
        screenshot_dir = self.config.get('screenshot_dir', 'dom_snapshots/screenshots')
        os.makedirs(screenshot_dir, exist_ok=True)
        return os.path.join(screenshot_dir, f"page_{url_hash}_{timestamp}.png")
    
    def save_snapshot(self, snapshot: PageSnapshot, output_dir: str = "dom_snapshots") -> str:
        """Save snapshot to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename based on URL and timestamp
        url_hash = hashlib.md5(snapshot.url.encode()).hexdigest()
        filename = f"snapshot_{url_hash}_{snapshot.timestamp.replace(':', '-').replace(' ', '_')}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert snapshot to dict for JSON serialization
        snapshot_dict = asdict(snapshot)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(snapshot_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Snapshot saved to {filepath}")
        return filepath
    
    def load_snapshot(self, filepath: str) -> PageSnapshot:
        """Load snapshot from disk"""
        with open(filepath, 'r', encoding='utf-8') as f:
            snapshot_dict = json.load(f)
        
        # Convert dict back to PageSnapshot
        elements = [ElementSnapshot(**elem) for elem in snapshot_dict['elements']]
        snapshot_dict['elements'] = elements
        
        return PageSnapshot(**snapshot_dict)
    
    async def cleanup(self):
        """Cleanup browser instances"""
        if self.playwright_browser:
            await self.playwright_browser.close()
        if self.driver:
            self.driver.quit()
    
    # Placeholder methods for additional functionality
    async def _analyze_page_structure_playwright(self) -> Dict[str, Any]:
        """Analyze page structure using Playwright"""
        return {"structure": "analyzed"}
    
    def _analyze_page_structure_selenium(self) -> Dict[str, Any]:
        """Analyze page structure using Selenium"""
        return {"structure": "analyzed"}
    
    async def _extract_forms_playwright(self) -> List[Dict[str, Any]]:
        """Extract form information using Playwright"""
        return []
    
    def _extract_forms_selenium(self) -> List[Dict[str, Any]]:
        """Extract form information using Selenium"""
        return []
    
    async def _extract_navigation_elements_playwright(self) -> List[Dict[str, Any]]:
        """Extract navigation elements using Playwright"""
        return []
    
    def _extract_navigation_elements_selenium(self) -> List[Dict[str, Any]]:
        """Extract navigation elements using Selenium"""
        return []
    
    async def _extract_interactive_elements_playwright(self) -> List[Dict[str, Any]]:
        """Extract interactive elements using Playwright"""
        return []
    
    def _extract_interactive_elements_selenium(self) -> List[Dict[str, Any]]:
        """Extract interactive elements using Selenium"""
        return []
    
    async def _get_performance_metrics_playwright(self) -> Dict[str, Any]:
        """Get performance metrics using Playwright"""
        return {}
    
    def _get_performance_metrics_selenium(self) -> Dict[str, Any]:
        """Get performance metrics using Selenium"""
        return {}
    
    async def _calculate_accessibility_score_playwright(self) -> float:
        """Calculate accessibility score using Playwright"""
        return 0.0
    
    def _calculate_accessibility_score_selenium(self) -> float:
        """Calculate accessibility score using Selenium"""
        return 0.0
    
    async def _get_accessibility_info_playwright(self, element) -> Dict[str, Any]:
        """Get accessibility information for an element"""
        return {}

