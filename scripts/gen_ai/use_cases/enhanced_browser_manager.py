"""Enhanced BrowserAutomationManager wrapper providing per-step locator capture
and accessibility/security heuristic analysis without editing large original file."""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import re
import logging

logger = logging.getLogger("EnhancedBrowserManager")

try:
    from .test_pilot import BrowserAutomationManager, TestStep, TestCase  # relative import
except Exception:
    from test_pilot import BrowserAutomationManager, TestStep, TestCase  # fallback

class EnhancedBrowserAutomationManager(BrowserAutomationManager):
    def __init__(self, azure_client=None):
        super().__init__(azure_client)
        # augment original instance with new collections if missing
        if not hasattr(self, 'step_locators'):
            self.step_locators: Dict[int, Dict[str, str]] = {}
        if not hasattr(self, 'accessibility_findings'):
            self.accessibility_findings: List[Dict[str, Any]] = []
        if not hasattr(self, 'security_findings'):
            self.security_findings: List[Dict[str, Any]] = []
        if 'security_issues' not in self.bug_report:
            self.bug_report['security_issues'] = []

        # Track actual element used per step
        self.actual_elements_used: Dict[int, Dict[str, Any]] = {}

        logger.info("🔧 EnhancedBrowserAutomationManager initialized with logging")

    def initialize_browser(self, base_url: str, headless: bool = False) -> bool:
        """
        Enhanced browser initialization with better error handling and network diagnostics

        Args:
            base_url: Base URL to start from
            headless: Run in headless mode

        Returns:
            Success status
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            import socket

            logger.info(f"🚀 Initializing browser for automation: {base_url}")

            # First, test network connectivity
            logger.info("   🔍 Testing network connectivity...")
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(base_url)
                hostname = parsed_url.hostname or parsed_url.netloc
                logger.info(f"   🌐 Resolving hostname: {hostname}")
                ip_address = socket.gethostbyname(hostname)
                logger.info(f"   ✅ DNS resolution successful: {hostname} -> {ip_address}")
            except socket.gaierror as e:
                logger.error(f"   ❌ DNS resolution failed for {hostname}: {e}")
                logger.error("   💡 Possible fixes:")
                logger.error("      1. Check your internet connection")
                logger.error("      2. Check if the URL is correct")
                logger.error("      3. Try using a different DNS server")
                logger.error("      4. Check your network proxy settings")
                return False
            except Exception as e:
                logger.warning(f"   ⚠️  DNS check failed: {e}")

            # Setup Chrome with advanced logging and network fixes
            chrome_options = Options()

            # Basic Chrome options
            if headless:
                chrome_options.add_argument('--headless=new')  # Use new headless mode
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')

            # Network-related fixes
            chrome_options.add_argument('--disable-web-security')  # For CORS issues
            chrome_options.add_argument('--disable-features=VizDisplayCompositor')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--ignore-certificate-errors')
            chrome_options.add_argument('--allow-insecure-localhost')

            # Proxy bypass (if needed)
            chrome_options.add_argument('--no-proxy-server')
            chrome_options.add_argument('--proxy-bypass-list=*')

            # User agent
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

            # Enable performance and network logging
            chrome_options.set_capability('goog:loggingPrefs', {
                'performance': 'ALL',
                'browser': 'ALL'
            })

            # Enable Chrome DevTools Protocol
            chrome_options.add_experimental_option('perfLoggingPrefs', {
                'enableNetwork': True,
                'enablePage': True,
            })

            # Exclude automation flags
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
            chrome_options.add_experimental_option('useAutomationExtension', False)

            logger.info("   🔧 Starting Chrome browser...")

            # Initialize driver with explicit service (better error messages)
            try:
                service = Service()
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                logger.info("   ✅ Chrome browser started successfully")
            except Exception as e:
                logger.error(f"   ❌ Failed to start Chrome: {e}")
                # Try without service specification
                logger.info("   🔄 Retrying without explicit service...")
                self.driver = webdriver.Chrome(options=chrome_options)
                logger.info("   ✅ Chrome browser started (fallback method)")

            self.driver.set_page_load_timeout(60)  # Increased timeout
            self.driver.set_script_timeout(30)

            # Navigate to base URL with retry logic
            logger.info(f"   🌐 Navigating to: {base_url}")
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"   🔄 Attempt {attempt}/{max_retries}...")
                    self.driver.get(base_url)
                    logger.info(f"   ✅ Page loaded successfully")
                    break
                except Exception as e:
                    logger.error(f"   ❌ Attempt {attempt} failed: {e}")
                    if attempt == max_retries:
                        logger.error(f"   ❌ All {max_retries} attempts failed")
                        if self.driver:
                            self.driver.quit()
                        return False
                    logger.info(f"   ⏳ Waiting before retry...")
                    import time
                    time.sleep(2)

            # Wait for page load
            logger.info("   ⏳ Waiting for page to be fully loaded...")
            self._wait_for_page_load()

            # Verify page loaded
            current_url = self.driver.current_url
            page_title = self.driver.title
            logger.info(f"   ✅ Page loaded: {page_title}")
            logger.info(f"   🔗 Current URL: {current_url}")

            logger.info("✅ Browser initialized successfully")
            return True

        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            logger.error("   💡 Install Selenium: pip install selenium")
            return False
        except Exception as e:
            logger.error(f"❌ Browser initialization failed: {str(e)}")
            logger.error("   💡 Debug information:")
            logger.error(f"      - Base URL: {base_url}")
            logger.error(f"      - Headless mode: {headless}")
            logger.error(f"      - Error type: {type(e).__name__}")

            # Cleanup on failure
            if hasattr(self, 'driver') and self.driver:
                try:
                    self.driver.quit()
                    logger.info("   🧹 Browser cleaned up after failure")
                except:
                    pass

            return False

    def _capture_element_locator(self, element, step_number: int, action_type: str) -> Dict[str, str]:
        """
        Capture the ACTUAL locator of an element that was successfully interacted with.

        Args:
            element: Selenium WebElement that was clicked/used
            step_number: Step number for tracking
            action_type: Type of action (click, input, etc.)

        Returns:
            Dict with locator information
        """
        try:
            locator_info = {}

            # Priority 1: ID (best)
            elem_id = element.get_attribute('id')
            if elem_id and elem_id.strip():
                locator_info['id'] = f"id:{elem_id}"
                locator_info['name'] = f"{elem_id}_locator"
                locator_info['priority'] = 1
                logger.info(f"   📍 Step {step_number}: Captured ID locator: {locator_info['id']}")
                return locator_info

            # Priority 2: Name (for inputs)
            elem_name = element.get_attribute('name')
            if elem_name and elem_name.strip():
                locator_info['name_attr'] = f"name:{elem_name}"
                locator_info['name'] = f"{elem_name}_locator"
                locator_info['priority'] = 2
                logger.info(f"   📍 Step {step_number}: Captured NAME locator: {locator_info['name_attr']}")
                return locator_info

            # Priority 3: data-* attributes
            for attr in ['data-testid', 'data-test', 'data-qa', 'data-cy']:
                data_attr = element.get_attribute(attr)
                if data_attr and data_attr.strip():
                    locator_info['data_attr'] = f"css:[{attr}='{data_attr}']"
                    locator_info['name'] = f"{data_attr}_locator"
                    locator_info['priority'] = 3
                    logger.info(f"   📍 Step {step_number}: Captured DATA-* locator: {locator_info['data_attr']}")
                    return locator_info

            # Priority 4: Unique text content (for links/buttons)
            elem_text = element.text.strip()
            tag_name = element.tag_name.lower()

            if elem_text and len(elem_text) < 50 and tag_name in ['a', 'button', 'span']:
                if tag_name == 'a':
                    locator_info['text'] = f"link:{elem_text}"
                    locator_info['name'] = f"{elem_text.lower().replace(' ', '_')[:30]}_link_locator"
                else:
                    # Build more precise XPath
                    locator_info['text'] = f"xpath://{tag_name}[normalize-space(text())='{elem_text}']"
                    locator_info['name'] = f"{elem_text.lower().replace(' ', '_')[:30]}_button_locator"
                locator_info['priority'] = 4
                logger.info(f"   📍 Step {step_number}: Captured TEXT locator: {locator_info['text']}")
                return locator_info

            # Priority 5: aria-label
            aria_label = element.get_attribute('aria-label')
            if aria_label and aria_label.strip():
                locator_info['aria'] = f"css:[aria-label='{aria_label}']"
                locator_info['name'] = f"{aria_label.lower().replace(' ', '_')[:30]}_aria_locator"
                locator_info['priority'] = 5
                logger.info(f"   📍 Step {step_number}: Captured ARIA locator: {locator_info['aria']}")
                return locator_info

            # Priority 6: Class-based (less reliable but better than nothing)
            elem_class = element.get_attribute('class')
            if elem_class:
                classes = elem_class.split()
                # Filter out common utility classes
                unique_classes = [c for c in classes if not any(util in c.lower() for util in ['col-', 'row-', 'btn-', 'text-', 'bg-', 'p-', 'm-', 'w-', 'h-'])]
                if unique_classes:
                    locator_info['class'] = f"css:.{unique_classes[0]}"
                    locator_info['name'] = f"{unique_classes[0]}_class_locator"
                    locator_info['priority'] = 6
                    logger.info(f"   📍 Step {step_number}: Captured CLASS locator: {locator_info['class']}")
                    return locator_info

            # Priority 7: XPath by position (last resort)
            # Build XPath relative to parent
            try:
                parent = element.find_element('xpath', '..')
                parent_tag = parent.tag_name
                siblings = parent.find_elements('tag name', tag_name)
                position = siblings.index(element) + 1
                locator_info['xpath_pos'] = f"xpath://{parent_tag}/{tag_name}[{position}]"
                locator_info['name'] = f"{tag_name}_position_{position}_locator"
                locator_info['priority'] = 7
                logger.info(f"   📍 Step {step_number}: Captured POSITIONAL XPath locator: {locator_info['xpath_pos']}")
                return locator_info
            except Exception:
                pass

            logger.warning(f"   ⚠️  Step {step_number}: Could not generate reliable locator for element")
            return {}

        except Exception as e:
            logger.error(f"   ❌ Step {step_number}: Error capturing element locator: {e}")
            return {}

    def _run_accessibility_checks(self):
        logger.debug("   🔍 Running accessibility checks...")
        try:
            from selenium.webdriver.common.by import By
            findings = []
            for img in self.driver.find_elements(By.TAG_NAME, 'img')[:50]:
                alt = img.get_attribute('alt')
                if alt is None or alt.strip() == '':
                    findings.append({'type': 'image_missing_alt', 'src': img.get_attribute('src')})
            for btn in self.driver.find_elements(By.TAG_NAME, 'button')[:50]:
                txt = btn.text.strip()
                aria = btn.get_attribute('aria-label')
                if (not txt) and (not aria):
                    findings.append({'type': 'button_no_label'})
            for inp in self.driver.find_elements(By.TAG_NAME, 'input')[:50]:
                t = (inp.get_attribute('type') or '').lower()
                if t in ['hidden', 'submit', 'button']:
                    continue
                placeholder = inp.get_attribute('placeholder')
                aria = inp.get_attribute('aria-label')
                id_attr = inp.get_attribute('id')
                labeled = False
                if id_attr:
                    try:
                        lbl = self.driver.find_element(By.XPATH, f"//label[@for='{id_attr}']")
                        if lbl and lbl.text.strip():
                            labeled = True
                    except Exception:
                        pass
                if not labeled and not placeholder and not aria:
                    findings.append({'type': 'input_no_label', 'name': inp.get_attribute('name')})
            if findings:
                self.accessibility_findings.extend(findings)
                self.bug_report['accessibility_issues'].extend(findings)
                logger.info(f"   ♿ Found {len(findings)} accessibility issues")
            else:
                logger.debug("   ♿ No accessibility issues found")
        except Exception as e:
            logger.debug(f"   ⚠️  Accessibility check error: {e}")

    def _run_security_checks(self):
        logger.debug("   🔒 Running security checks...")
        try:
            from selenium.webdriver.common.by import By
            findings = []
            current_url = self.driver.current_url
            if current_url.startswith('https://'):
                for tag in ['img', 'script', 'iframe']:
                    for el in self.driver.find_elements(By.TAG_NAME, tag)[:50]:
                        src = el.get_attribute('src') or el.get_attribute('data-src')
                        if src and src.startswith('http://'):
                            findings.append({'type': 'mixed_content', 'tag': tag, 'src': src})
            for form in self.driver.find_elements(By.TAG_NAME, 'form')[:30]:
                action = form.get_attribute('action')
                if action and action.startswith('http://'):
                    findings.append({'type': 'insecure_form_action', 'action': action})
            if findings:
                self.security_findings.extend(findings)
                self.bug_report['security_issues'].extend(findings)
                logger.info(f"   🔒 Found {len(findings)} security issues")
            else:
                logger.debug("   🔒 No security issues found")
        except Exception as e:
            logger.debug(f"   ⚠️  Security check error: {e}")

    async def execute_step_smartly(self, step: TestStep, test_case: TestCase) -> Tuple[bool, str]:
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 EXECUTING STEP {step.step_number}: {step.description}")
        logger.info(f"{'='*80}")

        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            # Capture state before action
            logger.info("   📊 Capturing pre-action state...")
            self.capture_network_logs()
            self.capture_console_errors()
            self.capture_dom_snapshot(step.description)
            self.capture_performance_metrics()

            description_lower = step.description.lower()
            success = False
            message = ""
            actual_element = None
            actual_locator_used = None

            # Smart action detection and execution
            if any(word in description_lower for word in ['navigate', 'open', 'go to', 'visit']):
                logger.info("   🌐 Action type: NAVIGATION")
                # Navigation
                url_match = re.search(r'https?://[^\s\)]+', step.description)
                if url_match:
                    url = url_match.group(0).rstrip('/').rstrip(',').rstrip('.')
                    logger.info(f"   🔗 Navigating to: {url}")
                    self.driver.get(url)
                    self._wait_for_page_load()
                    success = True
                    message = f"Navigated to {url}"
                    logger.info(f"   ✅ Navigation successful: {url}")
                else:
                    message = "No URL found in navigation step"
                    logger.error(f"   ❌ {message}")

            elif any(word in description_lower for word in ['click', 'press', 'select', 'choose']):
                logger.info("   🖱️  Action type: CLICK")
                # Click action - smart locator finding with detailed logging
                success, message, actual_element, actual_locator_used = await self._smart_click_enhanced(step, test_case)

            elif any(word in description_lower for word in ['enter', 'input', 'type', 'fill']):
                logger.info("   ⌨️  Action type: INPUT")
                # Input action - smart field finding with detailed logging
                success, message, actual_element, actual_locator_used = await self._smart_input_enhanced(step, test_case)

            elif any(word in description_lower for word in ['verify', 'check', 'confirm', 'validate']):
                logger.info("   ✔️  Action type: VERIFICATION")
                # Verification action
                success, message = self._smart_verify(step, test_case)
                logger.info(f"   {'✅' if success else '❌'} Verification: {message}")

            else:
                logger.info("   🔄 Action type: DEFAULT (attempting click)")
                # Default: try to find and click
                success, message, actual_element, actual_locator_used = await self._smart_click_enhanced(step, test_case)

            # Capture the ACTUAL locator that was used successfully
            if actual_element and success:
                logger.info(f"   🎯 Capturing ACTUAL element locator for step {step.step_number}...")
                locator_info = self._capture_element_locator(actual_element, step.step_number, "click" if "click" in description_lower else "input")

                if locator_info:
                    # Get the best locator (first one in priority order)
                    best_locator_key = next((k for k in ['id', 'name_attr', 'data_attr', 'text', 'aria', 'class', 'xpath_pos'] if k in locator_info), None)
                    if best_locator_key:
                        best_locator = locator_info[best_locator_key]
                        locator_name = locator_info['name']

                        # Store in step_locators
                        self.step_locators[step.step_number] = {
                            locator_name: best_locator
                        }

                        # Also update captured_locators
                        self.captured_locators[locator_name] = best_locator

                        # Store full element info
                        self.actual_elements_used[step.step_number] = {
                            'locator_name': locator_name,
                            'locator_value': best_locator,
                            'priority': locator_info.get('priority', 99),
                            'element_text': actual_element.text.strip()[:50] if actual_element.text else '',
                            'element_tag': actual_element.tag_name,
                            'all_locators': locator_info
                        }

                        logger.info(f"   ✅ ACTUAL locator saved: {locator_name} = '{best_locator}'")
                    else:
                        logger.warning(f"   ⚠️  Could not extract best locator from: {locator_info}")

            # Capture state after action
            logger.info("   📊 Capturing post-action state...")
            self.capture_screenshot(step.description)
            self.capture_network_logs()
            self.capture_console_errors()

            # Run checks
            self._run_accessibility_checks()
            self._run_security_checks()

            # Analyze step for issues
            self._analyze_step_for_issues(step, success, message)

            logger.info(f"   {'✅ SUCCESS' if success else '❌ FAILED'}: {message}")
            logger.info(f"{'='*80}\n")

            return success, message

        except Exception as e:
            logger.error(f"   ❌ EXCEPTION in execute_step_smartly: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"Error: {str(e)}"

    async def _smart_click_enhanced(self, step: TestStep, test_case: TestCase) -> Tuple[bool, str, Any, str]:
        """Enhanced smart click that returns the actual element and locator used"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            description = step.description.lower()

            # Try multiple strategies with detailed logging
            strategies = []

            # Strategy 1: Look for quoted text
            quoted = re.findall(r'"([^"]+)"', step.description)
            if quoted:
                logger.info(f"   📝 Found quoted text: '{quoted[0]}'")
                strategies.append(('link', quoted[0], f"Link with text: {quoted[0]}"))
                strategies.append(('xpath', f"//button[normalize-space(text())='{quoted[0]}']", f"Button with exact text: {quoted[0]}"))
                strategies.append(('xpath', f"//a[normalize-space(text())='{quoted[0]}']", f"Link with exact text: {quoted[0]}"))
                strategies.append(('xpath', f"//*[normalize-space(text())='{quoted[0]}']", f"Any element with exact text: {quoted[0]}"))
                strategies.append(('xpath', f"//button[contains(text(), '{quoted[0]}')]", f"Button containing: {quoted[0]}"))
                strategies.append(('xpath', f"//*[contains(text(), '{quoted[0]}')]", f"Any element containing: {quoted[0]}"))

            # Strategy 2: Extract key words and build smarter selectors
            key_words = ['button', 'link', 'menu', 'explore', 'continue', 'submit', 'checkout', 'plan', 'select', 'view', 'choose']
            for word in key_words:
                if word in description:
                    logger.info(f"   🔑 Found keyword: '{word}'")
                    strategies.append(('xpath', f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{word}')]", f"Button with keyword: {word}"))
                    strategies.append(('xpath', f"//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{word}')]", f"Link with keyword: {word}"))

            # Strategy 3: Common button/link patterns
            strategies.append(('css', 'button[type="submit"]', "Submit button"))
            strategies.append(('css', 'a.btn', "Link with btn class"))
            strategies.append(('css', 'button.btn', "Button with btn class"))

            logger.info(f"   🔍 Trying {len(strategies)} locator strategies...")

            # Try each strategy
            for idx, (by_type, value, description_str) in enumerate(strategies, 1):
                try:
                    if isinstance(by_type, str):
                        by_type = self._get_by_type(by_type)

                    logger.debug(f"      [{idx}/{len(strategies)}] Trying: {description_str}")

                    element = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((by_type, value))
                    )

                    # Get element details before clicking
                    elem_text = element.text.strip()[:50] if element.text else ''
                    elem_tag = element.tag_name

                    # Click it
                    element.click()

                    logger.info(f"   ✅ CLICKED using strategy [{idx}]: {description_str}")
                    logger.info(f"      Element: <{elem_tag}> with text: '{elem_text}'")

                    # Return success with actual element and locator
                    actual_locator = f"{by_type}:{value}" if hasattr(by_type, '__name__') else f"{value}"
                    return True, f"Successfully clicked: {description_str}", element, actual_locator

                except Exception as e:
                    logger.debug(f"      ✗ Strategy [{idx}] failed: {str(e)[:50]}")
                    continue

            logger.error(f"   ❌ All {len(strategies)} strategies failed for: {step.description}")
            return False, f"Could not find clickable element for: {step.description}", None, None

        except Exception as e:
            logger.error(f"   ❌ Click error: {str(e)}")
            return False, f"Click error: {str(e)}", None, None

    async def _smart_input_enhanced(self, step: TestStep, test_case: TestCase) -> Tuple[bool, str, Any, str]:
        """Enhanced smart input that returns the actual element and locator used"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            # Extract value to input
            value_match = re.search(r'"([^"]+)"', step.description)
            input_value = value_match.group(1) if value_match else "test_input"

            logger.info(f"   ⌨️  Input value: '{input_value}'")

            # Try to find input field with detailed logging
            strategies = [
                (By.NAME, 'search', "Name: search"),
                (By.NAME, 'domain', "Name: domain"),
                (By.NAME, 'email', "Name: email"),
                (By.NAME, 'username', "Name: username"),
                (By.ID, 'search', "ID: search"),
                (By.CSS_SELECTOR, 'input[type="text"]', "Input type=text"),
                (By.CSS_SELECTOR, 'input[type="search"]', "Input type=search"),
                (By.CSS_SELECTOR, 'input[placeholder]', "Input with placeholder")
            ]

            logger.info(f"   🔍 Trying {len(strategies)} input field strategies...")

            for idx, (by_type, value, desc) in enumerate(strategies, 1):
                try:
                    logger.debug(f"      [{idx}/{len(strategies)}] Trying: {desc}")

                    element = WebDriverWait(self.driver, 3).until(
                        EC.presence_of_element_located((by_type, value))
                    )
                    element.clear()
                    element.send_keys(input_value)

                    logger.info(f"   ✅ INPUT using strategy [{idx}]: {desc}")

                    actual_locator = f"{by_type}:{value}"
                    return True, f"Successfully entered: {input_value}", element, actual_locator

                except Exception as e:
                    logger.debug(f"      ✗ Strategy [{idx}] failed: {str(e)[:50]}")
                    continue

            logger.error(f"   ❌ All {len(strategies)} input strategies failed")
            return False, f"Could not find input field for: {step.description}", None, None

        except Exception as e:
            logger.error(f"   ❌ Input error: {str(e)}")
            return False, f"Input error: {str(e)}", None, None

    async def generate_ai_bug_report(self) -> str:
        logger.info("📝 Generating enhanced bug report...")
        # Use parent report then append summary of new findings
        base_report = await super().generate_ai_bug_report()

        # Add summary of actual elements used
        elements_summary = "\n## Actual Elements Used Per Step\n"
        if self.actual_elements_used:
            for step_num, elem_info in sorted(self.actual_elements_used.items()):
                elements_summary += f"- **Step {step_num}**: `{elem_info['locator_name']}` = `'{elem_info['locator_value']}'` (priority {elem_info['priority']})\n"
                if elem_info.get('element_text'):
                    elements_summary += f"  - Element text: \"{elem_info['element_text']}\"\n"
        else:
            elements_summary += "No elements tracked (navigation only or failed steps)\n"

        summary_addendum = f"\n## Extended Analysis\n- Accessibility Issues Detected: {len(self.accessibility_findings)}\n- Security Issues Detected: {len(self.security_findings)}\n{elements_summary}\n"
        return base_report + summary_addendum

