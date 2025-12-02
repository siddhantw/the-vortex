# Browser Agent use case for agentic AI-driven web navigation
# Leverages Azure OpenAI for planning and DOMSnapshotService for smart traversal

# Ensure streamlit/asyncio compatibility first
try:
    from gen_ai import streamlit_fix  # noqa: F401
except Exception:
    try:
        import streamlit_fix  # type: ignore # noqa: F401
    except Exception:
        pass

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Add parent of use_cases (gen_ai) to path for shared utilities
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_AI_DIR = os.path.dirname(CURRENT_DIR)
if GEN_AI_DIR not in sys.path:
    sys.path.insert(0, GEN_AI_DIR)

# Optional imports
AZURE_AVAILABLE = False
try:
    from azure_openai_client import AzureOpenAIClient  # type: ignore
    AZURE_AVAILABLE = True
except Exception:
    AZURE_AVAILABLE = False

NOTIFICATIONS_AVAILABLE = False
try:
    import notifications  # type: ignore
    NOTIFICATIONS_AVAILABLE = True
except Exception:
    NOTIFICATIONS_AVAILABLE = False

# DOM Snapshot service (wraps Selenium/Playwright discovery and screenshots)
DOM_AVAILABLE = False
try:
    from rag.dom_snapshot_service import DOMSnapshotService  # type: ignore
    DOM_AVAILABLE = True
except Exception:
    # Fallback relative import if needed
    try:
        dom_path = os.path.join(GEN_AI_DIR, 'rag')
        if dom_path not in sys.path:
            sys.path.insert(0, dom_path)
        from dom_snapshot_service import DOMSnapshotService  # type: ignore
        DOM_AVAILABLE = True
    except Exception:
        DOM_AVAILABLE = False

# Robot writer for export
ROBOT_WRITER_AVAILABLE = False
try:
    from robot_writer.robot_writer import RobotWriter  # type: ignore
    ROBOT_WRITER_AVAILABLE = True
except Exception:
    try:
        from gen_ai.robot_writer.robot_writer import RobotWriter  # type: ignore
        ROBOT_WRITER_AVAILABLE = True
    except Exception:
        ROBOT_WRITER_AVAILABLE = False

logger = logging.getLogger("browser_agent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class AgentStep:
    idx: int
    action: str  # click|input|select|navigate|wait_for|assert_text
    target: Optional[str] = None  # human-readable target text/label
    selectors: Optional[List[str]] = None  # candidate selectors (css/xpath/text=)
    value: Optional[str] = None
    wait_for: Optional[str] = None  # text/css/xpath to wait for
    notes: Optional[str] = None


@dataclass
class StepResult:
    step: AgentStep
    status: str  # success|skipped|failed
    detail: str
    attempted_selectors: List[str]
    screenshot: Optional[str] = None
    duration_sec: float = 0.0


class SimpleEventBus:
    """Tiny pub-sub to stream logs into Streamlit during execution."""
    def __init__(self, placeholder: Optional[st.delta_generator.DeltaGenerator] = None):
        self.lines: List[str] = []
        self.placeholder = placeholder

    def log(self, message: str):
        ts = time.strftime('%H:%M:%S')
        line = f"[{ts}] {message}"
        self.lines.append(line)
        if self.placeholder is not None:
            self.placeholder.code("\n".join(self.lines))
        logger.info(message)


class BrowserAgentCore:
    def __init__(self, headless: bool = True, prefer_playwright: bool = True, azure_client: Optional[AzureOpenAIClient] = None, event_bus: Optional[SimpleEventBus] = None,
                 retries: int = 2, backoff_ms: int = 200, screenshot_on_success: bool = False, screenshot_on_failure: bool = True,
                 incognito: bool = True, viewport: Optional[Dict[str, int]] = None,
                 post_click_wait_selector: Optional[str] = None, post_click_wait_timeout_ms: int = 3000,
                 fast_mode: bool = True, concurrent_probe_top_n: int = 3,
                 analyze_first: bool = True, dom_refresh_per_step: bool = True,
                 attempt_top_k: int = 6, use_azure_ranking: bool = False):
        self.headless = headless
        self.prefer_playwright = prefer_playwright
        self.azure = azure_client
        self.bus = event_bus or SimpleEventBus()
        self.dom: Optional[DOMSnapshotService] = None
        self.playwright_ready = False
        self.selenium_ready = False
        self.page_opened = False
        self.current_url: Optional[str] = None
        self.last_snapshot: Optional[Any] = None
        # Execution controls
        self.retries = max(0, int(retries))
        self.backoff_ms = max(0, int(backoff_ms))
        self.screenshot_on_success = bool(screenshot_on_success)
        self.screenshot_on_failure = bool(screenshot_on_failure)
        # Session controls
        self.incognito = bool(incognito)
        self.viewport = viewport or {"width": 1920, "height": 1080}
        self.post_click_wait_selector = (post_click_wait_selector or "").strip() or None
        self.post_click_wait_timeout_ms = int(post_click_wait_timeout_ms)
        # Performance controls
        self.fast_mode = bool(fast_mode)
        self.concurrent_probe_top_n = max(0, int(concurrent_probe_top_n))
        # Analysis controls
        self.analyze_first = bool(analyze_first)
        self.dom_refresh_per_step = bool(dom_refresh_per_step)
        # Ranking/attempt controls
        self.attempt_top_k = max(1, int(attempt_top_k))
        self.use_azure_ranking = bool(use_azure_ranking)
        # Artifacts
        self.artifact_dir = os.path.join(os.getcwd(), "generated_tests")
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.screenshots_dir = os.path.join(self.artifact_dir, "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        # Selector memory (per-domain cache of successful selectors)
        self.selector_memory_path = os.path.join(self.artifact_dir, "selector_memory.json")
        self.selector_memory: Dict[str, Dict[str, List[str]]] = {}
        self.last_used_selector: Optional[str] = None
        try:
            self._load_selector_memory()
        except Exception:
            self.selector_memory = {}

    async def initialize(self):
        if not DOM_AVAILABLE:
            self.bus.log("DOM snapshot service not available. Execution will run in plan-only mode.")
            return
        # Configure fast, clean, full-viewport sessions
        self.dom = DOMSnapshotService({
            "headless": self.headless,
            "viewport": {"width": int(self.viewport.get("width", 1920)), "height": int(self.viewport.get("height", 1080))},
            "incognito": self.incognito,
            "light_mode": True,
            "wait_after_load_ms": 300,
            "page_load_state": "domcontentloaded"
        })
        # Try playwright first if requested
        if self.prefer_playwright:
            try:
                self.playwright_ready = await self.dom.initialize_playwright()
            except Exception as e:
                self.bus.log(f"Playwright init failed: {e}")
                self.playwright_ready = False
        # Fallback to Selenium if Playwright not ready
        if not self.playwright_ready:
            try:
                self.selenium_ready = self.dom.initialize_selenium()
            except Exception as e:
                self.bus.log(f"Selenium init failed: {e}")
                self.selenium_ready = False
        if self.playwright_ready or self.selenium_ready:
            self.bus.log("Browser initialized successfully.")
        else:
            self.bus.log("No browser engine available. Running in plan-only mode.")

    async def close(self):
        try:
            if self.dom and getattr(self.dom, 'playwright_browser', None):
                await self.dom.playwright_browser.close()
            if self.dom and getattr(self.dom, 'playwright', None):
                await self.dom.playwright.stop()
            if self.dom and getattr(self.dom, 'driver', None):
                try:
                    self.dom.driver.quit()
                except Exception:
                    pass
        except Exception as e:
            self.bus.log(f"Cleanup encountered an issue: {e}")

    # New: planner JSON sanitizer
    def _sanitize_planner_json(self, text: str) -> Dict[str, Any]:
        raw = text.strip()
        # Strip markdown code fences if present
        if raw.startswith('```'):
            raw = raw.strip('`')
            # Remove possible language hint like ```json
            if raw.startswith('json'):
                raw = raw[4:]
        # Try direct JSON first
        try:
            return json.loads(raw)
        except Exception:
            pass
        # Extract JSON block between first { and last }
        if '{' in raw and '}' in raw:
            start = raw.find('{')
            end = raw.rfind('}') + 1
            candidate = raw[start:end]
            # Remove trailing commas before closing brackets
            candidate = candidate.replace(',\n]', '\n]').replace(',\n}', '\n}')
            try:
                return json.loads(candidate)
            except Exception:
                pass
        # Fallback: try to extract steps array and wrap
        try:
            import re
            m = re.search(r'"steps"\s*:\s*(\[[\s\S]*?])', raw, re.DOTALL)
            if m:
                steps_str = m.group(1)
                steps_str = steps_str.replace("'", '"')
                steps = json.loads(steps_str)
                return {"steps": steps}
        except Exception:
            pass
        raise ValueError("Unable to parse planner JSON response")

    async def capture_snapshot(self, url: str) -> Optional[Any]:
        if not self.dom:
            return None
        try:
            # Prefer non-navigating snapshot to avoid disrupting current flow
            if getattr(self.dom, 'playwright_page', None) or getattr(self.dom, 'driver', None):
                try:
                    snapshot = await self.dom.capture_current_page_snapshot()  # type: ignore[attr-defined]
                except Exception:
                    snapshot = await self.dom.capture_page_snapshot(url, wait_for_load=False)
            else:
                snapshot = await self.dom.capture_page_snapshot(url, wait_for_load=True)
            self.last_snapshot = snapshot
            return snapshot
        except Exception as e:
            self.bus.log(f"Snapshot capture failed: {e}")
            return None

    # Helper: URL resolution
    def _looks_like_url(self, s: str) -> bool:
        try:
            from urllib.parse import urlparse
            u = urlparse(s)
            return bool(u.scheme and u.netloc) or s.startswith('/')
        except Exception:
            return False

    def _resolve_url(self, base: str, target: str) -> str:
        try:
            from urllib.parse import urljoin
            return urljoin(base, target)
        except Exception:
            return target

    # New: selector resolution using current snapshot
    def _selectors_from_snapshot(self, target: Optional[str], prefer: Optional[str] = None) -> List[str]:
        selectors: List[str] = []
        if not target or not self.last_snapshot:
            return selectors
        t = target.strip().lower()
        try:
            elements = getattr(self.last_snapshot, 'elements', []) or []
            scored: List[Tuple[int, Any]] = []
            for el in elements:
                score = 0
                matched = False
                # Text match
                if getattr(el, 'text', None) and t in el.text.lower():
                    score += 5
                    matched = True
                # Attribute matches
                attrs = getattr(el, 'attributes', {}) or {}
                for key in ["aria-label", "placeholder", "name", "id", "data-testid", "data-test", "data-element-label", "title", "value"]:
                    val = attrs.get(key)
                    if val and t in str(val).lower():
                        score += 4
                        matched = True
                # Tag preference based on action intent (additive only; do not include elements without any text/attr match)
                if prefer == 'input' and getattr(el, 'tag_name', None) in ['input', 'textarea']:
                    score += 2
                if prefer == 'click' and getattr(el, 'tag_name', None) in ['button', 'a']:
                    score += 2
                if prefer == 'select' and getattr(el, 'tag_name', None) == 'select':
                    score += 2
                # Specialization: password/email
                if 'password' in t and (attrs or {}).get('type') == 'password':
                    score += 3
                    matched = True
                if 'email' in t and (attrs or {}).get('type') == 'email':
                    score += 3
                    matched = True
                # Only keep candidates that matched target text or attributes in some way
                if matched and score > 0:
                    scored.append((score, el))
            # Choose top matches and collect their robust selectors
            for _, el in sorted(scored, key=lambda x: x[0], reverse=True)[:6]:
                # Prefer alt selectors, then stable css, then xpath, then text
                for a in (getattr(el, 'alt_selectors', None) or [])[:3]:
                    selectors.append(a)
                for s in (getattr(el, 'css_selectors', None) or [])[:2]:
                    selectors.append(s)
                for x in (getattr(el, 'xpath_selectors', None) or [])[:1]:
                    selectors.append(x)
                if getattr(el, 'text', None):
                    txt = el.text.strip()
                    if txt:
                        selectors.append(f"text={txt}")
        except Exception:
            return selectors
        # de-dupe, preserve order
        seen = set()
        uniq: List[str] = []
        for s in selectors:
            if s not in seen and s:
                uniq.append(s)
                seen.add(s)
        return uniq

    # New: simple narration generator
    def _narrate_step(self, step: AgentStep) -> str:
        action = step.action
        tgt = step.target or ''
        if action == 'click':
            return f"Click the '{tgt}' element or button"
        if action == 'input':
            v = (step.value or '').strip()
            return f"Enter '{v}' into the '{tgt}' field"
        if action == 'select':
            v = (step.value or '').strip()
            return f"Select '{v}' from the '{tgt}' dropdown"
        if action == 'navigate':
            return f"Navigate to {tgt or 'the start page'}"
        if action == 'wait_for':
            return f"Wait for '{step.wait_for or tgt}' to appear"
        if action == 'assert_text':
            return f"Verify that '{tgt}' is visible on the page"
        return f"Perform action '{action}' on '{tgt}'"

    # New: simple heuristic planner for non-Azure environments
    def _heuristic_steps(self, instruction: str) -> List[AgentStep]:
        steps: List[AgentStep] = []
        text = (instruction or '').strip()
        low = text.lower()
        import re
        # Extract first quoted phrase as a useful token
        qmatch = re.search(r"'([^']{2,})'|\"([^\"]{2,})\"", text)
        quoted = (qmatch.group(1) or qmatch.group(2)) if qmatch else None
        # Login flow
        if any(k in low for k in ["login", "log in", "sign in", "signin"]):
            steps.append(AgentStep(idx=0, action='click', target='Login'))
            steps.append(AgentStep(idx=0, action='input', target='Email', value='testuser@example.com'))
            steps.append(AgentStep(idx=0, action='input', target='Password', value='TestPassword123!'))
            steps.append(AgentStep(idx=0, action='click', target='Sign in'))
        # Search flow
        if 'search' in low:
            val = quoted or 'pricing'
            steps.append(AgentStep(idx=0, action='input', target='Search', value=val))
            steps.append(AgentStep(idx=0, action='click', target='Search'))
        # Navigate to account/profile settings
        if any(k in low for k in ["account settings", "profile settings", "settings", "account"]):
            steps.append(AgentStep(idx=0, action='click', target='Account Settings'))
        if 'profile' in low:
            steps.append(AgentStep(idx=0, action='click', target='Profile'))
        # Update profile name pattern
        mname = re.search(r"update\s+profile\s+name\s+to\s+'([^']{2,})'|update\s+profile\s+name\s+to\s+\"([^\"]{2,})\"", low)
        if mname:
            newname = (mname.group(1) or mname.group(2)) if (mname.group(1) or mname.group(2)) else (quoted or 'QA Bot')
            steps.append(AgentStep(idx=0, action='input', target='Name', value=newname))
        # Save/apply
        if any(k in low for k in ["save", "apply", "submit"]):
            steps.append(AgentStep(idx=0, action='click', target='Save'))
        # If nothing actionable detected, try clicking the quoted token if present
        if not steps and quoted:
            steps.append(AgentStep(idx=0, action='click', target=quoted))
        return steps

    # New: alternate selector generator from a human target
    def _alternate_selectors_from_target(self, target: Optional[str]) -> List[str]:
        if not target:
            return []
        t = target.strip()
        lowered = t.lower()
        alts: List[str] = []
        # Strong clickable heuristics first
        # Prefer role selectors for better accessibility matching
        alts.append(f"role=button[name='{t}']")
        alts.append(f"role=link[name='{t}']")
        # Exact visible text next
        alts.append(f"text={t}")
        # Then smart locators
        alts.append(f"label={t}")
        alts.append(f"placeholder={t}")
        # Common attribute-driven selectors (CSS)
        for attr in ["aria-label", "placeholder", "name", "id", "data-testid", "data-test", "data-element-label", "title", "value", "alt", "aria-describedby", "aria-labelledby"]:
            alts.append(f"[{attr}='{t}']")
            alts.append(f"[{attr}*='{t}']")
        # Heuristics for login/email/password etc.
        if any(k in lowered for k in ["email", "user", "username", "login"]):
            alts.extend(["input[type='email']", "input[name*='user']", "input[name*='email']", "label=Email", "placeholder=Email"])
        if "password" in lowered:
            alts.extend(["input[type='password']", "input[name*='pass']", "label=Password", "placeholder=Password"])
        if any(k in lowered for k in ["submit", "sign in", "sign-in", "log in", "login", "get started", "start", "view plans", "choose plan", "select plan"]):
            alts.extend(["role=button[name='Login']", "role=button[name='Sign in']", "role=button[name='Get started']", "text=Login", "text=Sign in", "text=Get started"])
        # Generic clickable elements (keep at the very end to avoid accidental mismatches)
        alts.extend(["role=button", "button", "a[role='button']", "[role='button']", ".btn", ".button"])
        # De-dupe while preserving order
        seen = set()
        out: List[str] = []
        for a in alts:
            if a and a not in seen:
                out.append(a)
                seen.add(a)
        return out

    # New: allow explicit literal selectors in the target to take priority
    def _literal_selectors_from_target(self, target: Optional[str]) -> List[str]:
        if not target:
            return []
        t = target.strip()
        out: List[str] = []
        # Recognize Playwright text engines and role engines directly
        if t.startswith('role=') or t.startswith('text=') or t.startswith('label=') or t.startswith('placeholder=') or t.startswith('testid='):
            out.append(t)
        # XPath hints
        if t.startswith('xpath='):
            x = t[len('xpath='):].strip()
            if x.startswith('/'):
                out.append(x)
        if t.startswith('//') or t.startswith('/'):
            out.append(t)
        # CSS hints
        if t.startswith('css='):
            out.append(t[len('css='):].strip())
        # Simple CSS patterns: starts with '.', '#', '[' or tag[attr=...]
        import re
        if t.startswith(('.', '#', '[')) or re.match(r"^[A-Za-z][A-Za-z0-9_-]*\[[^]]+]$", t):
            out.append(t)
        # De-dupe
        seen = set()
        res: List[str] = []
        for s in out:
            if s and s not in seen:
                res.append(s)
                seen.add(s)
        return res

    def _input_specific_candidates(self, target: Optional[str], value: Optional[str]) -> List[str]:
        """Prioritized, input-focused locator candidates derived from target/value.
        Examples: label=Email, placeholder=Email, input[name*='email'], input[type='email'].
        """
        if not target and not value:
            return []
        t = (target or '').strip()
        tl = t.lower()
        val = (value or '').strip()
        vl = val.lower()
        cands: List[str] = []
        # Strong accessibility-first
        if t:
            cands.append(f"label={t}")
            cands.append(f"placeholder={t}")
            cands.append(f"role=textbox[name='{t}']")
        # Attribute-driven inputs
        def add_attr_like(keys: List[str], token: str):
            for k in keys:
                cands.append(f"input[{k}='{token}']")
                cands.append(f"input[{k}*='{token}']")
                cands.append(f"textarea[{k}='{token}']")
                cands.append(f"textarea[{k}*='{token}']")
        if t:
            for token in {t, tl}:
                add_attr_like(["name","id","aria-label","aria-labelledby","placeholder","data-testid","data-test","title"], token)
        # Heuristics by semantic intent from target or value
        looks_email = ("email" in tl) or ("@" in val)
        looks_pass = ("password" in tl)
        looks_search = ("search" in tl) or ("query" in tl) or (vl in ("q",))
        looks_phone = ("phone" in tl) or ("mobile" in tl)
        looks_name = ("name" in tl) and ("username" not in tl)
        if looks_email:
            cands.extend(["input[type='email']", "input[name*='email']", "input[id*='email']"])
        if looks_pass:
            cands.extend(["input[type='password']", "input[name*='pass']", "input[id*='pass']"])
        if looks_search:
            cands.extend(["input[type='search']", "input[name*='q']", "input[aria-label*='search']", "input[placeholder*='search']"])
        if looks_phone:
            cands.extend(["input[type='tel']", "input[name*='phone']", "input[id*='phone']"])
        if looks_name:
            cands.extend(["input[name*='name']", "input[id*='name']", "input[aria-label*='name']", "input[placeholder*='name']"])
        # Generic safe fallbacks last
        cands.extend([
            "input:not([type='hidden']):not([disabled])",
            "textarea:not([disabled])",
        ])
        # De-dupe while preserving order
        seen = set()
        out: List[str] = []
        for s in cands:
            if s and s not in seen:
                out.append(s)
                seen.add(s)
        return out

    def _parse_click_target(self, target: Optional[str]) -> Tuple[str, List[str]]:
        """Extract primary text and container qualifiers from a descriptive target.
        Examples:
        - "Choose Plan button within the RECOMMENDED plan container" -> ("Choose Plan", ["RECOMMENDED"])
        - "View Plans in WordPress Hosting" -> ("View Plans", ["WordPress Hosting"])
        """
        if not target:
            return "", []
        t = target.strip()
        # Split by common container prepositions
        import re
        parts = re.split(r"\bwithin\b|\binside\b|\bin\b|\bunder\b|\bunder the\b|\binside the\b", t, flags=re.IGNORECASE)
        primary = parts[0].strip()
        qualifiers: List[str] = []
        if len(parts) > 1:
            rest = ' '.join(parts[1:]).strip()
            # remove trailing words like container/section/plan(s)
            rest = re.sub(r"\b(container|section|area|pane|panel|card|box|module|tab|plan|plans)\b", "", rest, flags=re.IGNORECASE).strip()
            # extract quoted phrases or ALLCAPS tokens or Title Case sequences
            q: List[str] = []
            q.extend([m.strip('"\'').strip() for m in re.findall(r"['\"]([^'\"]{2,})['\"]", rest)])
            q.extend([m for m in re.findall(r"\b[A-Z]{3,}\b", rest)])
            # If nothing found, take last 2-4 words as a phrase
            if not q and rest:
                toks = [w for w in rest.split() if w]
                if toks:
                    q.append(' '.join(toks[-3:]).strip())
            qualifiers = [s for s in q if s]
        # Clean primary: drop generic nouns like button/link/cta
        primary = re.sub(r"\b(button|link|cta|tab)\b", "", primary, flags=re.IGNORECASE).strip()
        # If primary holds quotes, prefer inside
        m = re.search(r"['\"]([^'\"]{2,})['\"]", primary)
        if m:
            primary = m.group(1).strip()
        return primary, qualifiers

    def _scoped_xpath_candidates(self, primary: str, qualifiers: List[str]) -> List[str]:
        """Create robust scoped XPath selectors: find a container matching any qualifier, then find a button/link with the primary text.
        Returns raw XPath strings (without the 'xpath=' prefix).
        """
        if not primary or not qualifiers:
            return []
        # Escape quotes by switching to concat when needed
        def lit(s: str) -> str:
            if "'" not in s:
                return f"'{s}'"
            if '"' not in s:
                return f'"{s}"'
            parts = s.split("'")
            return "concat(" + ", ".join([f"'{p}'" if i == len(parts) - 1 else f"'{p}', \"'\", " for i, p in enumerate(parts)]) + ")"
        xpaths: List[str] = []
        # For each qualifier, try a few container scopes
        for qual in qualifiers:
            q = qual.strip()
            if not q:
                continue
            pq = q.replace("'", "\"")
            pp = primary.replace("'", "\"")
            # Container any element containing qualifier text
            container = f"//*[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), {json.dumps(q.lower())})]"
            # Descendant button/link by text or aria-label/title
            btn_cond = (
                f"(.//button | .//*[@role='button'] | .//a | .//*[@role='link'])[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), {json.dumps(primary.lower())})"
                f" or contains(@aria-label, {json.dumps(primary)}) or contains(@title, {json.dumps(primary)})]"
            )
            xpaths.append(f"{container}//{btn_cond}")
            # Slightly stricter: restrict container to sections/cards/divs
            container2 = f"//*[self::section or self::div or self::li or self::article][contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), {json.dumps(q.lower())})]"
            xpaths.append(f"{container2}//{btn_cond}")
        # De-dupe
        seen = set()
        res: List[str] = []
        for xp in xpaths:
            if xp not in seen:
                res.append(xp)
                seen.add(xp)
        return res

    def _azure_suggest_locators(self, target: Optional[str], action: str, base_candidates: List[str], qualifiers: Optional[List[str]] = None) -> List[str]:
        if not target or not self.azure or not getattr(self.azure, 'is_configured', lambda: False)():
            return []
        elements = self._collect_element_candidates_for_llm(target)
        if not elements:
            return []
        system = {
            'role': 'system',
            'content': (
                "You are a precise web test locator ranker. Given a user target, optional container qualifiers, a list of element summaries, and candidate selectors, "
                "return an ordered list of the best selectors to click or interact with. Prefer stable and accessible selectors first: role=button/link[name='...'], testid=..., CSS with ids/data attrs, then XPath. "
                "If qualifiers are provided, prioritize selectors that scope within containers whose text includes any qualifier. "
                "Only return selectors from provided candidates or from element.alt_selectors/css/xpath that directly map to those elements. Output JSON: {\"selectors\":[...]}"
            )
        }
        user_payload = {
            'target': target,
            'action': action,
            'qualifiers': qualifiers or [],
            'elements': elements,
            'candidates': base_candidates[:60]
        }
        user = {'role': 'user', 'content': json.dumps(user_payload, ensure_ascii=False)}
        try:
            resp = self.azure.chat_completion_create(messages=[system, user], temperature=0.2, max_tokens=500)
            text = resp.get('choices', [{}])[0].get('message', {}).get('content', '') if resp else ''
            ranked = self._extract_selector_list_from_json(text)
            pool = set(base_candidates) | {s for e in elements for s in (e.get('alt_selectors') or []) + (e.get('css') or []) + (e.get('xpath') or [])}
            final = []
            seen = set()
            for s in ranked:
                if s in pool and s not in seen:
                    final.append(s)
                    seen.add(s)
            return final[:15]
        except Exception:
            return []

    # --- Added helpers for execution, parsing, memory, and artifacts ---
    async def _open_url(self, url: str) -> None:
        """Open the URL using the active engine and update state + snapshot."""
        self.bus.log(f"Navigating to: {url}")
        # Playwright path
        if self.playwright_ready and self.dom:
            page = getattr(self.dom, 'playwright_page', None)
            try:
                if not page and getattr(self.dom, 'playwright_browser', None):
                    # Create a fresh context/page for stability
                    try:
                        context = await self.dom.playwright_browser.new_context(viewport=self.viewport)
                    except Exception:
                        context = await self.dom.playwright_browser.new_context()
                    page = await context.new_page()
                    setattr(self.dom, 'playwright_context', context)
                    setattr(self.dom, 'playwright_page', page)
                if page:
                    await page.goto(url, wait_until='domcontentloaded')
                    try:
                        await page.wait_for_load_state('networkidle', timeout=2000)
                    except Exception:
                        pass
                    try:
                        self.current_url = page.url
                    except Exception:
                        self.current_url = url
                    try:
                        await self.capture_snapshot(self.current_url or url)
                    except Exception:
                        pass
                    return
            except Exception as e:
                self.bus.log(f"Playwright navigation failed, will try Selenium: {e}")
        # Selenium path
        if self.selenium_ready and self.dom and getattr(self.dom, 'driver', None):
            try:
                driver = self.dom.driver
                driver.get(url)
                # Wait for ready state
                try:
                    for _ in range(20):
                        try:
                            rs = driver.execute_script("return document.readyState")
                            if rs == 'complete':
                                break
                        except Exception:
                            pass
                        time.sleep(0.15)
                except Exception:
                    pass
                try:
                    self.current_url = driver.current_url
                except Exception:
                    self.current_url = url
                try:
                    await self.capture_snapshot(self.current_url or url)
                except Exception:
                    pass
            except Exception as e:
                self.bus.log(f"Selenium navigation failed: {e}")

    async def _wait_for(self, selector_or_text: str, timeout_ms: int = 3000) -> bool:
        s = (selector_or_text or '').strip()
        if not s:
            return False
        # Playwright
        if self.playwright_ready and self.dom and getattr(self.dom, 'playwright_page', None):
            page = self.dom.playwright_page
            try:
                if s.startswith('xpath=') or s.startswith('//') or s.startswith('/'):
                    loc = page.locator(s if s.startswith('xpath=') else f"xpath={s}")
                    await loc.wait_for(state='visible', timeout=timeout_ms)
                    return True
                if s.startswith('text='):
                    loc = page.get_by_text(s[len('text='):])
                    await loc.wait_for(state='visible', timeout=timeout_ms)
                    return True
                if s.startswith('label='):
                    loc = page.get_by_label(s[len('label='):])
                    await loc.wait_for(state='visible', timeout=timeout_ms)
                    return True
                if s.startswith('placeholder='):
                    loc = page.get_by_placeholder(s[len('placeholder='):])
                    await loc.wait_for(state='visible', timeout=timeout_ms)
                    return True
                if s.startswith('role='):
                    loc = page.locator(s)
                    await loc.wait_for(state='visible', timeout=timeout_ms)
                    return True
                await page.wait_for_selector(s, state='visible', timeout=timeout_ms)
                return True
            except Exception:
                # Fallback to text search
                try:
                    loc = page.get_by_text(s)
                    await loc.wait_for(state='visible', timeout=timeout_ms)
                    return True
                except Exception:
                    return False
        # Selenium
        if self.selenium_ready and self.dom and getattr(self.dom, 'driver', None):
            from selenium.webdriver.common.by import By  # type: ignore
            from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
            from selenium.webdriver.support import expected_conditions as EC  # type: ignore
            driver = self.dom.driver
            try:
                by, expr = None, None
                if s.startswith('//') or s.startswith('/'):
                    by, expr = By.XPATH, s
                elif s.startswith('text='):
                    txt = s[len('text='):]
                    by, expr = By.XPATH, f"//*[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), {json.dumps(txt.lower())})]"
                elif s.startswith('role='):
                    role, name = self._parse_role_selector(s)
                    by, expr = By.XPATH, self._role_xpath(role or '', name)
                else:
                    by, expr = By.CSS_SELECTOR, s
                WebDriverWait(driver, timeout_ms/1000.0).until(EC.visibility_of_element_located((by, expr)))
                return True
            except Exception:
                return False
        return False

    async def _assert_text(self, text: str, timeout_ms: int = 3000) -> bool:
        t = (text or '').strip()
        if not t:
            return False
        if self.playwright_ready and self.dom and getattr(self.dom, 'playwright_page', None):
            page = self.dom.playwright_page
            try:
                loc = page.get_by_text(t)
                await loc.wait_for(state='visible', timeout=timeout_ms)
                return True
            except Exception:
                # Fallback: content scan
                try:
                    content = await page.content()
                    return t.lower() in (content or '').lower()
                except Exception:
                    return False
        if self.selenium_ready and self.dom and getattr(self.dom, 'driver', None):
            try:
                driver = self.dom.driver
                html = driver.page_source or ''
                return t.lower() in html.lower()
            except Exception:
                return False
        return False

    async def _capture_step_screenshot(self, step_idx: int, status: str) -> Optional[str]:
        ts = time.strftime('%Y%m%d_%H%M%S')
        fname = f"step_{step_idx}_{status}_{ts}.png"
        path = os.path.join(self.screenshots_dir, fname)
        try:
            if self.playwright_ready and self.dom and getattr(self.dom, 'playwright_page', None):
                await self.dom.playwright_page.screenshot(path=path, full_page=True)
                return path
            if self.selenium_ready and self.dom and getattr(self.dom, 'driver', None):
                self.dom.driver.save_screenshot(path)
                return path
        except Exception as e:
            self.bus.log(f"Screenshot failed: {e}")
        return None

    # --- Selector memory helpers ---
    def _get_domain(self, url: Optional[str]) -> str:
        try:
            from urllib.parse import urlparse
            netloc = urlparse(url or '').netloc.lower()
            return netloc.replace('www.', '')
        except Exception:
            return ''

    def _load_selector_memory(self) -> None:
        try:
            if os.path.exists(self.selector_memory_path):
                with open(self.selector_memory_path, 'r', encoding='utf-8') as f:
                    self.selector_memory = json.load(f)
            else:
                self.selector_memory = {}
        except Exception:
            self.selector_memory = {}

    def _save_selector_memory(self) -> None:
        try:
            with open(self.selector_memory_path, 'w', encoding='utf-8') as f:
                json.dump(self.selector_memory, f, indent=2)
        except Exception:
            pass

    def _remember_selector(self, domain: str, label: Optional[str], selector: Optional[str]) -> None:
        if not selector:
            return
        key = (label or '').strip()
        if not key:
            return
        dom = domain or ''
        self.selector_memory.setdefault(dom, {})
        self.selector_memory[dom].setdefault(key, [])
        arr = self.selector_memory[dom][key]
        if selector not in arr:
            arr.insert(0, selector)
        # Trim to 6 recent successful selectors per label
        self.selector_memory[dom][key] = arr[:6]
        self._save_selector_memory()

    def _get_memory_selectors(self, domain: str, label: Optional[str]) -> List[str]:
        key = (label or '').strip()
        if not key:
            return []
        return list(self.selector_memory.get(domain or '', {}).get(key, []))

    # --- Azure parsing and DOM element summarization ---
    def _extract_selector_list_from_json(self, text: str) -> List[str]:
        raw = (text or '').strip()
        if not raw:
            return []
        try:
            data = self._sanitize_planner_json(raw)
            if isinstance(data, dict) and isinstance(data.get('selectors'), list):
                return [str(x).strip() for x in data['selectors'] if str(x).strip()]
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
        # Fallback: try extracting a JSON array manually
        try:
            import re
            m = re.search(r"\[(?:\s*['\"][^][]+['\"]\s*,?\s*)+]", raw)
            if m:
                arr = json.loads(m.group(0))
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
        return []

    def _collect_element_candidates_for_llm(self, target: Optional[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            if not self.last_snapshot:
                return out
            for el in (getattr(self.last_snapshot, 'elements', []) or [])[:120]:
                try:
                    out.append({
                        'tag': getattr(el, 'tag_name', ''),
                        'text': (getattr(el, 'text', '') or '')[:120],
                        'attrs': {k: v for k, v in (getattr(el, 'attributes', {}) or {}).items() if k in ['id','name','placeholder','aria-label','data-testid','data-test','title']},
                        'alt_selectors': list((getattr(el, 'alt_selectors', []) or [])[:3]),
                        'css': list((getattr(el, 'css_selectors', []) or [])[:2]),
                        'xpath': list((getattr(el, 'xpath_selectors', []) or [])[:1]),
                    })
                except Exception:
                    continue
        except Exception:
            return out
        return out

    # --- Role selector parsing for Selenium fallbacks ---
    def _parse_role_selector(self, s: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            # role=button[name='Login'] or role=link[name="Sign in"]
            raw = s.strip()[len('role='):]
            role = raw
            name = None
            if '[' in raw and raw.endswith(']'):
                role, rest = raw.split('[', 1)
                role = role.strip()
                rest = rest[:-1]
                if rest.startswith('name='):
                    v = rest[len('name='):].strip()
                    if v.startswith("'") and v.endswith("'"):
                        name = v[1:-1]
                    elif v.startswith('"') and v.endswith('"'):
                        name = v[1:-1]
                    else:
                        name = v
            else:
                role = raw.strip()
            return role or None, name
        except Exception:
            return None, None

    def _role_xpath(self, role: str, name: Optional[str]) -> str:
        role = (role or '').strip().lower()
        name_pred = ''
        if name:
            val = json.dumps(name)
            name_pred = f"[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), {json.dumps(name.lower())}) or contains(@aria-label, {val}) or contains(@title, {val})]"
        if role in ('button',):
            return f"//(button|*[@role='button']){name_pred}"
        if role in ('link','a'):
            return f"//(a|*[@role='link']){name_pred}"
        if role in ('textbox','input'):
            return f"//(input|textarea|*[@role='textbox']){name_pred}"
        if role in ('combobox','select'):
            return f"//(select|*[@role='combobox']){name_pred}"
        # Generic fallback by aria role
        if role:
            return f"//*[@role={json.dumps(role)}]{name_pred}"
        # No role; match by name only
        return f"//*[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), {json.dumps((name or '').lower())})]"

    async def plan(self, url: str, instruction: str) -> List[AgentStep]:
        """Create a high-level plan of steps using Azure OpenAI with DOM context. Fallback to heuristics if Azure is unavailable."""
        steps: List[AgentStep] = []
        try:
            # Ensure we have a snapshot to inform planning
            if self.dom and (self.playwright_ready or self.selenium_ready):
                await self._open_url(url)
            else:
                await self.capture_snapshot(url)
        except Exception:
            pass

        # Always start with navigate unless planner returns one
        def ensure_navigate_prefix(planned: List[AgentStep]) -> List[AgentStep]:
            if planned and planned[0].action == 'navigate':
                return planned
            nav = AgentStep(idx=1, action='navigate', target=url)
            for i, s in enumerate(planned, start=2):
                s.idx = i
            return [nav] + planned

        # Azure-powered planning
        if self.azure and getattr(self.azure, 'is_configured', lambda: False)():
            try:
                # Prepare light DOM context
                ctx: Dict[str, Any] = {
                    'url': url,
                    'title': getattr(self.last_snapshot, 'title', '') if self.last_snapshot else '',
                    'top_elements': []
                }
                try:
                    elems = []
                    if self.last_snapshot:
                        for el in getattr(self.last_snapshot, 'elements', [])[:60]:
                            elems.append({
                                'tag': getattr(el, 'tag_name', ''),
                                'text': (getattr(el, 'text', '') or '')[:80],
                                'attrs': {k: v for k, v in (getattr(el, 'attributes', {}) or {}).items() if k in ['id','name','placeholder','aria-label','data-testid','data-test','title']}
                            })
                    ctx['top_elements'] = elems[:60]
                except Exception:
                    pass

                system = {
                    'role': 'system',
                    'content': (
                        "You are a senior test planner that outputs only strict JSON. Given a starting URL, a user instruction, and a light DOM context (tags, texts, key attributes), "
                        "return a concise step plan to accomplish the task using web UI actions. Allowed actions: navigate|click|input|select|wait_for|assert_text. "
                        "Rules: 1) Prefer accessible clicks (role=button/link[name='...'] or visible text) 2) For inputs, use field labels/placeholders for target 3) Include short waits where appropriate "
                        "4) Return JSON with {\"steps\":[{\"action\":...,\"target\":...,\"value\":...,\"wait_for\":...}, ...]} and nothing else. Keep steps <= 20."
                    )
                }
                user = {
                    'role': 'user',
                    'content': json.dumps({
                        'start_url': url,
                        'instruction': instruction,
                        'dom_context': ctx
                    }, ensure_ascii=False)
                }
                resp = self.azure.chat_completion_create(messages=[system, user], temperature=0.2, max_tokens=900)
                text = resp.get('choices', [{}])[0].get('message', {}).get('content', '') if resp else ''
                data = self._sanitize_planner_json(text)
                raw_steps = data.get('steps', []) if isinstance(data, dict) else []
                idx = 1
                for rs in raw_steps:
                    try:
                        action = str(rs.get('action', '')).strip().lower()
                        target = rs.get('target')
                        value = rs.get('value')
                        wait_for = rs.get('wait_for')
                        if action not in ('navigate','click','input','select','wait_for','assert_text'):
                            continue
                        steps.append(AgentStep(idx=idx, action=action, target=target, value=value, wait_for=wait_for))
                        idx += 1
                    except Exception:
                        continue
                steps = ensure_navigate_prefix(steps)
                # If planner gave nothing, fall back below
            except Exception as e:
                self.bus.log(f"Planner failed, falling back: {e}")
                steps = []

        # Heuristic fallback plan when Azure is unavailable or failed
        if not steps:
            # Base navigate + minimal wait
            steps = [
                AgentStep(idx=1, action='navigate', target=url),
                AgentStep(idx=2, action='wait_for', target=None, wait_for='body')
            ]
            # Add heuristic actions inferred from instruction
            inferred = self._heuristic_steps(instruction)
            if inferred:
                # reindex inferred steps after the first two
                for i, s in enumerate(inferred, start=3):
                    s.idx = i
                steps.extend(inferred)
            # naive assert if prompt contains a quoted string
            try:
                import re
                m = re.search(r"'([^']{3,})'|\"([^\"]{3,})\"", instruction)
                txt = (m.group(1) or m.group(2)) if m else None
                if txt:
                    steps.append(AgentStep(idx=len(steps)+1, action='assert_text', target=txt))
            except Exception:
                pass
        # Store last snapshot URL as current
        try:
            if self.last_snapshot and getattr(self.last_snapshot, 'url', None):
                self.current_url = self.last_snapshot.url
        except Exception:
            pass
        return steps

    def _rank_candidates(self, action: str, primary: Optional[str], qualifiers: Optional[List[str]], provided: List[str]) -> List[str]:
        label = (primary or '').strip() or None
        domain = self._get_domain(self.current_url)
        result: List[str] = []
        seen = set()
        def add_many(items: List[str]):
            for s in items:
                s2 = (s or '').strip()
                if not s2:
                    continue
                if s2 not in seen:
                    result.append(s2)
                    seen.add(s2)
        # 1) Explicit selectors from target text
        add_many(self._literal_selectors_from_target(primary))
        # 2) Memory of successful selectors for this label/domain
        add_many(self._get_memory_selectors(domain, label))
        # 3) DOM-derived candidates if we have a snapshot and accurate mode
        prefer = 'click'
        if action == 'input':
            prefer = 'input'
        elif action == 'select':
            prefer = 'select'
        if self.analyze_first:
            add_many(self._selectors_from_snapshot(primary, prefer=prefer))
        # 4) Scoped XPath based on qualifiers for click actions
        if action == 'click' and primary and qualifiers:
            xps = self._scoped_xpath_candidates(primary, qualifiers)
            add_many([f"{xp}" for xp in xps])
        # 5) Input-specific fallbacks when typing
        if action == 'input':
            add_many(self._input_specific_candidates(primary, None))
        # 6) Provided selectors from planner/steps
        add_many(provided)
        # 7) Heuristic alternates from target
        add_many(self._alternate_selectors_from_target(primary))
        # 8) Optional Azure ranking to reorder top N (keep pool stable)
        if self.use_azure_ranking:
            try:
                ranked = self._azure_suggest_locators(primary, action, result, qualifiers or [])
                if ranked:
                    # Merge ranked at front preserving others
                    ranked_set = set(ranked)
                    tail = [s for s in result if s not in ranked_set]
                    result = ranked + tail
            except Exception:
                pass
        # 9) Cap to attempt_top_k if fast mode or generally to avoid thrashing
        max_k = self.attempt_top_k
        return result[:max_k]

    async def _try_selectors(self, action: str, selectors: List[str], value: Optional[str]) -> Tuple[bool, List[str]]:
        attempted: List[str] = []
        if not self.dom:
            return False, attempted
        self.last_used_selector = None
        total_retries = max(1, self.retries)
        # Limit to top-K attempts explicitly
        selectors = (selectors or [])[: self.attempt_top_k]
        # Try each selector across engines with retries
        for idx, sel in enumerate(selectors):
            s = sel.strip()
            if not s:
                continue
            is_top_rank = idx <= 1  # top 2 candidates
            # execute with retry loop
            for attempt in range(total_retries):
                if s not in attempted:
                    attempted.append(s)
                # Dynamic timeouts: extend on last retry for top-ranked candidates
                is_last_try = (attempt == total_retries - 1)
                base_wait = 1500
                long_wait = 5000
                wait_timeout = long_wait if (is_last_try and is_top_rank) else base_wait
                click_timeout = 5000 if (is_last_try and is_top_rank) else 3000
                select_timeout = 2500 if (is_last_try and is_top_rank) else 1800
                input_timeout = 2200 if (is_last_try and is_top_rank) else 1800
                try:
                    # Playwright path
                    if self.playwright_ready and self.dom.playwright_page:
                        page = self.dom.playwright_page
                        self.bus.log(f"Attempt {attempt+1}: {action} via Playwright using selector: {s}")
                        locator = None
                        s_kind = 'css'
                        # Smart locator routing
                        if s.startswith('text='):
                            locator = page.get_by_text(s[len('text='):])
                            s_kind = 'text'
                        elif s.startswith('label='):
                            locator = page.get_by_label(s[len('label='):])
                            s_kind = 'label'
                        elif s.startswith('placeholder='):
                            locator = page.get_by_placeholder(s[len('placeholder='):])
                            s_kind = 'placeholder'
                        elif s.startswith('testid='):
                            locator = page.get_by_test_id(s[len('testid='):])
                            s_kind = 'testid'
                        elif s.startswith('role='):
                            # Prefer Playwright selector engine for complex role expressions
                            locator = page.locator(s)
                            s_kind = 'role'
                        elif s.startswith('//') or s.startswith('/'):
                            locator = page.locator(f"xpath={s}")
                            s_kind = 'xpath'
                        else:
                            locator = page.locator(s)
                            s_kind = 'css'

                        # Disambiguation and scoping
                        loc = locator
                        try:
                            cnt = await locator.count()
                        except Exception:
                            cnt = 1
                        if cnt > 1 and s_kind in ('css','xpath'):
                            nav_scopes = [
                                'header, nav, [role="navigation"], #header, .header, [id*="header"], [class*="header"], [id*="nav"], [class*="nav"], #hosting2, [id*="hosting"], [class*="hosting"], [id*="menu"], [class*="menu"]'
                            ]
                            narrowed = None
                            for scope in nav_scopes:
                                try:
                                    container = page.locator(scope)
                                    if await container.count() > 0:
                                        scoped = container.locator(s if s_kind=='css' else f"xpath={s}")
                                        if await scoped.count() > 0:
                                            narrowed = scoped
                                            break
                                except Exception:
                                    continue
                            if narrowed is not None:
                                loc = narrowed
                                try:
                                    cnt = await loc.count()
                                except Exception:
                                    cnt = 1
                        if cnt > 1:
                            chosen = None
                            max_try = min(4, cnt)
                            for i in range(max_try):
                                candidate = loc.nth(i)
                                try:
                                    # Allow longer mount time for top candidates on last retry
                                    cand_timeout = wait_timeout if (is_last_try and is_top_rank) else min(wait_timeout, 1200)
                                    await candidate.wait_for(state='visible', timeout=cand_timeout)
                                    chosen = candidate
                                    break
                                except Exception:
                                    continue
                            loc = chosen if chosen is not None else loc.first
                        else:
                            loc = loc.first

                        try:
                            await loc.scroll_into_view_if_needed(timeout=800)
                        except Exception:
                            pass

                        # If element might still be mounting, first ensure it's attached before requiring visibility
                        if is_last_try and is_top_rank:
                            try:
                                await loc.wait_for(state='attached', timeout=min(800, max(300, wait_timeout//3)))
                            except Exception:
                                pass

                        if action == 'click':
                            await loc.wait_for(state='visible', timeout=wait_timeout)
                            nav_happened = False
                            before_url = None
                            try:
                                before_url = page.url
                            except Exception:
                                before_url = None
                            # Try to infer if click is likely to navigate
                            nav_expected = False
                            try:
                                nav_expected = await loc.evaluate("(el) => (el.tagName && el.tagName.toLowerCase()==='a' && !!el.getAttribute('href')) || (el.tagName && el.tagName.toLowerCase()==='button' && (el.type==='submit')) || el.getAttribute('role')==='link'", timeout=800)
                            except Exception:
                                nav_expected = False
                            try:
                                if nav_expected:
                                    await loc.click(timeout=click_timeout)
                                    try:
                                        await page.wait_for_load_state('domcontentloaded', timeout=min(3000, click_timeout))
                                    except Exception:
                                        pass
                                else:
                                    await loc.click(no_wait_after=True, timeout=click_timeout)
                                # After click, wait a brief moment or for specific selector
                                if self.post_click_wait_selector:
                                    try:
                                        await self._wait_for(self.post_click_wait_selector, timeout_ms=self.post_click_wait_timeout_ms)
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        await page.wait_for_timeout(160)
                                    except Exception:
                                        pass
                                # Determine if navigation happened via URL change
                                try:
                                    if before_url is not None and page.url != before_url:
                                        nav_happened = True
                                except Exception:
                                    pass
                            except Exception:
                                # As a fallback, try a direct click without waiting
                                try:
                                    await loc.click(no_wait_after=True, timeout=click_timeout)
                                except Exception:
                                    raise
                            # Stabilize network briefly for SPA
                            try:
                                await page.wait_for_load_state('networkidle', timeout=1200)
                            except Exception:
                                pass
                            try:
                                self.current_url = page.url
                            except Exception:
                                pass
                            self.last_used_selector = s
                            return True, attempted
                        elif action == 'input':
                            await loc.wait_for(state='visible', timeout=input_timeout)
                            # Special handling for inputs: focus and set value directly
                            try:
                                await loc.focus(timeout=300)
                                await loc.fill(value or "", timeout=input_timeout)
                                await page.wait_for_timeout(40)
                                self.last_used_selector = s
                                return True, attempted
                            except Exception:
                                # Fallback to click and type
                                try:
                                    await loc.click(timeout=click_timeout)
                                    await page.keyboard.type(value or "", delay=50)
                                    await page.wait_for_timeout(40)
                                    self.last_used_selector = s
                                    return True, attempted
                                except Exception:
                                    # Last-resort: set value via script
                                    try:
                                        await loc.evaluate("(el, v) => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); }", value or "")
                                        self.last_used_selector = s
                                        return True, attempted
                                    except Exception:
                                        return False, attempted
                        elif action == 'select':
                            await loc.wait_for(state='visible', timeout=wait_timeout)
                            try:
                                await loc.select_option(value or "", timeout=select_timeout)
                            except Exception:
                                await loc.fill(value or "", timeout=select_timeout)
                            await page.wait_for_timeout(40)
                            self.last_used_selector = s
                            return True, attempted
                        elif action == 'wait_for':
                            timeout_total = max(wait_timeout, int(self.backoff_ms) * 8)
                            try:
                                # Prefer attached first when extending waits on last try for top candidates
                                if is_last_try and is_top_rank:
                                    try:
                                        await loc.wait_for(state='attached', timeout=min(800, max(300, wait_timeout//3)))
                                    except Exception:
                                        pass
                                await loc.wait_for(state='visible', timeout=timeout_total)
                            except Exception:
                                try:
                                    if s_kind == 'xpath':
                                        await page.wait_for_selector(f"xpath={s}", state='visible', timeout=wait_timeout)
                                    else:
                                        await page.wait_for_selector(s, state='visible', timeout=wait_timeout)
                                except Exception:
                                    raise
                            self.last_used_selector = s
                            return True, attempted
                    # Selenium path
                    if self.selenium_ready and self.dom.driver:
                        from selenium.webdriver.common.by import By  # type: ignore
                        from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
                        from selenium.webdriver.support import expected_conditions as EC  # type: ignore
                        driver = self.dom.driver
                        self.bus.log(f"Attempt {attempt+1}: {action} via Selenium using selector: {s}")
                        by, expr = None, None
                        if s.startswith('//'):
                            by, expr = By.XPATH, s
                        elif s.startswith('text='):
                            by, expr = By.XPATH, f"//*[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), {json.dumps(s[5:].lower())})]"
                        elif s.startswith('placeholder='):
                            by, expr = By.CSS_SELECTOR, f"[placeholder='{s[len('placeholder='):] }']"
                        elif s.startswith('label='):
                            label = s[len('label='):]
                            for_id = None
                            try:
                                for_id = driver.execute_script(
                                    "return (function(lbl){const m=Array.from(document.querySelectorAll('label')).find(l=> (l.innerText||l.textContent||'').trim()===lbl); return m? (m.getAttribute('for')||null) : null;})(arguments[0]);",
                                    label,
                                )
                            except Exception:
                                for_id = None
                            if for_id:
                                by, expr = By.CSS_SELECTOR, f"#{for_id}"
                            else:
                                by, expr = By.XPATH, f"//label[normalize-space()={json.dumps(label)}]//*[self::input or self::textarea or self::select]"
                        elif s.startswith('testid='):
                            val = s[len('testid='):]
                            by, expr = By.CSS_SELECTOR, f"[data-testid='{val}'], [data-test='{val}']"
                        elif s.startswith('role='):
                            role, name = self._parse_role_selector(s)
                            by, expr = By.XPATH, self._role_xpath(role or '', name)
                        else:
                            by, expr = By.CSS_SELECTOR, s
                        try:
                            WebDriverWait(driver, wait_timeout/1000).until(EC.presence_of_element_located((by, expr)))
                        except Exception:
                            pass
                        el = driver.find_element(by, expr)
                        try:
                            driver.execute_script("arguments[0].scrollIntoView({block:'center',inline:'center'});", el)
                        except Exception:
                            pass
                        if action == 'click':
                            try:
                                WebDriverWait(driver, wait_timeout/1000).until(EC.element_to_be_clickable((by, expr)))
                            except Exception:
                                pass
                            el.click()
                            time.sleep(0.15)
                            # Optional stabilization after click
                            if self.post_click_wait_selector:
                                try:
                                    by2, expr2 = By.CSS_SELECTOR, self.post_click_wait_selector
                                    if self.post_click_wait_selector.startswith('//'):
                                        by2, expr2 = By.XPATH, self.post_click_wait_selector
                                    WebDriverWait(driver, self.post_click_wait_timeout_ms/1000.0).until(EC.visibility_of_element_located((by2, expr2)))
                                except Exception:
                                    pass
                            try:
                                self.current_url = driver.current_url
                            except Exception:
                                pass
                            self.last_used_selector = s
                            return True, attempted
                        elif action == 'input':
                            try:
                                el.clear()
                            except Exception:
                                pass
                            el.send_keys(value or "")
                            time.sleep(0.05)
                            self.last_used_selector = s
                            return True, attempted
                        elif action == 'select':
                            try:
                                from selenium.webdriver.support.ui import Select  # type: ignore
                                Select(el).select_by_visible_text(value or "")
                            except Exception:
                                el.send_keys(value or "")
                            time.sleep(0.05)
                            self.last_used_selector = s
                            return True, attempted
                except Exception as ex:
                    self.bus.log(f"Retry due to: {ex}")
                    if self.playwright_ready and self.dom.playwright_page:
                        try:
                            await self.dom.playwright_page.wait_for_timeout(self.backoff_ms)
                        except Exception:
                            pass
                    else:
                        time.sleep(self.backoff_ms/1000.0)
                    continue
        # Last-resort heuristics when all candidates failed
        try:
            if action == 'input':
                # Playwright heuristics
                if self.playwright_ready and self.dom and self.dom.playwright_page:
                    page = self.dom.playwright_page
                    # 1) Focused field
                    try:
                        foc = page.locator("input:focus, textarea:focus")
                        if await foc.count() > 0:
                            await foc.first.fill(value or "", timeout=1200)
                            attempted.append(":focus")
                            self.last_used_selector = ":focus"
                            return True, attempted
                    except Exception:
                        pass
                    # 2) Semantic guesses by value/target
                    fallback_queries: List[str] = []
                    # Email/password/search by value
                    v = (value or '')
                    if '@' in v:
                        fallback_queries.extend(["input[type='email']", "input[name*='email']", "input[id*='email']"])
                    # Try common inputs
                    fallback_queries.extend([
                        "input[type='text']",
                        "input:not([type]), textarea",
                        "input[type='search']",
                        "textarea",
                    ])
                    # Try and fill first visible match
                    for q in fallback_queries:
                        try:
                            loc = page.locator(q)
                            if await loc.count() == 0:
                                continue
                            cand = loc.first
                            try:
                                await cand.wait_for(state='visible', timeout=1000)
                            except Exception:
                                continue
                            try:
                                await cand.fill(value or "", timeout=1500)
                            except Exception:
                                try:
                                    await cand.click(timeout=600)
                                    await page.keyboard.type(value or "")
                                except Exception:
                                    # programmatically set value
                                    try:
                                        vv = value or ""
                                        await cand.evaluate("(el, v) => { try { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); } catch(e){} }", vv)
                                    except Exception:
                                        continue
                            attempted.append(q)
                            self.last_used_selector = q
                            return True, attempted
                        except Exception:
                            continue
                # Selenium heuristics
                if self.selenium_ready and self.dom and self.dom.driver:
                    from selenium.webdriver.common.by import By  # type: ignore
                    from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
                    from selenium.webdriver.support import expected_conditions as EC  # type: ignore
                    driver = self.dom.driver
                    # 1) Focused element
                    try:
                        el = driver.switch_to.active_element
                        tag = (el.tag_name or '').lower()
                        if tag in ('input','textarea'):
                            try:
                                el.clear()
                            except Exception:
                                pass
                            el.send_keys(value or "")
                            attempted.append(":focus")
                            self.last_used_selector = ":focus"
                            return True, attempted
                    except Exception:
                        pass
                    # 2) Common queries
                    queries = [
                        (By.CSS_SELECTOR, "input[type='email']"),
                        (By.CSS_SELECTOR, "input[name*='email']"),
                        (By.CSS_SELECTOR, "input[type='text']"),
                        (By.CSS_SELECTOR, "input:not([type])"),
                        (By.CSS_SELECTOR, "textarea"),
                        (By.CSS_SELECTOR, "input[type='search']"),
                    ]
                    for by, expr in queries:
                        try:
                            els = driver.find_elements(by, expr)
                            if not els:
                                continue
                            el = els[0]
                            try:
                                driver.execute_script("arguments[0].scrollIntoView({block:'center',inline:'center'});", el)
                            except Exception:
                                pass
                            try:
                                WebDriverWait(driver, 1.2).until(EC.visibility_of(el))
                            except Exception:
                                pass
                            try:
                                el.clear()
                            except Exception:
                                pass
                            try:
                                el.send_keys(value or "")
                            except Exception:
                                try:
                                    driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input', {bubbles:true})); arguments[0].dispatchEvent(new Event('change', {bubbles:true}));", el, value or "")
                                except Exception:
                                    continue
                            attempted.append(expr)
                            self.last_used_selector = expr
                            return True, attempted
                        except Exception:
                            continue
        except Exception:
            pass
        return False, attempted

    async def execute(self, url: str, steps: List[AgentStep]) -> List[StepResult]:
        results: List[StepResult] = []
        if not steps:
            return results
        # Ensure initial snapshot when Accurate mode is enabled
        try:
            if self.analyze_first and (self.playwright_ready or self.selenium_ready) and (self.current_url or url):
                await self.capture_snapshot(self.current_url or url)
        except Exception:
            pass
        for s in steps:
            start = time.time()
            attempted: List[str] = []
            screenshot_path: Optional[str] = None
            try:
                self.bus.log(f"Step {s.idx}: {self._narrate_step(s)}")
                # Optional DOM refresh before each step for accuracy
                if self.dom_refresh_per_step and (self.playwright_ready or self.selenium_ready) and (self.current_url):
                    try:
                        await self.capture_snapshot(self.current_url)
                    except Exception:
                        pass
                # Navigate action
                if s.action == 'navigate':
                    if self.dom and (self.playwright_ready or self.selenium_ready):
                        tgt = (s.target or "").strip()
                        ok = True
                        if tgt and self._looks_like_url(tgt):
                            dest = self._resolve_url(self.current_url or url, tgt)
                            await self._open_url(dest)
                        else:
                            primary, quals = self._parse_click_target(tgt)
                            candidates = self._rank_candidates('click', primary, quals, s.selectors or [])
                            ok, attempted = await self._try_selectors('click', candidates, None)
                            if ok:
                                self._remember_selector(self._get_domain(self.current_url or url), primary, self.last_used_selector)
                        if ok and self.screenshot_on_success:
                            screenshot_path = await self._capture_step_screenshot(s.idx, 'success')
                        results.append(StepResult(s, 'success' if ok else 'failed', 'Navigated', attempted, screenshot_path, time.time()-start))
                        # Refresh snapshot and current URL after navigation attempt
                        try:
                            if self.playwright_ready and self.dom.playwright_page:
                                self.current_url = self.dom.playwright_page.url
                            elif self.selenium_ready and self.dom.driver:
                                self.current_url = self.dom.driver.current_url
                        except Exception:
                            pass
                        if self.current_url:
                            try:
                                await self.capture_snapshot(self.current_url)
                            except Exception:
                                pass
                        continue

                # Assert text action
                if s.action == 'assert_text' and (s.target or s.wait_for):
                    text = s.target or s.wait_for or ''
                    ok = await self._assert_text(text)
                    if ok and self.screenshot_on_success:
                        screenshot_path = await self._capture_step_screenshot(s.idx, 'success')
                    if not ok and self.screenshot_on_failure:
                        screenshot_path = await self._capture_step_screenshot(s.idx, 'failed')
                    results.append(StepResult(s, 'success' if ok else 'failed', 'Assertion ' + ('passed' if ok else 'failed'), attempted, screenshot_path, time.time()-start))
                    continue

                # For other actions, DOM-first candidate ranking
                if s.action == 'click':
                    primary, quals = self._parse_click_target(s.target)
                else:
                    primary, quals = (s.target or ''), []
                candidates = self._rank_candidates(s.action, primary, quals, s.selectors or [])
                ok, attempted = await self._try_selectors(s.action, candidates, s.value)
                if ok and s.action in ('click','input','select'):
                    self._remember_selector(self._get_domain(self.current_url or url), primary, self.last_used_selector)
                # Try a final wait if specified
                if s.action == 'wait_for' and (s.wait_for or (s.selectors and len(s.selectors) > 0)):
                    target_wait = s.selectors[0] if s.selectors else (s.wait_for or '')
                    ok = await self._wait_for(target_wait)

                status = 'success' if ok else 'skipped'
                detail = 'Executed' if ok else 'Could not locate element; bypassed.'

                if ok and self.screenshot_on_success:
                    screenshot_path = await self._capture_step_screenshot(s.idx, 'success')
                if not ok and self.screenshot_on_failure:
                    screenshot_path = await self._capture_step_screenshot(s.idx, 'failed')

                results.append(StepResult(s, status, detail, attempted, screenshot_path, time.time()-start))

                # After successful interactions, refresh snapshot quickly for next steps
                if ok:
                    try:
                        if self.playwright_ready and self.dom.playwright_page:
                            # Update current URL from page state
                            try:
                                self.current_url = self.dom.playwright_page.url
                            except Exception:
                                pass
                            await self.dom.playwright_page.wait_for_timeout(120)
                        elif self.selenium_ready and self.dom.driver:
                            try:
                                self.current_url = self.dom.driver.current_url
                            except Exception:
                                pass
                        if self.current_url:
                            await self.capture_snapshot(self.current_url)
                    except Exception:
                        pass

                # Post-wait if needed
                if ok and s.wait_for:
                    _ = await self._wait_for(s.wait_for)
            except Exception as e:
                if self.screenshot_on_failure:
                    try:
                        screenshot_path = await self._capture_step_screenshot(s.idx, 'failed')
                    except Exception:
                        screenshot_path = None
                results.append(StepResult(s, 'failed', str(e), attempted, screenshot_path, time.time()-start))
                continue
        return results

    def to_test_case(self, name: str, base_url: str, results: List[StepResult]) -> Dict[str, Any]:
        steps_str: List[str] = []
        for r in results:
            act = r.step.action
            tgt = r.step.target or ''
            val = r.step.value or ''
            if act == 'navigate':
                steps_str.append(f"Open Browser    {tgt or base_url}    ${{BROWSER}}")
                steps_str.append("Maximize Browser Window")
            elif act == 'click':
                # Prefer last attempted selector when available
                sel = (r.attempted_selectors[0] if r.attempted_selectors else (r.step.selectors[0] if r.step.selectors else 'body'))
                steps_str.append(f"Click Element    {sel}")
            elif act == 'input':
                sel = (r.attempted_selectors[0] if r.attempted_selectors else (r.step.selectors[0] if r.step.selectors else 'input'))
                steps_str.append(f"Input Text    {sel}    {val}")
            elif act == 'wait_for':
                sel = r.step.wait_for or (r.step.selectors[0] if r.step.selectors else 'body')
                steps_str.append(f"Wait Until Element Is Visible    {sel}    ${{TIMEOUT}}")
            elif act == 'assert_text':
                steps_str.append(f"Page Should Contain    {tgt}")
        tc = {
            'name_robot': name.replace(' ', '_'),
            'description': 'Generated by Browser Agent',
            '_automatable': True,
            'steps': steps_str,
            'expected': '',
            'preconditions': f"Start at {base_url}",
            'postconditions': 'Close browser',
            'priority': 'P2',
            'severity': 'Medium',
            'brand': ''
        }
        return tc


def show_ui():  # Entry point consumed by main_ui
    st.markdown("### 🧭 Browser Agent — Agentic AI for Smart Web Navigation")
    st.info("Describe the journey in plain English. The agent will plan actions, traverse the UI, try alternates when stuck, and log every step.")

    with st.form("browser_agent_form", clear_on_submit=False):
        url = st.text_input("Start URL", value="https://example.com", help="Where the journey begins.")
        prompt = st.text_area("What should the agent do?", height=120, placeholder="Example: Login with test user, go to account settings, update profile name to 'QA Bot', and save.")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            headless = st.checkbox("Headless", value=False)
            screenshot_on_success = st.checkbox("Screenshot on success", value=False)
        with col2:
            prefer_playwright = st.checkbox("Prefer Playwright", value=True)
            screenshot_on_failure = st.checkbox("Screenshot on failure", value=True)
        with col3:
            plan_only = st.checkbox("Plan only (no execution)", value=False)
            retries = st.number_input("Retries/step", min_value=1, max_value=5, value=2, step=1)
        with col4:
            max_steps = st.number_input("Max steps", min_value=1, max_value=100, value=20, step=1)
            backoff_ms = st.number_input("Retry backoff (ms)", min_value=50, max_value=5000, value=150, step=50)
        with col5:
            fast_mode = st.checkbox("Fast mode", value=True)
            # Repurpose probe top N as attempt top K
            attempt_top_k = st.number_input("Max candidates/step", min_value=1, max_value=20, value=6, step=1)
        with col6:
            analyze_first = st.checkbox("Accurate mode (analyze DOM first)", value=True)
            dom_refresh_per_step = st.checkbox("Refresh DOM each step", value=True)
            use_azure_ranking = st.checkbox("Use Azure ranking", value=False, disabled=not AZURE_AVAILABLE)
        submitted = st.form_submit_button("Run Browser Agent")

    log_placeholder = st.empty()
    bus = SimpleEventBus(log_placeholder)

    # Persistent state for last run
    if "browser_agent_results" not in st.session_state:
        st.session_state.browser_agent_results = []
        st.session_state.browser_agent_steps = []
        st.session_state.browser_agent_url = url
        st.session_state.browser_agent_robot_file = None

    if submitted:
        if not url.strip() or not prompt.strip():
            st.error("Please provide both URL and instruction prompt.")
            return
        # Initialize Azure client if available
        azure = AzureOpenAIClient() if AZURE_AVAILABLE else None
        agent = BrowserAgentCore(headless=headless, prefer_playwright=prefer_playwright, azure_client=azure, event_bus=bus,
                                 retries=int(retries), backoff_ms=int(backoff_ms),
                                 screenshot_on_success=bool(screenshot_on_success), screenshot_on_failure=bool(screenshot_on_failure),
                                 fast_mode=bool(fast_mode), concurrent_probe_top_n=int(3),
                                 analyze_first=bool(analyze_first), dom_refresh_per_step=bool(dom_refresh_per_step),
                                 attempt_top_k=int(attempt_top_k), use_azure_ranking=bool(use_azure_ranking))
        # Update session URL for downstream export
        st.session_state.browser_agent_url = url

        async def run():
            try:
                await agent.initialize()
                steps = await agent.plan(url, prompt)
                # Trim to max steps
                steps = steps[:max_steps]
                st.session_state.browser_agent_steps = [asdict(s) for s in steps]
                bus.log(f"Planned {len(steps)} step(s).")
                results: List[StepResult] = []
                if not plan_only:
                    results = await agent.execute(url, steps)
                else:
                    bus.log("Plan-only mode. Skipping execution.")
                st.session_state.browser_agent_results = [asdict(r) for r in results]
                # Persist logs
                try:
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    log_path = os.path.join(os.getcwd(), "generated_tests", f"browser_agent_{ts}.log")
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, 'w', encoding='utf-8') as lf:
                        lf.write("\n".join(bus.lines))
                    st.session_state.browser_agent_log_file = log_path
                    bus.log(f"Logs saved to: {log_path}")
                except Exception as e:
                    bus.log(f"Failed to save logs: {e}")
                # Notifications
                if NOTIFICATIONS_AVAILABLE:
                    if results and all(r.status in ("success", "skipped") for r in results):
                        notifications.handle_execution_result("browser_agent", True, "Browser Agent run completed.")
                    else:
                        notifications.handle_execution_result("browser_agent", True, "Browser Agent plan generated.")
            except Exception as e:
                if NOTIFICATIONS_AVAILABLE:
                    notifications.handle_execution_result("browser_agent", False, str(e))
                bus.log(f"Run failed: {e}")
            finally:
                try:
                    await agent.close()
                except Exception:
                    pass

        asyncio.run(run())

    # Show planned steps
    if st.session_state.browser_agent_steps:
        st.subheader("Planned Steps")
        st.dataframe(st.session_state.browser_agent_steps, use_container_width=True)

    # Show results
    if st.session_state.browser_agent_results:
        st.subheader("Execution Results")
        df = st.session_state.browser_agent_results
        st.dataframe(df, use_container_width=True)

    # Documentation and export
    if st.session_state.browser_agent_steps:
        st.markdown("---")
        st.subheader("Document the Flow")
        tc_name = st.text_input("Test Case Name", value="Browser_Agent_Flow")
        colx, coly = st.columns([1,1])
        with colx:
            if st.button("Generate Test Case Table"):
                # Build a table view
                rows = []
                # Helper to narrate
                def narr(sdict: Dict[str, Any]) -> str:
                    try:
                        return BrowserAgentCore()._narrate_step(AgentStep(**sdict))
                    except Exception:
                        return ""
                for r in st.session_state.browser_agent_results or []:
                    rows.append({
                        "Step": r["step"]["idx"],
                        "Action": r["step"]["action"],
                        "Target": r["step"].get("target"),
                        "Value": r["step"].get("value"),
                        "Status": r["status"],
                        "Detail": r["detail"],
                        "Narration": narr(r["step"]),
                        "Screenshot": r.get("screenshot")
                    })
                if not rows:  # plan only
                    for s in st.session_state.browser_agent_steps:
                        rows.append({
                            "Step": s["idx"],
                            "Action": s["action"],
                            "Target": s.get("target"),
                            "Value": s.get("value"),
                            "Status": "planned",
                            "Detail": "",
                            "Narration": narr(s),
                            "Screenshot": ""
                        })
                st.session_state.browser_agent_table = rows
                # Persist table to CSV
                try:
                    import pandas as _pd
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    table_path = os.path.join(os.getcwd(), "generated_tests", f"browser_agent_table_{ts}.csv")
                    _pd.DataFrame(rows).to_csv(table_path, index=False)
                    st.session_state.browser_agent_table_file = table_path
                    st.success("Generated table below and saved CSV.")
                except Exception as e:
                    st.warning(f"Table CSV save failed: {e}")
        with coly:
            can_export = ROBOT_WRITER_AVAILABLE
            if st.button("Export to Robot Framework", disabled=not can_export):
                try:
                    agent_steps = [AgentStep(**s) for s in st.session_state.browser_agent_steps]
                    results: List[StepResult] = []
                    if st.session_state.browser_agent_results:
                        results = [StepResult(step=AgentStep(**r["step"]), status=r["status"], detail=r["detail"], attempted_selectors=r.get("attempted_selectors", []), screenshot=r.get("screenshot"), duration_sec=r.get("duration_sec", 0.0)) for r in st.session_state.browser_agent_results]
                    else:
                        # synthesize from plan
                        results = [StepResult(step=s, status="planned", detail="", attempted_selectors=s.selectors or [], screenshot=None, duration_sec=0.0) for s in agent_steps]
                    # Build test case and write
                    core = BrowserAgentCore()
                    tc = core.to_test_case(tc_name, st.session_state.browser_agent_url or "", results)
                    output_dir = os.path.join(os.getcwd(), "generated_tests")
                    os.makedirs(output_dir, exist_ok=True)
                    writer = RobotWriter(output_dir)
                    path = writer.write([tc], filename=f"{tc['name_robot']}.robot")
                    st.session_state.browser_agent_robot_file = path
                    st.success(f"Robot Framework file created: {path}")
                    try:
                        with open(path, 'rb') as f:
                            st.download_button("Download .robot", f, file_name=os.path.basename(path), mime="text/plain")
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Export failed: {e}")

    if st.session_state.get("browser_agent_table"):
        st.subheader("Test Case Table")
        st.dataframe(st.session_state["browser_agent_table"], use_container_width=True)
        # Download buttons for artifacts
        col_a, col_b = st.columns([1,1])
        with col_a:
            log_file = st.session_state.get("browser_agent_log_file")
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file, 'rb') as f:
                        st.download_button("Download Logs", f, file_name=os.path.basename(log_file), mime="text/plain")
                except Exception:
                    pass
        with col_b:
            table_file = st.session_state.get("browser_agent_table_file")
            if table_file and os.path.exists(table_file):
                try:
                    with open(table_file, 'rb') as f:
                        st.download_button("Download Table (CSV)", f, file_name=os.path.basename(table_file), mime="text/csv")
                except Exception:
                    pass

