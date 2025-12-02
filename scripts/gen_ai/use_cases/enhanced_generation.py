"""Enhanced generation workflow using EnhancedBrowserAutomationManager without modifying original large file.
Call: from enhanced_generation import analyze_and_generate_enhanced
"""
from typing import Tuple
from datetime import datetime
import os
import asyncio

try:
    from .test_pilot import TestCase, TestPilotEngine
    from .enhanced_browser_manager import EnhancedBrowserAutomationManager
    from .test_pilot_enhancements import generate_locator_file_from_captured
except Exception:
    from test_pilot import TestCase, TestPilotEngine
    from enhanced_browser_manager import EnhancedBrowserAutomationManager
    from test_pilot_enhancements import generate_locator_file_from_captured

async def analyze_and_generate_enhanced(engine: TestPilotEngine, test_case: TestCase, base_url: str, headless: bool = True) -> Tuple[bool, str, str, str]:
    """Run enhanced browser-based generation.
    Returns: success, summary/script content, suite_path, bug_report_content
    """
    browser_mgr = EnhancedBrowserAutomationManager(engine.azure_client)
    if not browser_mgr.initialize_browser(base_url, headless):
        return False, '', '', 'Browser init failed'
    for step in test_case.steps:
        await browser_mgr.execute_step_smartly(step, test_case)
    bug_report = await browser_mgr.generate_ai_bug_report()
    bug_report_path = os.path.join(engine.output_dir, f"bug_report_{test_case.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(bug_report_path, 'w') as f:
        f.write(bug_report)
    test_case.metadata['captured_locators'] = browser_mgr.captured_locators
    test_case.metadata['step_locators'] = browser_mgr.step_locators
    test_case.metadata['accessibility_issues'] = browser_mgr.accessibility_findings
    test_case.metadata['security_issues'] = browser_mgr.security_findings
    # Run AI step analysis if configured
    if engine.azure_client and engine.azure_client.is_configured():
        try:
            enhanced_success, enhanced_tc, _ = await engine.analyze_steps_with_ai(test_case)
            if enhanced_success:
                test_case = enhanced_tc
        except Exception:
            pass
    # Monkey patch locator generation using captured locators only
    locator_content = generate_locator_file_from_captured(test_case, os.getcwd())
    # Temporarily store pre-generated locator content so engine.generate_robot_script uses ours
    original_generate_locator_file = engine._generate_locator_file
    engine._generate_locator_file = lambda tc: locator_content
    success, summary, suite_path = engine.generate_robot_script(test_case, include_comments=True)
    engine._generate_locator_file = original_generate_locator_file
    return success, summary, suite_path, bug_report
