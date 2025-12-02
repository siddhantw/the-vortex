"""
Locator Reuse Manager - Smart Detection and Reuse of Existing Locators

This module scans the existing codebase for keywords, locators, and variables,
and provides intelligent matching and reuse capabilities for the TestPilot system.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger("LocatorReuseManager")


@dataclass
class ExistingLocator:
    """Represents an existing locator found in the codebase"""
    name: str
    value: str
    file_path: str
    line_number: int
    category: str  # e.g., 'button', 'input', 'link', 'dropdown'
    context: str  # e.g., 'login', 'checkout', 'navigation'


@dataclass
class ExistingKeyword:
    """Represents an existing Robot Framework keyword"""
    name: str
    file_path: str
    line_number: int
    arguments: List[str]
    documentation: str
    steps: List[str]
    category: str  # e.g., 'navigation', 'form_fill', 'verification'


@dataclass
class ExistingVariable:
    """Represents an existing variable"""
    name: str
    value: str
    file_path: str
    line_number: int
    type: str  # e.g., 'url', 'credential', 'test_data'


class LocatorReuseManager:
    """
    Intelligent manager for detecting and reusing existing test automation assets
    """

    def __init__(self, root_dir: str):
        """
        Initialize the manager with the root directory of the project

        Args:
            root_dir: Root directory of the test automation project
        """
        self.root_dir = root_dir
        self.tests_dir = os.path.join(root_dir, 'tests')
        self.locators_cache: Dict[str, ExistingLocator] = {}
        self.keywords_cache: Dict[str, ExistingKeyword] = {}
        self.variables_cache: Dict[str, ExistingVariable] = {}
        self._initialized = False

    def initialize(self) -> bool:
        """
        Scan the codebase and build cache of existing assets

        Returns:
            Success status
        """
        try:
            logger.info("🔍 Scanning codebase for existing locators, keywords, and variables...")

            # Scan locators
            self._scan_locators()

            # Scan keywords
            self._scan_keywords()

            # Scan variables
            self._scan_variables()

            self._initialized = True

            logger.info(f"✅ Found {len(self.locators_cache)} locators, "
                       f"{len(self.keywords_cache)} keywords, "
                       f"{len(self.variables_cache)} variables")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize LocatorReuseManager: {str(e)}")
            return False

    def _scan_locators(self):
        """Scan for Python locator files"""
        locators_dir = os.path.join(self.tests_dir, 'locators')

        if not os.path.exists(locators_dir):
            logger.warning(f"Locators directory not found: {locators_dir}")
            return

        for root, dirs, files in os.walk(locators_dir):
            for file in files:
                if file.endswith('_locators.py'):
                    file_path = os.path.join(root, file)
                    self._parse_locator_file(file_path)

    def _parse_locator_file(self, file_path: str):
        """Parse a Python locator file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                # Match pattern: variable_name_locator = "xpath/css/id value"
                match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*_locator)\s*=\s*["\']([^"\']+)["\']', line.strip())
                if match:
                    locator_name = match.group(1)
                    locator_value = match.group(2)

                    # Infer category and context from name
                    category = self._infer_locator_category(locator_name)
                    context = self._infer_locator_context(locator_name, file_path)

                    locator = ExistingLocator(
                        name=locator_name,
                        value=locator_value,
                        file_path=file_path,
                        line_number=line_num,
                        category=category,
                        context=context
                    )

                    self.locators_cache[locator_name] = locator

        except Exception as e:
            logger.debug(f"Could not parse locator file {file_path}: {str(e)}")

    def _scan_keywords(self):
        """Scan for Robot Framework keyword files"""
        keywords_dir = os.path.join(self.tests_dir, 'keywords')

        if not os.path.exists(keywords_dir):
            logger.warning(f"Keywords directory not found: {keywords_dir}")
            return

        for root, dirs, files in os.walk(keywords_dir):
            for file in files:
                if file.endswith('_keywords.robot') or file.endswith('.robot'):
                    file_path = os.path.join(root, file)
                    self._parse_keyword_file(file_path)

    def _parse_keyword_file(self, file_path: str):
        """Parse a Robot Framework keyword file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all keywords
            keyword_pattern = r'^([A-Z][A-Za-z0-9\s]+)\n((?:    \[.*?\]\n)*)((?:    .*\n)*)'
            matches = re.finditer(keyword_pattern, content, re.MULTILINE)

            for match in matches:
                keyword_name = match.group(1).strip()
                metadata_lines = match.group(2)
                step_lines = match.group(3)

                # Parse documentation
                doc_match = re.search(r'\[Documentation\]\s+(.+)', metadata_lines)
                documentation = doc_match.group(1).strip() if doc_match else ""

                # Parse arguments
                args_match = re.search(r'\[Arguments\]\s+(.+)', metadata_lines)
                arguments = []
                if args_match:
                    arguments = [arg.strip() for arg in args_match.group(1).split()]

                # Parse steps
                steps = [line.strip() for line in step_lines.split('\n') if line.strip()]

                # Infer category
                category = self._infer_keyword_category(keyword_name, documentation, steps)

                # Get line number
                line_number = content[:match.start()].count('\n') + 1

                keyword = ExistingKeyword(
                    name=keyword_name,
                    file_path=file_path,
                    line_number=line_number,
                    arguments=arguments,
                    documentation=documentation,
                    steps=steps,
                    category=category
                )

                self.keywords_cache[keyword_name] = keyword

        except Exception as e:
            logger.debug(f"Could not parse keyword file {file_path}: {str(e)}")

    def _scan_variables(self):
        """Scan for Robot Framework and Python variable files"""
        variables_dir = os.path.join(self.tests_dir, 'variables')

        if not os.path.exists(variables_dir):
            logger.warning(f"Variables directory not found: {variables_dir}")
            return

        for root, dirs, files in os.walk(variables_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.py'):
                    self._parse_python_variable_file(file_path)
                elif file.endswith('.robot'):
                    self._parse_robot_variable_file(file_path)

    def _parse_python_variable_file(self, file_path: str):
        """Parse Python variable file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                # Match pattern: VARIABLE_NAME = "value"
                match = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=\s*["\']([^"\']+)["\']', line.strip())
                if match:
                    var_name = match.group(1)
                    var_value = match.group(2)
                    var_type = self._infer_variable_type(var_name, var_value)

                    variable = ExistingVariable(
                        name=var_name,
                        value=var_value,
                        file_path=file_path,
                        line_number=line_num,
                        type=var_type
                    )

                    self.variables_cache[var_name] = variable

        except Exception as e:
            logger.debug(f"Could not parse variable file {file_path}: {str(e)}")

    def _parse_robot_variable_file(self, file_path: str):
        """Parse Robot Framework variable file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find Variables section
            variables_section = re.search(r'\*\*\* Variables \*\*\*(.*?)(?:\*\*\*|$)', content, re.DOTALL)
            if variables_section:
                lines = variables_section.group(1).split('\n')
                for line_num, line in enumerate(lines, 1):
                    # Match pattern: ${VARIABLE_NAME}    value
                    match = re.match(r'^\$\{([^}]+)\}\s+(.+)$', line.strip())
                    if match:
                        var_name = match.group(1)
                        var_value = match.group(2)
                        var_type = self._infer_variable_type(var_name, var_value)

                        variable = ExistingVariable(
                            name=var_name,
                            value=var_value,
                            file_path=file_path,
                            line_number=line_num,
                            type=var_type
                        )

                        self.variables_cache[var_name] = variable

        except Exception as e:
            logger.debug(f"Could not parse variable file {file_path}: {str(e)}")

    def _infer_locator_category(self, locator_name: str) -> str:
        """Infer locator category from name"""
        name_lower = locator_name.lower()

        if any(word in name_lower for word in ['btn', 'button']):
            return 'button'
        elif any(word in name_lower for word in ['input', 'field', 'textbox']):
            return 'input'
        elif any(word in name_lower for word in ['link', 'anchor']):
            return 'link'
        elif any(word in name_lower for word in ['dropdown', 'select', 'combo']):
            return 'dropdown'
        elif any(word in name_lower for word in ['checkbox', 'check']):
            return 'checkbox'
        elif any(word in name_lower for word in ['radio']):
            return 'radio'
        elif any(word in name_lower for word in ['label', 'text', 'heading', 'title']):
            return 'text'
        else:
            return 'element'

    def _infer_locator_context(self, locator_name: str, file_path: str) -> str:
        """Infer locator context from name and file path"""
        combined = (locator_name + file_path).lower()

        if any(word in combined for word in ['login', 'signin', 'authentication']):
            return 'login'
        elif any(word in combined for word in ['checkout', 'cart', 'payment', 'billing']):
            return 'checkout'
        elif any(word in combined for word in ['register', 'signup', 'create_account']):
            return 'registration'
        elif any(word in combined for word in ['search', 'find', 'query']):
            return 'search'
        elif any(word in combined for word in ['profile', 'account', 'settings']):
            return 'account'
        elif any(word in combined for word in ['navigation', 'menu', 'navbar', 'header']):
            return 'navigation'
        else:
            return 'general'

    def _infer_keyword_category(self, name: str, documentation: str, steps: List[str]) -> str:
        """Infer keyword category from its characteristics"""
        combined = (name + documentation + ' '.join(steps)).lower()

        if any(word in combined for word in ['navigate', 'open', 'go to', 'visit']):
            return 'navigation'
        elif any(word in combined for word in ['verify', 'check', 'validate', 'assert']):
            return 'verification'
        elif any(word in combined for word in ['click', 'select', 'choose', 'press']):
            return 'interaction'
        elif any(word in combined for word in ['input', 'enter', 'type', 'fill']):
            return 'form_fill'
        elif any(word in combined for word in ['wait', 'sleep', 'delay']):
            return 'synchronization'
        elif any(word in combined for word in ['login', 'signin', 'authenticate']):
            return 'authentication'
        else:
            return 'general'

    def _infer_variable_type(self, name: str, value: str) -> str:
        """Infer variable type from name and value"""
        name_lower = name.lower()
        value_lower = value.lower()

        if any(word in name_lower for word in ['url', 'link', 'endpoint']) or value_lower.startswith('http'):
            return 'url'
        elif any(word in name_lower for word in ['user', 'username', 'email', 'password']):
            return 'credential'
        elif any(word in name_lower for word in ['timeout', 'wait', 'delay']):
            return 'timing'
        elif any(word in name_lower for word in ['path', 'dir', 'directory', 'file']):
            return 'path'
        else:
            return 'test_data'

    def find_matching_locator(self, captured_locator: Dict[str, Any],
                            similarity_threshold: float = 0.7) -> Optional[ExistingLocator]:
        """
        Find a matching existing locator for a newly captured locator

        Args:
            captured_locator: Dictionary with 'name', 'value', 'category', 'context'
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            Best matching ExistingLocator or None
        """
        if not self._initialized:
            self.initialize()

        best_match = None
        best_score = 0.0

        for existing in self.locators_cache.values():
            score = self._calculate_locator_similarity(captured_locator, existing)

            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match = existing

        if best_match:
            logger.info(f"   🎯 Found matching locator: {best_match.name} "
                       f"(similarity: {best_score:.2f})")

        return best_match

    def _calculate_locator_similarity(self, captured: Dict[str, Any],
                                     existing: ExistingLocator) -> float:
        """Calculate similarity score between captured and existing locator"""
        score = 0.0
        weights = {
            'value': 0.5,
            'category': 0.2,
            'context': 0.2,
            'name': 0.1
        }

        # Compare locator values
        value_sim = SequenceMatcher(None,
                                   captured.get('value', ''),
                                   existing.value).ratio()
        score += value_sim * weights['value']

        # Compare categories
        if captured.get('category') == existing.category:
            score += weights['category']

        # Compare contexts
        if captured.get('context') == existing.context:
            score += weights['context']

        # Compare names
        name_sim = SequenceMatcher(None,
                                  captured.get('name', ''),
                                  existing.name).ratio()
        score += name_sim * weights['name']

        return score

    def find_matching_keyword(self, action_description: str,
                            similarity_threshold: float = 0.6) -> Optional[ExistingKeyword]:
        """
        Find a matching existing keyword for an action

        Args:
            action_description: Natural language description of the action
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            Best matching ExistingKeyword or None
        """
        if not self._initialized:
            self.initialize()

        best_match = None
        best_score = 0.0

        action_lower = action_description.lower()

        for existing in self.keywords_cache.values():
            # Calculate semantic similarity
            keyword_text = (existing.name + ' ' + existing.documentation).lower()
            score = SequenceMatcher(None, action_lower, keyword_text).ratio()

            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match = existing

        if best_match:
            logger.info(f"   🎯 Found matching keyword: {best_match.name} "
                       f"(similarity: {best_score:.2f})")

        return best_match

    def get_import_statement(self, asset_type: str, file_path: str) -> str:
        """
        Generate appropriate import/resource statement for an asset

        Args:
            asset_type: 'locator', 'keyword', or 'variable'
            file_path: Full path to the asset file

        Returns:
            Import/Resource statement string
        """
        # Make path relative to tests directory
        rel_path = os.path.relpath(file_path, self.tests_dir)

        if asset_type == 'keyword':
            return f"Resource    {rel_path}"
        elif asset_type == 'locator':
            return f"Variables    {rel_path}"
        elif asset_type == 'variable':
            return f"Variables    {rel_path}"
        else:
            return f"# Import: {rel_path}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about cached assets"""
        if not self._initialized:
            self.initialize()

        return {
            'total_locators': len(self.locators_cache),
            'total_keywords': len(self.keywords_cache),
            'total_variables': len(self.variables_cache),
            'locators_by_category': self._group_by_attr(self.locators_cache.values(), 'category'),
            'locators_by_context': self._group_by_attr(self.locators_cache.values(), 'context'),
            'keywords_by_category': self._group_by_attr(self.keywords_cache.values(), 'category'),
            'variables_by_type': self._group_by_attr(self.variables_cache.values(), 'type')
        }

    def _group_by_attr(self, items, attr: str) -> Dict[str, int]:
        """Group items by attribute and count"""
        result = {}
        for item in items:
            key = getattr(item, attr, 'unknown')
            result[key] = result.get(key, 0) + 1
        return result

