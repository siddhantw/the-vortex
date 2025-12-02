import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RobotWriter")

class RobotWriter:
    """
    Writes test cases to Robot Framework format
    """
    def __init__(self, output_dir: str):
        """
        Initialize RobotWriter with output directory

        Args:
            output_dir: Directory where Robot Framework files will be written
        """
        self.output_dir = output_dir
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"RobotWriter initialized with output directory: {output_dir}")

    def write(self, test_cases: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Write test cases to a Robot Framework file

        Args:
            test_cases: List of test case dictionaries
            filename: Optional filename, defaults to "generated_tests.robot"

        Returns:
            Path to the generated Robot Framework file
        """
        if not test_cases:
            logger.warning("No test cases provided to write")
            return ""

        # Default filename with timestamp if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_tests_{timestamp}.robot"

        file_path = os.path.join(self.output_dir, filename)
        automatable_count = 0

        with open(file_path, "w") as f:
            # Add header section with metadata
            f.write("*** Settings ***\n")
            f.write("Documentation    Auto-generated test cases\n")
            f.write("...              Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("Library    SeleniumLibrary\n")
            f.write("Library    BuiltIn\n")
            f.write("\n*** Variables ***\n")
            f.write("${BROWSER}    chrome\n")
            f.write("\n*** Test Cases ***\n")

            for tc in test_cases:
                # Only write automatable cases
                if not tc.get('_automatable', False):
                    continue

                automatable_count += 1
                f.write(f"{tc['name_robot']}\n")
                f.write(f"    [Documentation]    {tc.get('description', '')}\n")

                # Add more metadata as tags
                tags = []
                if tc.get('priority'):
                    tags.append(tc.get('priority'))
                if tc.get('severity'):
                    tags.append(tc.get('severity'))
                if tc.get('brand'):
                    tags.append(tc.get('brand'))
                if tc.get('TC id'):
                    tags.append(tc.get('TC id'))

                f.write(f"    [Tags]    {' '.join(tags)}\n")
                f.write(f"    [Setup]    Log    {tc.get('preconditions', '')}\n")

                # Write test steps with proper formatting
                for step in tc.get('steps', []):
                    # Format step if it doesn't start with a keyword
                    if step and not any(step.startswith(kw) for kw in ["Click", "Input", "Select", "Wait", "Verify", "Navigate", "Open", "Log"]):
                        f.write(f"    Log    {step}\n")
                    else:
                        f.write(f"    {step}\n")

                # Properly handle expected results with appropriate keywords
                expected = tc.get('expected', '')
                if expected:
                    if "should" in expected.lower():
                        f.write(f"    Log    Verifying: {expected}\n")
                    else:
                        f.write(f"    Should Be True    ${{{expected}}}\n")

                f.write(f"    [Teardown]    Log    {tc.get('postconditions', '')}\n\n")

        logger.info(f"Robot Framework test cases written to {file_path}")
        logger.info(f"Generated {automatable_count} automatable test cases out of {len(test_cases)} total")

        return file_path

    def write_multiple_files(self, test_cases: List[Dict[str, Any]], group_by: str = 'brand') -> List[str]:
        """
        Write test cases to multiple Robot Framework files, grouped by a specific attribute

        Args:
            test_cases: List of test case dictionaries
            group_by: Attribute to group test cases by (brand, priority, severity)

        Returns:
            List of paths to generated Robot Framework files
        """
        if not test_cases:
            logger.warning("No test cases provided to write")
            return []

        # Group test cases by the specified attribute
        grouped_cases = {}
        for tc in test_cases:
            if not tc.get('_automatable', False):
                continue

            group_value = tc.get(group_by, 'default')
            if group_value not in grouped_cases:
                grouped_cases[group_value] = []

            grouped_cases[group_value].append(tc)

        # Write each group to a separate file
        file_paths = []
        for group, cases in grouped_cases.items():
            safe_group = str(group).replace(' ', '_').replace(',', '').lower()
            filename = f"generated_tests_{safe_group}.robot"
            file_path = self.write(cases, filename)
            if file_path:
                file_paths.append(file_path)

        return file_paths
