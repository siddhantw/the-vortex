"""
TestPilot Configuration Manager
Handles loading, validation, and management of TestPilot configurations
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️ PyYAML not available. Install with: pip install pyyaml")


class TestPilotConfigManager:
    """Manages TestPilot configurations with validation and defaults"""

    DEFAULT_CONFIG = {
        'browser': {
            'type': 'chrome',
            'headless': True,
            'timeout': 30,
            'implicit_wait': 10,
            'page_load_timeout': 60,
            'window_size': '1920x1080'
        },
        'features': {
            'self_healing': True,
            'visual_validation': True,
            'performance_testing': False,
            'accessibility_testing': True,
            'security_testing': True,
            'distributed_testing': False,
            'flaky_detection': True
        },
        'ai': {
            'provider': 'azure_openai',
            'model': 'gpt-4.1-mini',
            'temperature': 0.3,
            'max_tokens': 4000,
            'retry_attempts': 3,
            'timeout': 60
        },
        'distributed': {
            'enabled': False,
            'nodes': [],
            'max_parallel_per_node': 5,
            'health_check_interval': 60,
            'retry_failed_nodes': True
        },
        'metrics': {
            'enabled': True,
            'export_format': 'json',
            'retention_days': 90,
            'auto_export': False,
            'export_interval_hours': 24
        },
        'reporting': {
            'screenshot_on_failure': True,
            'detailed_logs': True,
            'send_notifications': False,
            'notification_channels': [],
            'include_performance_metrics': True
        },
        'locators': {
            'priority_order': ['id', 'name', 'css', 'xpath'],
            'fuzzy_match_threshold': 0.8,
            'self_heal_confidence_threshold': 0.7,
            'max_alternatives': 5
        },
        'execution': {
            'max_retries': 3,
            'retry_delay_seconds': 5,
            'fail_fast': False,
            'parallel_execution': False,
            'max_parallel_tests': 4
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Look for config in standard locations
            self.config_path = self._find_config_file()

        self.config = self.load_config()
        self._validate_on_init()

    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in standard locations"""
        search_paths = [
            Path('testpilot_config.yaml'),
            Path('testpilot_config.yml'),
            Path('testpilot_config.json'),
            Path('.testpilot/config.yaml'),
            Path('.testpilot/config.yml'),
            Path('.testpilot/config.json'),
            Path(os.path.expanduser('~/.testpilot/config.yaml')),
            Path(os.path.expanduser('~/.testpilot/config.yml')),
            Path(os.path.expanduser('~/.testpilot/config.json'))
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if self.config_path and self.config_path.exists():
            try:
                if self.config_path.suffix == '.json':
                    with open(self.config_path, 'r') as f:
                        loaded_config = json.load(f)
                elif self.config_path.suffix in ['.yaml', '.yml']:
                    if not YAML_AVAILABLE:
                        print("⚠️ YAML config found but PyYAML not available. Using defaults.")
                        return self.DEFAULT_CONFIG.copy()
                    with open(self.config_path, 'r') as f:
                        loaded_config = yaml.safe_load(f)
                else:
                    print(f"⚠️ Unsupported config format: {self.config_path.suffix}")
                    return self.DEFAULT_CONFIG.copy()

                # Merge with defaults (loaded config takes precedence)
                return self._merge_configs(self.DEFAULT_CONFIG, loaded_config)

            except Exception as e:
                print(f"⚠️ Error loading config from {self.config_path}: {e}")
                print("Using default configuration.")
                return self.DEFAULT_CONFIG.copy()
        else:
            return self.DEFAULT_CONFIG.copy()

    def _merge_configs(self, default: Dict, override: Dict) -> Dict:
        """Recursively merge configurations"""
        result = default.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def save_config(self, backup: bool = True):
        """
        Save current configuration to file

        Args:
            backup: Create backup of existing config
        """
        if not self.config_path:
            self.config_path = Path('testpilot_config.yaml')

        # Create backup if requested
        if backup and self.config_path.exists():
            backup_path = self.config_path.with_suffix(
                f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}{self.config_path.suffix}'
            )
            import shutil
            shutil.copy2(self.config_path, backup_path)
            print(f"📦 Backup created: {backup_path}")

        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        try:
            if self.config_path.suffix == '.json':
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif self.config_path.suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    print("⚠️ PyYAML not available. Saving as JSON instead.")
                    self.config_path = self.config_path.with_suffix('.json')
                    with open(self.config_path, 'w') as f:
                        json.dump(self.config, f, indent=2)
                else:
                    with open(self.config_path, 'w') as f:
                        yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")

            print(f"✅ Configuration saved: {self.config_path}")

        except Exception as e:
            print(f"❌ Error saving config: {e}")
            raise

    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path (e.g., 'browser.type')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value):
        """
        Set configuration value using dot notation

        Args:
            key_path: Dot-separated path (e.g., 'browser.type')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set value
        config[keys[-1]] = value

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if valid, False otherwise
        """
        errors = []

        # Required sections
        required_sections = ['browser', 'features', 'ai']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")

        # Browser validation
        if 'browser' in self.config:
            browser = self.config['browser']
            valid_types = ['chrome', 'firefox', 'edge', 'safari']
            if browser.get('type') not in valid_types:
                errors.append(f"Invalid browser type: {browser.get('type')}. Must be one of {valid_types}")

            if not isinstance(browser.get('timeout', 30), (int, float)) or browser.get('timeout', 30) <= 0:
                errors.append(f"Invalid browser timeout: {browser.get('timeout')}")

        # AI validation
        if 'ai' in self.config:
            ai = self.config['ai']
            if ai.get('temperature', 0.3) < 0 or ai.get('temperature', 0.3) > 1:
                errors.append(f"Invalid AI temperature: {ai.get('temperature')}. Must be between 0 and 1")

            if not isinstance(ai.get('max_tokens', 4000), int) or ai.get('max_tokens', 4000) <= 0:
                errors.append(f"Invalid AI max_tokens: {ai.get('max_tokens')}")

        # Features validation
        if 'features' in self.config:
            features = self.config['features']
            for key, value in features.items():
                if not isinstance(value, bool):
                    errors.append(f"Feature '{key}' must be boolean, got {type(value).__name__}")

        # Distributed validation
        if 'distributed' in self.config and self.config['distributed'].get('enabled'):
            nodes = self.config['distributed'].get('nodes', [])
            if not nodes:
                errors.append("Distributed testing enabled but no nodes configured")

        # Print errors
        if errors:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False

        print("✅ Configuration is valid")
        return True

    def _validate_on_init(self):
        """Validate config on initialization (non-blocking)"""
        try:
            self.validate()
        except Exception as e:
            print(f"⚠️ Configuration validation warning: {e}")

    def print_config(self, section: Optional[str] = None):
        """
        Print current configuration

        Args:
            section: Specific section to print (optional)
        """
        print("\n" + "="*60)
        print("📋 TestPilot Configuration")
        print("="*60)

        if self.config_path:
            print(f"Source: {self.config_path}")
        else:
            print("Source: Defaults (no config file found)")
        print()

        if section:
            if section in self.config:
                print(f"[{section}]")
                self._print_dict(self.config[section], indent=2)
            else:
                print(f"Section '{section}' not found")
        else:
            for key, value in self.config.items():
                print(f"[{key}]")
                self._print_dict(value, indent=2)
                print()

        print("="*60 + "\n")

    def _print_dict(self, d: Dict, indent: int = 0):
        """Recursively print dictionary"""
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}")

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.DEFAULT_CONFIG.copy()
        print("✅ Configuration reset to defaults")

    def export_template(self, output_path: str):
        """
        Export configuration as template

        Args:
            output_path: Path to save template
        """
        template_path = Path(output_path)

        # Add comments to template
        template = self.DEFAULT_CONFIG.copy()

        try:
            if template_path.suffix == '.json':
                with open(template_path, 'w') as f:
                    json.dump(template, f, indent=2)
            elif template_path.suffix in ['.yaml', '.yml']:
                if YAML_AVAILABLE:
                    with open(template_path, 'w') as f:
                        f.write("# TestPilot Configuration Template\n")
                        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
                else:
                    print("⚠️ PyYAML not available. Exporting as JSON instead.")
                    template_path = template_path.with_suffix('.json')
                    with open(template_path, 'w') as f:
                        json.dump(template, f, indent=2)

            print(f"✅ Configuration template exported: {template_path}")

        except Exception as e:
            print(f"❌ Error exporting template: {e}")
            raise


# CLI for configuration management
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TestPilot Configuration Manager')
    parser.add_argument('command', choices=['show', 'validate', 'export-template', 'reset'])
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--section', help='Show specific section')
    parser.add_argument('--output', help='Output path for export-template')

    args = parser.parse_args()

    config = TestPilotConfigManager(args.config)

    if args.command == 'show':
        config.print_config(args.section)
    elif args.command == 'validate':
        if config.validate():
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has errors")
    elif args.command == 'export-template':
        output = args.output or 'testpilot_config_template.yaml'
        config.export_template(output)
    elif args.command == 'reset':
        config.reset_to_defaults()
        if args.config:
            config.save_config(backup=True)

