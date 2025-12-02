#!/usr/bin/env python3
"""Wrapper to launch RobotMCP with warning filters applied.

Suppresses:
- DeprecationWarning from robot.utils.is_string (noise until ecosystem updates)
- UserWarning from robotmcp.config.library_registry duplicate priority validation

This avoids modifying site-packages while keeping console clean.
"""

import warnings


def _configure_warning_filters() -> None:
    # Hide RF deprecations emitted from robot.utils
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"robot\.utils",
    )

    # Hide library registry validation noise about duplicate priorities
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"robotmcp\.config\.library_registry",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Library registry validation errors:.*",
        category=UserWarning,
    )


def main() -> None:
    _configure_warning_filters()

    # Start the RobotMCP server
    from robotmcp.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
