"""
RBAC Configuration for Vortex Portal
Defines roles, permissions, and access control policies
"""

from enum import Enum
from typing import Dict, Set
from dataclasses import dataclass, field

class Permission(Enum):
    """System-wide permissions"""
    # Module Access Permissions
    VIEW_MODULE = "view_module"
    EXECUTE_MODULE = "execute_module"
    CONFIGURE_MODULE = "configure_module"

    # Data Permissions
    VIEW_DATA = "view_data"
    EXPORT_DATA = "export_data"
    DELETE_DATA = "delete_data"

    # Admin Permissions
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_SETTINGS = "manage_settings"

    # Advanced Features
    API_ACCESS = "api_access"
    BULK_OPERATIONS = "bulk_operations"
    ADVANCED_ANALYTICS = "advanced_analytics"

class ModuleCategory(Enum):
    """Module categories for granular access control"""
    TEST_GENERATION = "test_generation"
    TEST_MAINTENANCE = "test_maintenance"
    AUTOMATION = "automation"
    DEVOPS = "devops"
    ANALYSIS = "analysis"
    PREDICTIVE = "predictive"
    ALL = "all"

@dataclass
class Role:
    """Role definition with permissions and module access"""
    name: str
    display_name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    module_access: Dict[str, Set[str]] = field(default_factory=dict)  # category -> module_ids
    max_daily_executions: int = 1000
    max_concurrent_sessions: int = 5
    can_use_ai_features: bool = True
    priority_level: int = 1  # 1=lowest, 5=highest
    is_system_role: bool = False  # Cannot be deleted

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission"""
        return permission in self.permissions

    def can_access_module(self, module_id: str) -> bool:
        """Check if role can access specific module"""
        # Check if user has access to all modules
        if ModuleCategory.ALL.value in self.module_access:
            return True

        # Check specific module access across all categories
        for modules in self.module_access.values():
            if module_id in modules:
                return True
        return False

    def get_accessible_modules(self) -> Set[str]:
        """Get all accessible module IDs"""
        if ModuleCategory.ALL.value in self.module_access:
            return {"*"}  # All modules

        accessible = set()
        for modules in self.module_access.values():
            accessible.update(modules)
        return accessible

# Define system roles with comprehensive permissions
SYSTEM_ROLES: Dict[str, Role] = {
    "super_admin": Role(
        name="super_admin",
        display_name="Super Administrator",
        description="Full system access with all permissions. Can manage users, roles, and system settings.",
        permissions={
            Permission.VIEW_MODULE,
            Permission.EXECUTE_MODULE,
            Permission.CONFIGURE_MODULE,
            Permission.VIEW_DATA,
            Permission.EXPORT_DATA,
            Permission.DELETE_DATA,
            Permission.MANAGE_USERS,
            Permission.MANAGE_ROLES,
            Permission.VIEW_AUDIT_LOGS,
            Permission.MANAGE_SETTINGS,
            Permission.API_ACCESS,
            Permission.BULK_OPERATIONS,
            Permission.ADVANCED_ANALYTICS
        },
        module_access={ModuleCategory.ALL.value: {"*"}},
        max_daily_executions=10000,
        max_concurrent_sessions=20,
        can_use_ai_features=True,
        priority_level=5,
        is_system_role=True
    ),

    "admin": Role(
        name="admin",
        display_name="Administrator",
        description="Administrative access with user management. Cannot modify system roles.",
        permissions={
            Permission.VIEW_MODULE,
            Permission.EXECUTE_MODULE,
            Permission.CONFIGURE_MODULE,
            Permission.VIEW_DATA,
            Permission.EXPORT_DATA,
            Permission.DELETE_DATA,
            Permission.MANAGE_USERS,
            Permission.VIEW_AUDIT_LOGS,
            Permission.API_ACCESS,
            Permission.BULK_OPERATIONS,
            Permission.ADVANCED_ANALYTICS
        },
        module_access={ModuleCategory.ALL.value: {"*"}},
        max_daily_executions=5000,
        max_concurrent_sessions=10,
        can_use_ai_features=True,
        priority_level=4,
        is_system_role=True
    ),

    "power_user": Role(
        name="power_user",
        display_name="Power User",
        description="Advanced user with access to all modules and advanced features.",
        permissions={
            Permission.VIEW_MODULE,
            Permission.EXECUTE_MODULE,
            Permission.CONFIGURE_MODULE,
            Permission.VIEW_DATA,
            Permission.EXPORT_DATA,
            Permission.API_ACCESS,
            Permission.BULK_OPERATIONS,
            Permission.ADVANCED_ANALYTICS
        },
        module_access={ModuleCategory.ALL.value: {"*"}},
        max_daily_executions=2000,
        max_concurrent_sessions=8,
        can_use_ai_features=True,
        priority_level=3,
        is_system_role=True
    ),

    "developer": Role(
        name="developer",
        display_name="Developer",
        description="Development team member with access to test generation and automation modules.",
        permissions={
            Permission.VIEW_MODULE,
            Permission.EXECUTE_MODULE,
            Permission.CONFIGURE_MODULE,
            Permission.VIEW_DATA,
            Permission.EXPORT_DATA,
            Permission.API_ACCESS
        },
        module_access={
            ModuleCategory.TEST_GENERATION.value: {
                "test_pilot",
                "dynamic_tc_generation",
                "api_generation"
            },
            ModuleCategory.TEST_MAINTENANCE.value: {
                "visual_ai_testing",
                "fos_checks"
            },
            ModuleCategory.AUTOMATION.value: {
                "performance_testing"
            },
            ModuleCategory.DEVOPS.value: {
                "edb_query_manager",
                "newfold_migration_toolkit"
            },
            ModuleCategory.ANALYSIS.value: {
                "pull_requests_reviewer"
            }
        },
        max_daily_executions=1000,
        max_concurrent_sessions=5,
        can_use_ai_features=True,
        priority_level=2,
        is_system_role=True
    ),

    "qa_engineer": Role(
        name="qa_engineer",
        display_name="QA Engineer",
        description="Quality assurance engineer with testing and analysis capabilities.",
        permissions={
            Permission.VIEW_MODULE,
            Permission.EXECUTE_MODULE,
            Permission.CONFIGURE_MODULE,
            Permission.VIEW_DATA,
            Permission.EXPORT_DATA,
            Permission.API_ACCESS
        },
        module_access={
            ModuleCategory.TEST_GENERATION.value: {
                "test_pilot",
                "dynamic_tc_generation",
                "intelligent_test_data_generation",
                "api_generation"
            },
            ModuleCategory.TEST_MAINTENANCE.value: {
                "self_healing_tests",
                "visual_ai_testing",
                "robocop_lint_checker",
                "smart_test_optimizer",
                "fos_checks"
            },
            ModuleCategory.AUTOMATION.value: {
                "performance_testing",
                "security_penetration_testing"
            },
            ModuleCategory.DEVOPS.value: {
                "edb_query_manager",
                "rf_dashboard_analytics",
                "jenkins_dashboard"
            },
            ModuleCategory.ANALYSIS.value: {
                "auto_documentation",
                "smart_cx_navigator",
                "pull_requests_reviewer",
                "manual_test_analysis"
            },
            ModuleCategory.PREDICTIVE.value: {
                "intelligent_bug_predictor",
                "ai_test_execution_orchestrator",
                "ai_quality_assurance_guardian"
            }
        },
        max_daily_executions=500,
        max_concurrent_sessions=3,
        can_use_ai_features=True,
        priority_level=2,
        is_system_role=True
    ),

    "devops_engineer": Role(
        name="devops_engineer",
        display_name="DevOps Engineer",
        description="DevOps team member with CI/CD and infrastructure access.",
        permissions={
            Permission.VIEW_MODULE,
            Permission.EXECUTE_MODULE,
            Permission.VIEW_DATA,
            Permission.EXPORT_DATA,
            Permission.API_ACCESS
        },
        module_access={
            ModuleCategory.DEVOPS.value: {
                "rf_dashboard_analytics",
                "jenkins_dashboard",
                "database_insights",
                "ai_test_environment_manager",
                "edb_query_manager",
                "newfold_migration_toolkit"
            },
            ModuleCategory.AUTOMATION.value: {
                "performance_testing",
                "security_penetration_testing"
            }
        },
        max_daily_executions=1000,
        max_concurrent_sessions=5,
        can_use_ai_features=False,
        priority_level=2,
        is_system_role=True
    ),

    "analyst": Role(
        name="analyst",
        display_name="Analyst",
        description="Analysis and reporting role with read-only access to analytics modules.",
        permissions={
            Permission.VIEW_MODULE,
            Permission.VIEW_DATA,
            Permission.EXPORT_DATA
        },
        module_access={
            ModuleCategory.ANALYSIS.value: {
                "auto_documentation",
                "smart_cx_navigator",
                "manual_test_analysis"
            },
            ModuleCategory.DEVOPS.value: {
                "rf_dashboard_analytics",
                "jenkins_dashboard",
                "database_insights"
            },
            ModuleCategory.PREDICTIVE.value: {
                "intelligent_bug_predictor",
                "ai_quality_assurance_guardian"
            }
        },
        max_daily_executions=200,
        max_concurrent_sessions=2,
        can_use_ai_features=False,
        priority_level=1,
        is_system_role=True
    ),

    "viewer": Role(
        name="viewer",
        display_name="Viewer",
        description="Read-only access to view modules and data without execution capabilities.",
        permissions={
            Permission.VIEW_MODULE,
            Permission.VIEW_DATA
        },
        module_access={},  # Specific modules assigned per user
        max_daily_executions=50,
        max_concurrent_sessions=1,
        can_use_ai_features=False,
        priority_level=1,
        is_system_role=True
    ),

    "guest": Role(
        name="guest",
        display_name="Guest",
        description="Limited guest access for demos and trials. Restricted to basic modules.",
        permissions={
            Permission.VIEW_MODULE
        },
        module_access={
            ModuleCategory.TEST_GENERATION.value: {
                "test_pilot",
                "dynamic_tc_generation",
                "intelligent_test_data_generation",
                "api_generation"
            },ModuleCategory.TEST_MAINTENANCE.value: {
                "visual_ai_testing",
                "robocop_lint_checker",
                "fos_checks"
            },
            ModuleCategory.AUTOMATION.value: {
                "performance_testing"
            },
            ModuleCategory.DEVOPS.value: {
                "rf_dashboard_analytics",
                "jenkins_dashboard"
            },
            ModuleCategory.ANALYSIS.value: {
                "pull_requests_reviewer"
            }
        },
        max_daily_executions=10,
        max_concurrent_sessions=1,
        can_use_ai_features=False,
        priority_level=1,
        is_system_role=True
    )
}

# Module to category mapping for easy lookup
MODULE_TO_CATEGORY: Dict[str, ModuleCategory] = {
    # Test Generation & Intelligence
    "test_pilot": ModuleCategory.TEST_GENERATION,
    "dynamic_tc_generation": ModuleCategory.TEST_GENERATION,
    "intelligent_test_data_generation": ModuleCategory.TEST_GENERATION,
    "api_generation": ModuleCategory.TEST_GENERATION,

    # Test Maintenance & Quality
    "self_healing_tests": ModuleCategory.TEST_MAINTENANCE,
    "visual_ai_testing": ModuleCategory.TEST_MAINTENANCE,
    "robocop_lint_checker": ModuleCategory.TEST_MAINTENANCE,
    "smart_test_optimizer": ModuleCategory.TEST_MAINTENANCE,
    "fos_checks": ModuleCategory.TEST_MAINTENANCE,

    # Automation & Integration
    "ai_cross_platform_orchestrator": ModuleCategory.AUTOMATION,
    "performance_testing": ModuleCategory.AUTOMATION,
    "security_penetration_testing": ModuleCategory.AUTOMATION,
    "browser_agent": ModuleCategory.AUTOMATION,

    # DevOps & Monitoring
    "rf_dashboard_analytics": ModuleCategory.DEVOPS,
    "jenkins_dashboard": ModuleCategory.DEVOPS,
    "edb_query_manager": ModuleCategory.DEVOPS,
    "database_insights": ModuleCategory.DEVOPS,
    "ai_test_environment_manager": ModuleCategory.DEVOPS,
    "newfold_migration_toolkit": ModuleCategory.DEVOPS,

    # Analysis & Documentation
    "auto_documentation": ModuleCategory.ANALYSIS,
    "smart_cx_navigator": ModuleCategory.ANALYSIS,
    "pull_requests_reviewer": ModuleCategory.ANALYSIS,
    "manual_test_analysis": ModuleCategory.ANALYSIS,

    # Predictive Intelligence
    "intelligent_bug_predictor": ModuleCategory.PREDICTIVE,
    "ai_test_execution_orchestrator": ModuleCategory.PREDICTIVE,
    "ai_quality_assurance_guardian": ModuleCategory.PREDICTIVE,
}

# Compliance and security settings
COMPLIANCE_CONFIG = {
    "password_policy": {
        "min_length": 12,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_special_chars": True,
        "max_age_days": 90,
        "prevent_reuse_count": 5
    },
    "session_policy": {
        "max_session_duration_hours": 12,
        "idle_timeout_minutes": 30,
        "require_mfa_for_roles": ["super_admin", "admin"],
        "max_failed_login_attempts": 5,
        "lockout_duration_minutes": 30
    },
    "audit_policy": {
        "log_all_access": True,
        "log_data_exports": True,
        "log_config_changes": True,
        "log_failed_access": True,
        "retention_days": 365
    },
    "data_protection": {
        "encrypt_sensitive_data": True,
        "mask_pii_in_logs": True,
        "require_reason_for_export": True,
        "watermark_exports": True
    }
}

