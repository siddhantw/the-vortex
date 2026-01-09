"""
User Model
Standalone user data class to avoid circular imports
"""

from typing import List, Dict, Set, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from .rbac_config import Role, SYSTEM_ROLES


@dataclass
class User:
    """User model with comprehensive attributes"""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    roles: List[str] = field(default_factory=list)
    custom_permissions: Set[str] = field(default_factory=set)
    custom_module_access: Set[str] = field(default_factory=set)
    full_name: str = ""
    department: str = ""
    groups: List[str] = field(default_factory=list)

    # Account status
    is_active: bool = True
    is_locked: bool = False
    locked_until: Optional[str] = None
    must_change_password: bool = True

    # Security tracking
    failed_login_attempts: int = 0
    last_login: Optional[str] = None
    last_password_change: Optional[str] = None
    password_history: List[str] = field(default_factory=list)
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

    # Session management
    active_sessions: List[str] = field(default_factory=list)

    # Usage tracking
    daily_execution_count: int = 0
    last_execution_reset: Optional[str] = None
    total_executions: int = 0

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "system"
    modified_at: Optional[str] = None
    modified_by: Optional[str] = None
    last_accessed: Optional[str] = None

    # Compliance
    terms_accepted: bool = False
    terms_accepted_at: Optional[str] = None
    data_usage_consent: bool = False

    def get_all_roles(self) -> List[Role]:
        """Get all Role objects for this user"""
        return [SYSTEM_ROLES[role_name] for role_name in self.roles if role_name in SYSTEM_ROLES]

    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role"""
        return role_name in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission (from roles or custom)"""
        if permission in self.custom_permissions:
            return True

        for role in self.get_all_roles():
            if any(p.value == permission for p in role.permissions):
                return True
        return False

    def can_access_module(self, module_id: str) -> bool:
        """Check if user can access specific module"""
        if module_id in self.custom_module_access:
            return True

        for role in self.get_all_roles():
            if role.can_access_module(module_id):
                return True
        return False

    def get_max_daily_executions(self) -> int:
        """Get maximum daily executions from all roles"""
        if not self.roles:
            return 0
        return max(SYSTEM_ROLES[role].max_daily_executions for role in self.roles if role in SYSTEM_ROLES)

    def get_max_concurrent_sessions(self) -> int:
        """Get maximum concurrent sessions from all roles"""
        if not self.roles:
            return 1
        return max(SYSTEM_ROLES[role].max_concurrent_sessions for role in self.roles if role in SYSTEM_ROLES)

    def increment_execution_count(self) -> bool:
        """Increment execution count and check if limit exceeded"""
        today = datetime.now().date().isoformat()

        # Reset counter if it's a new day
        if self.last_execution_reset != today:
            self.daily_execution_count = 0
            self.last_execution_reset = today

        # Check if limit exceeded
        max_executions = self.get_max_daily_executions()
        if self.daily_execution_count >= max_executions:
            return False

        self.daily_execution_count += 1
        self.total_executions += 1
        return True

    def to_dict(self) -> Dict:
        """Convert user to dictionary (excluding sensitive data for display)"""
        data = asdict(self)
        # Remove sensitive fields
        data.pop('password_hash', None)
        data.pop('salt', None)
        data.pop('mfa_secret', None)
        data.pop('password_history', None)
        # Convert sets to lists for JSON serialization
        data['custom_permissions'] = list(data.get('custom_permissions', []))
        data['custom_module_access'] = list(data.get('custom_module_access', []))
        return data

