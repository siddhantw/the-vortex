"""
Authentication Manager - Session management and access control
"""

import secrets
from typing import Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
import streamlit as st

from .user_manager import User, UserManager
from .audit_logger import AuditLogger, AuditAction, AuditSeverity
from .rbac_config import Permission, COMPLIANCE_CONFIG

@dataclass
class Session:
    """User session"""
    session_id: str
    user: User
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if session is expired"""
        max_duration = timedelta(
            hours=COMPLIANCE_CONFIG['session_policy']['max_session_duration_hours']
        )
        return datetime.now() - self.created_at > max_duration

    def is_idle(self) -> bool:
        """Check if session is idle"""
        idle_timeout = timedelta(
            minutes=COMPLIANCE_CONFIG['session_policy']['idle_timeout_minutes']
        )
        return datetime.now() - self.last_activity > idle_timeout

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

class AuthManager:
    """Manages authentication and authorization"""

    def __init__(self, user_manager: UserManager, audit_logger: AuditLogger):
        self.user_manager = user_manager
        self.audit_logger = audit_logger
        self.active_sessions: Dict[str, Session] = {}

    def login(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[Session]:
        """Authenticate user and create session"""
        user = self.user_manager.authenticate(username, password)

        if not user:
            # Log failed login
            self.audit_logger.log(
                action=AuditAction.LOGIN_FAILED,
                username=username,
                user_id="unknown",
                success=False,
                severity=AuditSeverity.WARNING,
                failure_reason="Invalid credentials",
                ip_address=ip_address,
                user_agent=user_agent
            )
            return None

        # Check concurrent session limit
        user_sessions = [s for s in self.active_sessions.values() if s.user.username == username]
        max_sessions = user.get_max_concurrent_sessions()

        if len(user_sessions) >= max_sessions:
            # Remove oldest session
            oldest = min(user_sessions, key=lambda s: s.created_at)
            self.logout(oldest.session_id)

        # Create session
        session_id = secrets.token_urlsafe(32)
        session = Session(
            session_id=session_id,
            user=user,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.active_sessions[session_id] = session
        user.active_sessions.append(session_id)

        # Log successful login
        self.audit_logger.log(
            action=AuditAction.LOGIN_SUCCESS,
            username=user.username,
            user_id=user.user_id,
            success=True,
            severity=AuditSeverity.INFO,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id
        )

        return session

    def logout(self, session_id: str) -> bool:
        """End user session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        # Log logout
        self.audit_logger.log(
            action=AuditAction.LOGOUT,
            username=session.user.username,
            user_id=session.user.user_id,
            success=True,
            severity=AuditSeverity.INFO,
            session_id=session_id
        )

        # Remove session
        if session_id in session.user.active_sessions:
            session.user.active_sessions.remove(session_id)
        del self.active_sessions[session_id]

        return True

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session and validate"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        # Check if expired or idle
        if session.is_expired() or session.is_idle():
            self.logout(session_id)
            return None

        # Update activity
        session.update_activity()
        return session

    def check_permission(
        self,
        session_id: str,
        permission: Permission,
        log_check: bool = False
    ) -> bool:
        """Check if user has permission"""
        session = self.get_session(session_id)
        if not session:
            return False

        has_permission = session.user.has_permission(permission.value)

        if log_check:
            self.audit_logger.log(
                action=AuditAction.PERMISSION_CHECK,
                username=session.user.username,
                user_id=session.user.user_id,
                success=has_permission,
                severity=AuditSeverity.INFO if has_permission else AuditSeverity.WARNING,
                details={'permission': permission.value},
                session_id=session_id
            )

        return has_permission

    def check_module_access(
        self,
        session_id: str,
        module_id: str,
        permission: Permission = Permission.VIEW_MODULE
    ) -> bool:
        """Check if user can access module with specific permission"""
        session = self.get_session(session_id)
        if not session:
            return False

        # Check module access
        can_access = session.user.can_access_module(module_id)

        # Check permission
        has_permission = self.check_permission(session_id, permission)

        result = can_access and has_permission

        # Log access check
        if not result:
            self.audit_logger.log(
                action=AuditAction.ACCESS_DENIED,
                username=session.user.username,
                user_id=session.user.user_id,
                success=False,
                severity=AuditSeverity.WARNING,
                module_id=module_id,
                details={
                    'permission': permission.value,
                    'has_module_access': can_access,
                    'has_permission': has_permission
                },
                session_id=session_id
            )
        else:
            self.audit_logger.log(
                action=AuditAction.ACCESS_GRANTED,
                username=session.user.username,
                user_id=session.user.user_id,
                success=True,
                severity=AuditSeverity.INFO,
                module_id=module_id,
                details={'permission': permission.value},
                session_id=session_id
            )

        return result

    def log_module_execution(
        self,
        session_id: str,
        module_id: str,
        success: bool = True,
        details: Optional[Dict] = None
    ):
        """Log module execution"""
        session = self.get_session(session_id)
        if not session:
            return

        # Check execution limit
        if not session.user.increment_execution_count():
            self.audit_logger.log(
                action=AuditAction.RATE_LIMIT_EXCEEDED,
                username=session.user.username,
                user_id=session.user.user_id,
                success=False,
                severity=AuditSeverity.WARNING,
                module_id=module_id,
                details={
                    'daily_limit': session.user.get_max_daily_executions(),
                    'current_count': session.user.daily_execution_count
                },
                session_id=session_id
            )
            return False

        # Log execution
        self.audit_logger.log(
            action=AuditAction.MODULE_EXECUTE,
            username=session.user.username,
            user_id=session.user.user_id,
            success=success,
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            module_id=module_id,
            details=details or {},
            session_id=session_id
        )

        return True

    def cleanup_sessions(self):
        """Remove expired and idle sessions"""
        expired_sessions = [
            sid for sid, session in self.active_sessions.items()
            if session.is_expired() or session.is_idle()
        ]

        for session_id in expired_sessions:
            self.logout(session_id)

        return len(expired_sessions)

class StreamlitAuthManager:
    """Streamlit-specific authentication wrapper"""

    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager

    def initialize_session_state(self):
        """Initialize Streamlit session state for auth"""
        if 'auth_session_id' not in st.session_state:
            st.session_state.auth_session_id = None
        if 'auth_user' not in st.session_state:
            st.session_state.auth_user = None
        if 'auth_initialized' not in st.session_state:
            st.session_state.auth_initialized = True

    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        self.initialize_session_state()

        if not st.session_state.auth_session_id:
            return False

        session = self.auth_manager.get_session(st.session_state.auth_session_id)
        if not session:
            # Clear invalid session
            st.session_state.auth_session_id = None
            st.session_state.auth_user = None
            return False

        # Update user in session state
        st.session_state.auth_user = session.user
        return True

    def get_current_user(self) -> Optional[User]:
        """Get currently authenticated user"""
        if self.is_authenticated():
            return st.session_state.auth_user
        return None

    def login(self, username: str, password: str) -> bool:
        """Login user"""
        self.initialize_session_state()

        session = self.auth_manager.login(username, password)
        if session:
            st.session_state.auth_session_id = session.session_id
            st.session_state.auth_user = session.user
            return True
        return False

    def logout(self):
        """Logout user"""
        if st.session_state.auth_session_id:
            self.auth_manager.logout(st.session_state.auth_session_id)

        st.session_state.auth_session_id = None
        st.session_state.auth_user = None

    def require_auth(self):
        """Decorator/wrapper to require authentication"""
        if not self.is_authenticated():
            st.error("⛔ Authentication required")
            st.stop()

    def require_permission(self, permission: Permission):
        """Require specific permission"""
        self.require_auth()

        if not self.auth_manager.check_permission(
            st.session_state.auth_session_id,
            permission
        ):
            st.error(f"⛔ Permission denied: {permission.value}")
            st.stop()

    def require_module_access(self, module_id: str, permission: Permission = Permission.VIEW_MODULE):
        """Require module access with permission"""
        self.require_auth()

        if not self.auth_manager.check_module_access(
            st.session_state.auth_session_id,
            module_id,
            permission
        ):
            st.error(f"⛔ Access denied to module: {module_id}")
            st.stop()

    def check_permission(self, permission: Permission) -> bool:
        """Check permission without blocking"""
        if not self.is_authenticated():
            return False

        return self.auth_manager.check_permission(
            st.session_state.auth_session_id,
            permission
        )

    def check_module_access(self, module_id: str, permission: Permission = Permission.VIEW_MODULE) -> bool:
        """Check module access without blocking"""
        if not self.is_authenticated():
            return False

        return self.auth_manager.check_module_access(
            st.session_state.auth_session_id,
            module_id,
            permission
        )

    def is_admin(self) -> bool:
        """Check if current user is super_admin or admin"""
        user = self.get_current_user()
        if not user:
            return False

        # Only super_admin and admin roles have admin access
        admin_roles = {'super_admin', 'admin'}
        return any(role in admin_roles for role in user.roles)

    def require_admin_access(self):
        """Require admin role (super_admin or admin only)"""
        self.require_auth()

        if not self.is_admin():
            st.error("⛔ Access Denied: Admin privileges required. Only super_admin and admin roles can access this area.")

            # Log unauthorized access attempt
            user = self.get_current_user()
            if user:
                self.auth_manager.audit_logger.log(
                    action=AuditAction.ACCESS_DENIED,
                    username=user.username,
                    user_id=user.user_id,
                    success=False,
                    severity=AuditSeverity.WARNING,
                    details={
                        'attempted_access': 'admin_panel',
                        'user_roles': user.roles,
                        'reason': 'insufficient_role_privileges'
                    },
                    session_id=st.session_state.auth_session_id
                )

            st.stop()

