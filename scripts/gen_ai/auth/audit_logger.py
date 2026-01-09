"""
Audit Logging System for compliance and security monitoring
"""

import json
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

class AuditAction(Enum):
    """Types of actions to audit"""
    # Authentication
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"

    # Authorization
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHECK = "permission_check"

    # Module Access
    MODULE_VIEW = "module_view"
    MODULE_EXECUTE = "module_execute"
    MODULE_CONFIGURE = "module_configure"

    # Data Operations
    DATA_VIEW = "data_view"
    DATA_EXPORT = "data_export"
    DATA_DELETE = "data_delete"

    # User Management
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    ROLE_ASSIGN = "role_assign"
    ROLE_REVOKE = "role_revoke"

    # System
    SETTINGS_CHANGE = "settings_change"
    AUDIT_VIEW = "audit_view"
    AUDIT_EXPORT = "audit_export"

    # Security Events
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Audit event record"""
    event_id: str
    timestamp: str
    action: str
    severity: str
    username: str
    user_id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Action details
    module_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None

    # Result
    success: bool = True
    failure_reason: Optional[str] = None

    # Additional context
    details: Dict[str, Any] = None
    affected_user: Optional[str] = None
    changes: Optional[Dict] = None

    # Session info
    session_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

class AuditLogger:
    """Comprehensive audit logging system"""

    def __init__(self, log_path: str = None):
        if log_path is None:
            auth_dir = Path(__file__).parent
            log_path = auth_dir / "audit_logs.jsonl"

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create index file for faster searches
        self.index_path = self.log_path.with_suffix('.index.json')
        self._load_index()

    def _load_index(self):
        """Load audit log index for faster queries"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    self.index = json.load(f)
            except:
                self.index = {'users': {}, 'actions': {}, 'dates': {}}
        else:
            self.index = {'users': {}, 'actions': {}, 'dates': {}}

    def _update_index(self, event: AuditEvent):
        """Update index with new event"""
        # Index by user
        if event.username not in self.index['users']:
            self.index['users'][event.username] = []
        self.index['users'][event.username].append(event.event_id)

        # Index by action
        if event.action not in self.index['actions']:
            self.index['actions'][event.action] = []
        self.index['actions'][event.action].append(event.event_id)

        # Index by date
        date = event.timestamp.split('T')[0]
        if date not in self.index['dates']:
            self.index['dates'][date] = []
        self.index['dates'][date].append(event.event_id)

        # Save index periodically (every 10 events)
        if sum(len(v) for v in self.index['users'].values()) % 10 == 0:
            self._save_index()

    def _save_index(self):
        """Save index to file"""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            print(f"Error saving audit index: {e}")

    def log(
        self,
        action: AuditAction,
        username: str,
        user_id: str,
        success: bool = True,
        severity: AuditSeverity = AuditSeverity.INFO,
        module_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict] = None,
        failure_reason: Optional[str] = None,
        affected_user: Optional[str] = None,
        changes: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log an audit event"""
        import secrets

        event = AuditEvent(
            event_id=secrets.token_hex(16),
            timestamp=datetime.now().isoformat(),
            action=action.value,
            severity=severity.value,
            username=username,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            module_id=module_id,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            failure_reason=failure_reason,
            details=details or {},
            affected_user=affected_user,
            changes=changes,
            session_id=session_id
        )

        # Write to log file (JSONL format for streaming)
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')

            # Update index
            self._update_index(event)

        except Exception as e:
            print(f"Error writing audit log: {e}")

    def get_events(
        self,
        username: Optional[str] = None,
        action: Optional[AuditAction] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[AuditSeverity] = None,
        success: Optional[bool] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Query audit events with filters"""
        events = []

        if not self.log_path.exists():
            return events

        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    if len(events) >= limit:
                        break

                    try:
                        event_data = json.loads(line)
                        event = AuditEvent(**event_data)

                        # Apply filters
                        if username and event.username != username:
                            continue

                        if action and event.action != action.value:
                            continue

                        if severity and event.severity != severity.value:
                            continue

                        if success is not None and event.success != success:
                            continue

                        event_time = datetime.fromisoformat(event.timestamp)
                        if start_date and event_time < start_date:
                            continue

                        if end_date and event_time > end_date:
                            continue

                        events.append(event)
                    except:
                        continue

        except Exception as e:
            print(f"Error reading audit log: {e}")

        return events

    def get_user_activity(self, username: str, days: int = 7) -> List[AuditEvent]:
        """Get recent activity for a user"""
        start_date = datetime.now() - timedelta(days=days)
        return self.get_events(username=username, start_date=start_date)

    def get_failed_logins(self, days: int = 1) -> List[AuditEvent]:
        """Get recent failed login attempts"""
        start_date = datetime.now() - timedelta(days=days)
        return self.get_events(
            action=AuditAction.LOGIN_FAILED,
            start_date=start_date
        )

    def get_security_events(self, days: int = 7) -> List[AuditEvent]:
        """Get recent security-related events"""
        start_date = datetime.now() - timedelta(days=days)
        security_actions = [
            AuditAction.LOGIN_FAILED,
            AuditAction.ACCOUNT_LOCKED,
            AuditAction.ACCESS_DENIED,
            AuditAction.SUSPICIOUS_ACTIVITY,
            AuditAction.RATE_LIMIT_EXCEEDED
        ]

        events = []
        for action in security_actions:
            events.extend(self.get_events(action=action, start_date=start_date))

        return sorted(events, key=lambda e: e.timestamp, reverse=True)

    def get_admin_actions(self, days: int = 7) -> List[AuditEvent]:
        """Get recent administrative actions"""
        start_date = datetime.now() - timedelta(days=days)
        admin_actions = [
            AuditAction.USER_CREATE,
            AuditAction.USER_UPDATE,
            AuditAction.USER_DELETE,
            AuditAction.ROLE_ASSIGN,
            AuditAction.ROLE_REVOKE,
            AuditAction.SETTINGS_CHANGE
        ]

        events = []
        for action in admin_actions:
            events.extend(self.get_events(action=action, start_date=start_date))

        return sorted(events, key=lambda e: e.timestamp, reverse=True)

    def get_statistics(self, days: int = 7) -> Dict:
        """Get audit statistics"""
        start_date = datetime.now() - timedelta(days=days)
        events = self.get_events(start_date=start_date, limit=10000)

        stats = {
            'total_events': len(events),
            'by_action': {},
            'by_user': {},
            'by_severity': {},
            'success_rate': 0,
            'failed_events': 0,
            'security_events': 0
        }

        for event in events:
            # Count by action
            stats['by_action'][event.action] = stats['by_action'].get(event.action, 0) + 1

            # Count by user
            stats['by_user'][event.username] = stats['by_user'].get(event.username, 0) + 1

            # Count by severity
            stats['by_severity'][event.severity] = stats['by_severity'].get(event.severity, 0) + 1

            # Count failed events
            if not event.success:
                stats['failed_events'] += 1

            # Count security events
            if event.severity in ['warning', 'error', 'critical']:
                stats['security_events'] += 1

        # Calculate success rate
        if stats['total_events'] > 0:
            stats['success_rate'] = ((stats['total_events'] - stats['failed_events']) /
                                    stats['total_events'] * 100)

        return stats

    def cleanup_old_logs(self, retention_days: int = None):
        """Remove logs older than retention period"""
        if retention_days is None:
            retention_days = COMPLIANCE_CONFIG['audit_policy']['retention_days']

        cutoff_date = datetime.now() - timedelta(days=retention_days)

        if not self.log_path.exists():
            return

        # Read and filter events
        kept_events = []
        removed_count = 0

        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    try:
                        event_data = json.loads(line)
                        event_time = datetime.fromisoformat(event_data['timestamp'])

                        if event_time >= cutoff_date:
                            kept_events.append(line)
                        else:
                            removed_count += 1
                    except:
                        continue

            # Write back kept events
            with open(self.log_path, 'w') as f:
                f.writelines(kept_events)

            # Rebuild index
            self.index = {'users': {}, 'actions': {}, 'dates': {}}
            for line in kept_events:
                try:
                    event_data = json.loads(line)
                    event = AuditEvent(**event_data)
                    self._update_index(event)
                except:
                    continue

            self._save_index()

            return removed_count

        except Exception as e:
            print(f"Error cleaning up audit logs: {e}")
            return 0

# Import compliance config at the end to avoid circular import
from .rbac_config import COMPLIANCE_CONFIG

