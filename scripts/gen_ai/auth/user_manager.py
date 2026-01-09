"""
User Management System with secure credential storage
"""

import json
import hashlib
import secrets
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from dataclasses import asdict
from pathlib import Path
import re

from .rbac_config import SYSTEM_ROLES, COMPLIANCE_CONFIG
from .security_manager import get_security_manager
from .secure_user_storage import get_secure_storage
from .user_model import User

class UserManager:
    """Manages users with secure storage and operations"""

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            # Default to auth directory - use secure storage
            self.secure_storage = get_secure_storage()
            self.security_manager = get_security_manager()
            self.storage_path = self.secure_storage.storage_path
        else:
            # Custom path - legacy mode (not recommended)
            self.storage_path = Path(storage_path)
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.secure_storage = None
            self.security_manager = None

        self.users: Dict[str, User] = {}
        self._load_users()
        self._ensure_default_admin()

    def _load_users(self):
        """Load users from storage (encrypted if available)"""
        if self.secure_storage:
            # Use encrypted storage
            self.users = self.secure_storage.load_users()
        elif self.storage_path.exists():
            # Legacy unencrypted storage
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for user_data in data.get('users', []):
                        # Convert lists back to sets where needed
                        if 'custom_permissions' in user_data:
                            user_data['custom_permissions'] = set(user_data['custom_permissions'])
                        if 'custom_module_access' in user_data:
                            user_data['custom_module_access'] = set(user_data['custom_module_access'])
                        user = User(**user_data)
                        self.users[user.username] = user
            except Exception as e:
                print(f"Error loading users: {e}")

    def _save_users(self):
        """Save users to storage (encrypted if available)"""
        if self.secure_storage:
            # Use encrypted storage
            self.secure_storage.save_users(self.users)
        else:
            # Legacy unencrypted storage
            try:
                data = {
                    'users': [self._user_to_storage_dict(user) for user in self.users.values()],
                    'last_modified': datetime.now().isoformat()
                }
                with open(self.storage_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Error saving users: {e}")

    def _user_to_storage_dict(self, user: User) -> Dict:
        """Convert user to dictionary for storage (includes sensitive data)"""
        data = asdict(user)
        # Convert sets to lists for JSON serialization
        data['custom_permissions'] = list(data.get('custom_permissions', []))
        data['custom_module_access'] = list(data.get('custom_module_access', []))
        return data

    def _ensure_default_admin(self):
        """Ensure default admin user exists"""
        if not self.users:
            # Create default admin with secure password
            # Password: Admin@2026!Test (meets all security requirements)
            admin_user = self.create_user(
                username="admin",
                email="admin@newfold.com",
                password="Admin@2026!Test",
                roles=["super_admin"],
                full_name="System Administrator",
                created_by="system"
            )
            if admin_user:
                admin_user.must_change_password = True
                self._save_users()

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        """
        Hash password with salt using PBKDF2
        Note: When using secure_storage, this uses 600,000 iterations
        Legacy mode uses 100,000 iterations
        """
        security_manager = get_security_manager()
        if security_manager:
            # Use enhanced security (600,000 iterations)
            password_hash, _ = security_manager.hash_password(password, salt)
            return password_hash
        else:
            # Legacy mode (100,000 iterations)
            return hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            ).hex()

    @staticmethod
    def _generate_salt() -> str:
        """Generate random salt"""
        return secrets.token_hex(32)

    @staticmethod
    def _validate_password(password: str) -> tuple[bool, str]:
        """Validate password against policy"""
        policy = COMPLIANCE_CONFIG['password_policy']

        if len(password) < policy['min_length']:
            return False, f"Password must be at least {policy['min_length']} characters"

        if policy['require_uppercase'] and not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"

        if policy['require_lowercase'] and not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"

        if policy['require_numbers'] and not re.search(r'\d', password):
            return False, "Password must contain at least one number"

        if policy['require_special_chars'] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"

        return True, "Password is valid"

    @staticmethod
    def _validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[str],
        full_name: str = "",
        department: str = "",
        groups: List[str] = None,
        created_by: str = "system"
    ) -> Optional[User]:
        """Create new user"""
        # Validate inputs
        if username in self.users:
            raise ValueError(f"Username '{username}' already exists")

        if not self._validate_email(email):
            raise ValueError("Invalid email format")

        is_valid, message = self._validate_password(password)
        if not is_valid:
            raise ValueError(message)

        # Validate roles
        for role in roles:
            if role not in SYSTEM_ROLES:
                raise ValueError(f"Invalid role: {role}")

        # Create user
        salt = self._generate_salt()
        password_hash = self._hash_password(password, salt)

        user = User(
            user_id=secrets.token_hex(16),
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            roles=roles,
            full_name=full_name,
            department=department,
            groups=groups or [],
            created_by=created_by,
            password_history=[password_hash]
        )

        # Check if MFA required for role
        session_policy = COMPLIANCE_CONFIG['session_policy']
        if any(role in session_policy['require_mfa_for_roles'] for role in roles):
            user.mfa_enabled = True
            # Generate encrypted MFA secret
            if self.security_manager:
                encrypted_secret = self.security_manager.generate_mfa_secret()
                user.mfa_secret = f"enc_{encrypted_secret}"
            else:
                # Legacy mode
                user.mfa_secret = secrets.token_hex(16)

        self.users[username] = user
        self._save_users()
        return user

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with secure password verification"""
        user = self.users.get(username)
        if not user:
            return None

        # Check if account is locked
        if user.is_locked:
            if user.locked_until:
                unlock_time = datetime.fromisoformat(user.locked_until)
                if datetime.now() < unlock_time:
                    return None
                else:
                    # Unlock account
                    user.is_locked = False
                    user.locked_until = None
                    user.failed_login_attempts = 0
            else:
                return None

        # Check if account is active
        if not user.is_active:
            return None

        # Verify password using secure comparison
        password_valid = False
        if self.security_manager and self.secure_storage:
            # Try new format first (600K iterations)
            password_valid = self.security_manager.verify_password(
                password, user.password_hash, user.salt
            )

            # If that fails, try legacy format (100K iterations) for backward compatibility
            if not password_valid:
                legacy_hash = self._hash_password(password, user.salt)
                password_valid = secrets.compare_digest(legacy_hash, user.password_hash)

                # If legacy format works, upgrade the password hash
                if password_valid:
                    print(f"⚠️  Upgrading password hash for {user.username} to new format")
                    new_hash, _ = self.security_manager.hash_password(password, user.salt)
                    user.password_hash = new_hash
                    self._save_users()
        else:
            # Legacy mode - always use 100K iterations
            password_hash = self._hash_password(password, user.salt)
            password_valid = secrets.compare_digest(password_hash, user.password_hash)

        if not password_valid:
            # Increment failed attempts
            user.failed_login_attempts += 1

            max_attempts = COMPLIANCE_CONFIG['session_policy']['max_failed_login_attempts']
            if user.failed_login_attempts >= max_attempts:
                # Lock account
                user.is_locked = True
                lockout_minutes = COMPLIANCE_CONFIG['session_policy']['lockout_duration_minutes']
                user.locked_until = (datetime.now() + timedelta(minutes=lockout_minutes)).isoformat()

            self._save_users()
            return None

        # Successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.now().isoformat()
        user.last_accessed = datetime.now().isoformat()
        self._save_users()

        return user

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        user = self.authenticate(username, old_password)
        if not user:
            return False

        # Validate new password
        is_valid, message = self._validate_password(new_password)
        if not is_valid:
            raise ValueError(message)

        # Check password history
        policy = COMPLIANCE_CONFIG['password_policy']
        new_hash = self._hash_password(new_password, user.salt)

        if new_hash in user.password_history[-policy['prevent_reuse_count']:]:
            raise ValueError("Password was recently used. Please choose a different password.")

        # Update password
        user.password_hash = new_hash
        user.password_history.append(new_hash)
        user.last_password_change = datetime.now().isoformat()
        user.must_change_password = False

        self._save_users()
        return True

    def reset_password(self, username: str, new_password: str, reset_by: str) -> bool:
        """Admin password reset"""
        user = self.users.get(username)
        if not user:
            return False

        # Validate new password
        is_valid, message = self._validate_password(new_password)
        if not is_valid:
            raise ValueError(message)

        # Update password
        salt = self._generate_salt()
        user.salt = salt
        user.password_hash = self._hash_password(new_password, salt)
        user.password_history.append(user.password_hash)
        user.last_password_change = datetime.now().isoformat()
        user.must_change_password = True
        user.modified_by = reset_by
        user.modified_at = datetime.now().isoformat()

        self._save_users()
        return True

    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.users.get(username)

    def update_user(self, username: str, **kwargs) -> bool:
        """Update user attributes"""
        user = self.users.get(username)
        if not user:
            return False

        allowed_updates = [
            'email', 'full_name', 'department', 'roles', 'groups',
            'is_active', 'custom_permissions', 'custom_module_access'
        ]

        for key, value in kwargs.items():
            if key in allowed_updates:
                setattr(user, key, value)

        user.modified_at = datetime.now().isoformat()
        self._save_users()
        return True

    def delete_user(self, username: str) -> bool:
        """Delete user"""
        if username in self.users:
            del self.users[username]
            self._save_users()
            return True
        return False

    def list_users(self) -> List[Dict]:
        """List all users (without sensitive data)"""
        return [user.to_dict() for user in self.users.values()]

    def get_users_by_role(self, role: str) -> List[User]:
        """Get all users with specific role"""
        return [user for user in self.users.values() if role in user.roles]

    def get_users_by_group(self, group: str) -> List[User]:
        """Get all users in specific group"""
        return [user for user in self.users.values() if group in user.groups]

