"""
Enterprise-Grade Security Manager
Provides encryption, secure key management, and hardened credential storage
"""

import os
import secrets
import base64
import json
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class SecurityManager:
    """
    Enterprise security manager with multiple layers of protection:
    - AES-256-GCM encryption for data at rest
    - Argon2/PBKDF2 for password hashing (600,000+ iterations)
    - Secure key derivation and storage
    - Encrypted MFA secrets
    - Session token encryption
    - Key rotation support
    """

    # Security constants
    PBKDF2_ITERATIONS = 600_000  # OWASP 2023 recommendation
    SALT_LENGTH = 32
    KEY_LENGTH = 32

    def __init__(self, key_storage_path: Optional[str] = None):
        """Initialize security manager with encrypted key storage"""
        if key_storage_path is None:
            auth_dir = Path(__file__).parent
            key_storage_path = auth_dir / ".keys"

        self.key_storage_path = Path(key_storage_path)
        self.key_storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions on key storage
        if not self.key_storage_path.exists():
            self.key_storage_path.touch(mode=0o600)
        else:
            os.chmod(self.key_storage_path, 0o600)

        self._master_key = self._get_or_create_master_key()
        self._encryption_keys = self._load_or_create_encryption_keys()

    def _get_or_create_master_key(self) -> bytes:
        """
        Get or create master encryption key
        In production, this should be stored in a secure key management service (KMS)
        like AWS KMS, Azure Key Vault, or HashiCorp Vault
        """
        env_key = os.environ.get('JARVIS_MASTER_KEY')

        if env_key:
            # Use environment variable (preferred for production)
            return base64.b64decode(env_key)

        # For development: derive from machine-specific entropy
        # WARNING: In production, use a proper KMS!
        key_file = self.key_storage_path.parent / ".master_key"

        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()

        # Generate new master key
        master_key = AESGCM.generate_key(bit_length=256)

        with open(key_file, 'wb') as f:
            f.write(master_key)

        os.chmod(key_file, 0o600)

        print(f"""
        ⚠️  SECURITY WARNING ⚠️
        A new master encryption key has been generated at: {key_file}
        
        For PRODUCTION use, you MUST:
        1. Store this key in a secure Key Management Service (KMS)
        2. Set the JARVIS_MASTER_KEY environment variable
        3. Delete the local key file
        4. Restrict file system access
        
        Master Key (base64): {base64.b64encode(master_key).decode()}
        """)

        return master_key

    def _load_or_create_encryption_keys(self) -> Dict[str, bytes]:
        """Load or create data encryption keys (DEKs)"""
        if self.key_storage_path.exists():
            try:
                with open(self.key_storage_path, 'rb') as f:
                    encrypted_data = f.read()

                if encrypted_data:
                    # Decrypt keys using master key
                    aesgcm = AESGCM(self._master_key)
                    nonce = encrypted_data[:12]
                    ciphertext = encrypted_data[12:]
                    decrypted = aesgcm.decrypt(nonce, ciphertext, None)
                    return json.loads(decrypted.decode())
            except Exception as e:
                print(f"Error loading encryption keys: {e}")

        # Create new keys
        keys = {
            'user_data': base64.b64encode(AESGCM.generate_key(bit_length=256)).decode(),
            'mfa_secrets': base64.b64encode(AESGCM.generate_key(bit_length=256)).decode(),
            'session_tokens': base64.b64encode(AESGCM.generate_key(bit_length=256)).decode(),
            'created_at': datetime.now().isoformat(),
            'version': 1
        }

        self._save_encryption_keys(keys)
        return keys

    def _save_encryption_keys(self, keys: Dict[str, Any]):
        """Encrypt and save data encryption keys"""
        # Encrypt keys with master key
        aesgcm = AESGCM(self._master_key)
        nonce = os.urandom(12)
        plaintext = json.dumps(keys).encode()
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        with open(self.key_storage_path, 'wb') as f:
            f.write(nonce + ciphertext)

        os.chmod(self.key_storage_path, 0o600)

    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        Hash password using PBKDF2-HMAC-SHA256 with 600,000 iterations
        Returns: (password_hash, salt)

        Note: For even better security, consider using Argon2id:
        from argon2 import PasswordHasher
        ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)
        """
        if salt is None:
            salt = secrets.token_hex(self.SALT_LENGTH)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_LENGTH,
            salt=salt.encode('utf-8'),
            iterations=self.PBKDF2_ITERATIONS,
            backend=default_backend()
        )

        password_hash = base64.b64encode(
            kdf.derive(password.encode('utf-8'))
        ).decode('utf-8')

        return password_hash, salt

    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        computed_hash, _ = self.hash_password(password, salt)
        # Use constant-time comparison to prevent timing attacks
        return secrets.compare_digest(computed_hash, password_hash)

    def encrypt_sensitive_data(self, data: str, key_name: str = 'user_data') -> str:
        """
        Encrypt sensitive data using AES-256-GCM
        Returns: base64-encoded encrypted data with nonce
        """
        key = base64.b64decode(self._encryption_keys[key_name])
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)

        ciphertext = aesgcm.encrypt(
            nonce,
            data.encode('utf-8'),
            None  # No additional authenticated data
        )

        # Combine nonce and ciphertext
        encrypted = nonce + ciphertext
        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt_sensitive_data(self, encrypted_data: str, key_name: str = 'user_data') -> str:
        """Decrypt sensitive data"""
        try:
            key = base64.b64decode(self._encryption_keys[key_name])
            aesgcm = AESGCM(key)

            encrypted_bytes = base64.b64decode(encrypted_data)
            nonce = encrypted_bytes[:12]
            ciphertext = encrypted_bytes[12:]

            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)

    def generate_mfa_secret(self) -> str:
        """Generate encrypted MFA secret"""
        secret = secrets.token_hex(20)  # 160 bits for TOTP
        # Encrypt the secret before storage
        return self.encrypt_sensitive_data(secret, 'mfa_secrets')

    def decrypt_mfa_secret(self, encrypted_secret: str) -> str:
        """Decrypt MFA secret for TOTP generation"""
        return self.decrypt_sensitive_data(encrypted_secret, 'mfa_secrets')

    def encrypt_session_token(self, token: str) -> str:
        """Encrypt session token"""
        return self.encrypt_sensitive_data(token, 'session_tokens')

    def decrypt_session_token(self, encrypted_token: str) -> str:
        """Decrypt session token"""
        return self.decrypt_sensitive_data(encrypted_token, 'session_tokens')

    def rotate_encryption_keys(self, key_names: Optional[list] = None) -> Dict[str, str]:
        """
        Rotate encryption keys
        Returns mapping of old key names to new key names for re-encryption
        """
        if key_names is None:
            key_names = ['user_data', 'mfa_secrets', 'session_tokens']

        old_keys = self._encryption_keys.copy()
        rotation_map = {}

        for key_name in key_names:
            # Generate new key
            new_key = base64.b64encode(AESGCM.generate_key(bit_length=256)).decode()

            # Store old key with timestamp
            old_key_name = f"{key_name}_old_{datetime.now().timestamp()}"
            rotation_map[key_name] = old_key_name

            self._encryption_keys[old_key_name] = self._encryption_keys[key_name]
            self._encryption_keys[key_name] = new_key

        self._encryption_keys['last_rotation'] = datetime.now().isoformat()
        self._save_encryption_keys(self._encryption_keys)

        return rotation_map

    def re_encrypt_data(self, encrypted_data: str, old_key_name: str, new_key_name: str) -> str:
        """Re-encrypt data with new key after rotation"""
        # Decrypt with old key
        plaintext = self.decrypt_sensitive_data(encrypted_data, old_key_name)
        # Encrypt with new key
        return self.encrypt_sensitive_data(plaintext, new_key_name)

    def generate_audit_hash(self, data: Dict[str, Any]) -> str:
        """
        Generate tamper-proof hash for audit logs
        Uses HMAC-SHA256 with master key
        """
        from cryptography.hazmat.primitives import hmac

        h = hmac.HMAC(self._master_key, hashes.SHA256(), backend=default_backend())
        h.update(json.dumps(data, sort_keys=True).encode('utf-8'))
        return base64.b64encode(h.finalize()).decode('utf-8')

    def verify_audit_hash(self, data: Dict[str, Any], audit_hash: str) -> bool:
        """Verify audit log hasn't been tampered with"""
        computed_hash = self.generate_audit_hash(data)
        return secrets.compare_digest(computed_hash, audit_hash)

    def secure_delete(self, file_path: Path, passes: int = 3):
        """
        Securely delete file by overwriting with random data
        Note: On SSDs, this may not be fully effective due to wear leveling
        """
        if not file_path.exists():
            return

        size = file_path.stat().st_size

        with open(file_path, 'ba+', buffering=0) as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(size))

        file_path.unlink()

    def get_security_info(self) -> Dict[str, Any]:
        """Get security configuration information"""
        return {
            'password_hashing': {
                'algorithm': 'PBKDF2-HMAC-SHA256',
                'iterations': self.PBKDF2_ITERATIONS,
                'salt_length': self.SALT_LENGTH,
                'key_length': self.KEY_LENGTH
            },
            'encryption': {
                'algorithm': 'AES-256-GCM',
                'key_length': 256,
                'keys_in_use': len([k for k in self._encryption_keys.keys()
                                   if not k.startswith('_') and k not in ['created_at', 'version', 'last_rotation']])
            },
            'key_storage': {
                'master_key_source': 'environment' if os.environ.get('JARVIS_MASTER_KEY') else 'file',
                'key_file': str(self.key_storage_path),
                'created_at': self._encryption_keys.get('created_at'),
                'last_rotation': self._encryption_keys.get('last_rotation', 'Never')
            },
            'recommendations': self._get_security_recommendations()
        }

    def _get_security_recommendations(self) -> list:
        """Get security recommendations based on current configuration"""
        recommendations = []

        if not os.environ.get('JARVIS_MASTER_KEY'):
            recommendations.append({
                'severity': 'HIGH',
                'message': 'Master key stored in file system. Use KMS or environment variable in production.',
                'action': 'Set JARVIS_MASTER_KEY environment variable'
            })

        if not self._encryption_keys.get('last_rotation'):
            recommendations.append({
                'severity': 'MEDIUM',
                'message': 'Encryption keys have never been rotated.',
                'action': 'Implement key rotation policy (recommend quarterly)'
            })

        # Check file permissions
        try:
            stat_info = self.key_storage_path.stat()
            if stat_info.st_mode & 0o077:
                recommendations.append({
                    'severity': 'HIGH',
                    'message': 'Key file has overly permissive permissions.',
                    'action': 'Set permissions to 0600 (owner read/write only)'
                })
        except:
            pass

        return recommendations


# Singleton instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get singleton security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

