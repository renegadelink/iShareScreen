"""Optional password storage in the operating system's credential vault.

Passwords never go into the library file. On Windows they are kept in Windows
Credential Manager (DPAPI-protected, bound to the signed-in Windows account,
not roamed); elsewhere the `keyring` package is used when it is installed. Any
process running as the same user can read them, which is how every credential
vault behaves, so saving a password is always an explicit per-computer choice.
"""
from __future__ import annotations
import sys
from typing import Optional

TARGET_PREFIX = 'iShareScreen/'


class CredentialError(Exception):
    """The vault refused or failed an operation; the message is user-facing."""


class CredentialStore:
    available = False
    description = 'no credential vault'

    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, key: str, secret: str, username: str = '') -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError


class UnavailableStore(CredentialStore):
    """Used when the platform has no vault we can reach: saving is switched off."""
    def __init__(self, reason='No credential vault is available on this system.'):
        self.reason = reason
        self.description = reason

    def get(self, key):
        return None

    def set(self, key, secret, username=''):
        raise CredentialError(self.reason)

    def delete(self, key):
        pass


class MemoryCredentialStore(CredentialStore):
    """In-process store for tests."""
    available = True
    description = 'memory'

    def __init__(self):
        self.items: dict[str, tuple[str, str]] = {}
        self.fail_with: Optional[str] = None     # make set() fail, to test error handling

    def get(self, key):
        item = self.items.get(key)
        return item[0] if item else None

    def set(self, key, secret, username=''):
        if self.fail_with:
            raise CredentialError(self.fail_with)
        self.items[key] = (secret, username)

    def delete(self, key):
        self.items.pop(key, None)


class WindowsCredentialStore(CredentialStore):
    """Windows Credential Manager through advapi32 (generic credentials)."""
    available = True
    description = 'Windows Credential Manager'
    _GENERIC, _PERSIST_LOCAL_MACHINE = 1, 2
    _ERROR_NOT_FOUND = 1168

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self._ct, self._wt = ctypes, wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [('dwLowDateTime', wintypes.DWORD), ('dwHighDateTime', wintypes.DWORD)]

        class CREDENTIAL(ctypes.Structure):
            _fields_ = [('Flags', wintypes.DWORD), ('Type', wintypes.DWORD), ('TargetName', wintypes.LPWSTR),
                        ('Comment', wintypes.LPWSTR), ('LastWritten', FILETIME),
                        ('CredentialBlobSize', wintypes.DWORD), ('CredentialBlob', ctypes.POINTER(ctypes.c_ubyte)),
                        ('Persist', wintypes.DWORD), ('AttributeCount', wintypes.DWORD),
                        ('Attributes', ctypes.c_void_p), ('TargetAlias', wintypes.LPWSTR),
                        ('UserName', wintypes.LPWSTR)]
        self._CREDENTIAL = CREDENTIAL
        api = ctypes.WinDLL('advapi32', use_last_error=True)    # so get_last_error() is reliable
        api.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIAL), wintypes.DWORD]
        api.CredWriteW.restype = wintypes.BOOL
        api.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(CREDENTIAL))]
        api.CredReadW.restype = wintypes.BOOL
        api.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        api.CredDeleteW.restype = wintypes.BOOL
        api.CredFree.argtypes = [ctypes.c_void_p]
        self._api = api

    def _error(self, action):
        code = self._ct.get_last_error()
        return CredentialError(f'Windows Credential Manager could not {action} (error {code}).')

    def get(self, key):
        ct = self._ct
        pointer = ct.POINTER(self._CREDENTIAL)()
        if not self._api.CredReadW(TARGET_PREFIX + key, self._GENERIC, 0, ct.byref(pointer)):
            if ct.get_last_error() == self._ERROR_NOT_FOUND:
                return None
            raise self._error('read the saved password')
        try:
            cred = pointer.contents
            blob = ct.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
            return blob.decode('utf-16-le')
        finally:
            self._api.CredFree(pointer)

    def set(self, key, secret, username=''):
        ct = self._ct
        blob = secret.encode('utf-16-le')
        if len(blob) > 2560:
            raise CredentialError('That password is too long for Windows Credential Manager.')
        buffer = (ct.c_ubyte * len(blob)).from_buffer_copy(blob)
        cred = self._CREDENTIAL()
        cred.Type = self._GENERIC
        cred.TargetName = TARGET_PREFIX + key
        cred.Comment = 'iShareScreen saved password'
        cred.CredentialBlobSize = len(blob)
        cred.CredentialBlob = ct.cast(buffer, ct.POINTER(ct.c_ubyte))
        cred.Persist = self._PERSIST_LOCAL_MACHINE
        cred.UserName = username or 'iShareScreen'
        if not self._api.CredWriteW(ct.byref(cred), 0):
            raise self._error('save the password')

    def delete(self, key):
        if not self._api.CredDeleteW(TARGET_PREFIX + key, self._GENERIC, 0):
            if self._ct.get_last_error() != self._ERROR_NOT_FOUND:
                raise self._error('remove the saved password')


class KeyringStore(CredentialStore):
    """macOS Keychain / Secret Service through the optional `keyring` package."""
    available = True

    def __init__(self, keyring):
        self._kr = keyring
        self.description = type(keyring.get_keyring()).__name__

    def get(self, key):
        try:
            return self._kr.get_password('iShareScreen', key)
        except Exception as error:
            raise CredentialError(f'The keychain could not be read: {error}') from error

    def set(self, key, secret, username=''):
        try:
            self._kr.set_password('iShareScreen', key, secret)
        except Exception as error:
            raise CredentialError(f'The keychain refused the password: {error}') from error

    def delete(self, key):
        try:
            self._kr.delete_password('iShareScreen', key)
        except Exception as error:
            if type(error).__name__ != 'PasswordDeleteError':
                raise CredentialError(f'The keychain could not remove the password: {error}') from error


def default_store() -> CredentialStore:
    if sys.platform == 'win32':
        try:
            return WindowsCredentialStore()
        except Exception as error:      # pragma: no cover - ctypes/advapi32 should always exist on Windows
            return UnavailableStore(f'Windows Credential Manager is not reachable ({error}).')
    try:
        import keyring
        keyring.get_keyring()
        return KeyringStore(keyring)
    except Exception:
        return UnavailableStore('Saving passwords needs the optional "keyring" package on this system.')
