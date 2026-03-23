from typing import Optional

try:
    import keyring
    from keyring.errors import KeyringError
except Exception:  # pragma: no cover - exercised via fallback path
    keyring = None

    class KeyringError(Exception):
        pass


SERVICE_NAME = "PolyCouncil"


def secure_store_available() -> bool:
    return keyring is not None


def get_secret(name: str) -> str:
    if keyring is None:
        return ""
    try:
        return keyring.get_password(SERVICE_NAME, name) or ""
    except KeyringError:
        return ""
    except Exception:
        return ""


def set_secret(name: str, value: str) -> bool:
    if keyring is None:
        return False
    try:
        if value:
            keyring.set_password(SERVICE_NAME, name, value)
        else:
            keyring.delete_password(SERVICE_NAME, name)
        return True
    except KeyringError:
        return False
    except Exception:
        return False


def delete_secret(name: str) -> bool:
    if keyring is None:
        return False
    try:
        keyring.delete_password(SERVICE_NAME, name)
        return True
    except KeyringError:
        return False
    except Exception:
        return False
