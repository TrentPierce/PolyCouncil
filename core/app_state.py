import os
import platform
import shutil
import sys
from pathlib import Path

APP_NAME = "PolyCouncil"


def _home_dir() -> Path:
    return Path.home()


def _config_home() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("APPDATA", _home_dir() / "AppData" / "Roaming"))
    if system == "Darwin":
        return _home_dir() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_CONFIG_HOME", _home_dir() / ".config"))


def _data_home() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", _home_dir() / "AppData" / "Local"))
    if system == "Darwin":
        return _home_dir() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_DATA_HOME", _home_dir() / ".local" / "share"))


def _log_home() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", _home_dir() / "AppData" / "Local"))
    if system == "Darwin":
        return _home_dir() / "Library" / "Logs"
    return Path(os.environ.get("XDG_STATE_HOME", _home_dir() / ".local" / "state"))


def app_config_dir() -> Path:
    return _config_home() / APP_NAME


def app_data_dir() -> Path:
    return _data_home() / APP_NAME


def app_log_dir() -> Path:
    return _log_home() / APP_NAME


def legacy_root_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def ensure_app_dirs() -> None:
    app_config_dir().mkdir(parents=True, exist_ok=True)
    app_data_dir().mkdir(parents=True, exist_ok=True)
    app_log_dir().mkdir(parents=True, exist_ok=True)


def migrate_legacy_file(legacy_path: Path, target_path: Path) -> None:
    if not legacy_path.exists() or target_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(legacy_path, target_path)
    except Exception:
        return


def migrate_legacy_state() -> None:
    ensure_app_dirs()
    legacy_root = legacy_root_dir()
    migrate_legacy_file(legacy_root / "council_settings.json", app_config_dir() / "settings.json")
    migrate_legacy_file(legacy_root / "council_stats.db", app_data_dir() / "council_stats.db")
    migrate_legacy_file(legacy_root / "config" / "user_personas.json", app_config_dir() / "user_personas.json")
