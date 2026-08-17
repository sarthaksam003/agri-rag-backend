from .config import get_settings

__all__ = ["get_settings", "get_constants"]


def get_constants():
    from .constants import get_constants as _get_constants

    return _get_constants()