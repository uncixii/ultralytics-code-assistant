"""Show authored error messages without exposing raw service responses or secrets."""

from urllib.error import HTTPError, URLError

from pymongo.errors import PyMongoError


class PublicError(ValueError):
    """An application validation error whose message is safe to show to users."""


class OpenRouterError(RuntimeError):
    """An OpenRouter failure with an application-authored, sanitized message."""


def error_message(error: Exception) -> str:
    """Reveal only messages explicitly authored for users, never arbitrary exceptions."""
    if isinstance(error, (PublicError, OpenRouterError)):
        return str(error)
    category = type(error).__name__
    if isinstance(error, PyMongoError):
        return (
            f"{category}. Check MONGODB_URI, Atlas network access, "
            "and the database user's permissions."
        )
    if isinstance(error, (HTTPError, URLError, TimeoutError, ConnectionError)):
        return f"{category}. Check network access, service credentials and request limits."
    if isinstance(error, FileNotFoundError):
        return f"{category}. Check the input path and fetch the pinned source if needed."
    if isinstance(error, PermissionError):
        return f"{category}. Check read/write permissions for the required local files."
    return f"{category}. Check configuration and local model availability."
