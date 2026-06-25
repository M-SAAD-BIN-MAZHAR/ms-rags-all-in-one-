"""Custom exception hierarchy for MS_RAG.

All MS_RAG exceptions inherit from MSRAGError so callers can catch the
base class when they want to handle any framework error uniformly.
"""


class MSRAGError(Exception):
    """Base exception for all MS_RAG errors."""


class ConnectionError(MSRAGError):
    """Raised when a vector DB or external service connection fails.

    Attributes:
        db_type: The vector DB type identifier (e.g. "pinecone", "chroma").
        original: The underlying exception that caused the failure.
    """

    def __init__(self, message: str, db_type: str = "", original: Exception | None = None) -> None:
        super().__init__(message)
        self.db_type = db_type
        self.original = original


class IngestionError(MSRAGError):
    """Raised when document loading, chunking, or embedding fails.

    Attributes:
        document_path: The path or URL of the document that failed.
        original: The underlying exception that caused the failure.
    """

    def __init__(self, message: str, document_path: str = "", original: Exception | None = None) -> None:
        super().__init__(message)
        self.document_path = document_path
        self.original = original


class CredentialError(MSRAGError):
    """Raised when required credentials are missing or invalid.

    Attributes:
        provider_id: The provider whose credentials are missing (e.g. "openai").
        field: The specific credential field that is missing.
    """

    def __init__(self, message: str, provider_id: str = "", field: str = "") -> None:
        super().__init__(message)
        self.provider_id = provider_id
        self.field = field


class SessionLoadError(MSRAGError):
    """Raised when a Pipeline_Config JSON file cannot be found or parsed.

    Attributes:
        file_path: The path to the config file that failed to load.
        original: The underlying exception (FileNotFoundError, JSONDecodeError, etc.).
    """

    def __init__(self, message: str, file_path: str = "", original: Exception | None = None) -> None:
        super().__init__(message)
        self.file_path = file_path
        self.original = original


class ValidationError(MSRAGError):
    """Raised by validate_numeric() when a value is outside its allowed range.

    Attributes:
        field_name: The name of the field that failed validation.
        value: The value that was rejected.
        min_val: The minimum allowed value (inclusive).
        max_val: The maximum allowed value (inclusive).
    """

    def __init__(
        self,
        message: str,
        field_name: str = "",
        value: int | float | None = None,
        min_val: int | float | None = None,
        max_val: int | float | None = None,
    ) -> None:
        super().__init__(message)
        self.field_name = field_name
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
