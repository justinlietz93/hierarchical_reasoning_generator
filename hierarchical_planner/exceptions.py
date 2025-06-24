"""
Custom exception classes for the Hierarchical Planner application.

Defines a hierarchy of exceptions for specific error conditions like
configuration issues, file processing problems, API errors, etc.
"""

class HierarchicalPlannerError(Exception):
    """Base exception class for this application."""
    pass

# --- Configuration Errors ---
class ConfigError(HierarchicalPlannerError):
    """Base class for configuration-related errors."""
    pass

class ConfigNotFoundError(ConfigError):
    """Raised when the configuration file cannot be found."""
    pass

class ConfigParsingError(ConfigError):
    """Raised when the configuration file cannot be parsed (e.g., invalid YAML)."""
    pass

class ApiKeyError(ConfigError):
    """Raised for issues related to the API key configuration or resolution."""
    pass

# --- File Processing Errors ---
class FileProcessingError(HierarchicalPlannerError):
    """Base class for file input/output errors."""
    pass

class FileNotFoundError(FileProcessingError):
    """Raised when an expected input file is not found."""
    # Note: Shadowing built-in FileNotFoundError, maybe rename?
    # Let's keep it for now for clarity within our exception hierarchy.
    pass

# Add the alias that the other modules are trying to import
PlannerFileNotFoundError = FileNotFoundError
"""Alias for FileNotFoundError to avoid shadowing built-in exception."""

class FileReadError(FileProcessingError):
    """Raised when there's an error reading from a file."""
    pass

class FileWriteError(FileProcessingError):
    """Raised when there's an error writing to a file."""
    pass

# --- API Call Errors ---
class ApiCallError(HierarchicalPlannerError):
    """Raised when an API call to the generative model fails after retries."""
    pass

class ApiResponseError(ApiCallError):
    """Raised when the API returns an error or unexpected/invalid response structure."""
    pass

class ApiBlockedError(ApiCallError):
    """Raised when the API call is blocked due to safety settings or other reasons."""
    def __init__(self, message, reason=None, ratings=None):
        super().__init__(message)
        self.reason = reason
        self.ratings = ratings

    def __str__(self):
        details = super().__str__()
        if self.reason:
            details += f" Reason: {self.reason}."
        if self.ratings:
            details += f" Safety Ratings: {self.ratings}."
        return details


# --- JSON Processing Errors ---
class JsonProcessingError(HierarchicalPlannerError):
    """Base class for JSON parsing or serialization errors."""
    pass

class JsonParsingError(JsonProcessingError):
    """Raised when parsing JSON data fails."""
    pass

class JsonSerializationError(JsonProcessingError):
    """Raised when serializing data to JSON fails."""
    pass


# --- Plan Generation/Validation Errors ---
class PlanError(HierarchicalPlannerError):
    """Base class for errors during plan generation or validation."""
    pass

class PlanGenerationError(PlanError):
    """Raised when a crucial part of the plan (e.g., phases) cannot be generated."""
    pass

class PlanValidationError(PlanError):
    """Raised when the generated plan fails structural or content validation."""
    pass

# --- Project Builder Errors ---
class ProjectBuilderError(HierarchicalPlannerError):
    """Raised when there's an error during project building."""
    pass

class LLMClientError(HierarchicalPlannerError):
    """Raised when there's an error with LLM client operations."""
    pass

class ValidationError(HierarchicalPlannerError):
    """Raised when validation fails."""
    pass
