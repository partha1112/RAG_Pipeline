class PromptInjectionException(BaseException):
    def __init__(
        self,
        message: str = "Potential prompt injection detected.",
        error_code: str = "INJ_1001",
        details: dict | None = None,
    ):
        super().__init__(message, error_code, details)


class OutOfScopeException(BaseException):
    def __init__(
        self,
        message: str = "Query is out of scope for this assistant.",
        error_code: str = "OOS_1002",
        details: dict | None = None,
    ):
        super().__init__(message, error_code, details)

class PoorRetrievalQualityException(Exception):
    def __init__(
        self,
        message: str = "Context retrieval confidence score is below threshold.",
        error_code: str = "RET_1003",
        details: str = "Unable to retrieve context with the provided query. Please rephrase your query."
    ):
        super().__init__(message)  # Passes message to standard Exception args
        self.message = message
        self.error_code = error_code
        self.details = details 


class HallucinationException(Exception):
    def __init__(
            self,
            message: str = "Generated output is not grounded in retrieved context.",
            error_code: str = "HAL_1011",
            details: dict | None = None,
        ):
            super().__init__(message, error_code, details)

class ToxicOutputException(Exception):
    def __init__(
            self,
            message: str =  "Generated output contains unsafe content.",
            error_code: str = "TOX_1012",
            details: dict | None = None,
        ):
            super().__init__(message, error_code, details)

class FormatValidationException(Exception):
    def __init__(
            self,
            message: str = "Generated output failed schema validation.",
            error_code: str = "FRM_1013",
            details: dict | None = None,
        ):
            super().__init__(message, error_code, details)
