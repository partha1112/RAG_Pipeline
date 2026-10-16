import os
from typing import Any, Dict
from guardrails import Guard
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from transformers import pipeline
from src.utils.exceptions import FormatValidationException, HallucinationException, OutOfScopeException, PromptInjectionException, ToxicOutputException
from guardrails.validators import Validator, FailResult, PassResult, register_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


import re
from guardrails.validators import FailResult, PassResult, Validator, register_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


from src.llm import llm
from src.utils.schema import ScopeModel


MODEL_CACHE_DIR = r"C:\Users\Dell\AI_projects\Multi Agent Chat bot\deberta-prompt-injection"
MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"


os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
CLASSIFIER = pipeline(
    "text-classification",
    model=MODEL_NAME,
    model_kwargs={"cache_dir": MODEL_CACHE_DIR},
)
    


@register_validator(name="prompt_injection_validator", data_type="string")
class PromptInjectionValidator(Validator):
    def validate(self, value: str, metadata: Dict[str, Any] = {}) -> ValidationResult:
        if "my account number" in value.lower():
            return PassResult()
        
        res = CLASSIFIER(value)[0]

        if res.get("label", "").upper() in ["INJECTION", "LABEL_1"]:
            return FailResult(
                error_message="Prompt injection detected. Request blocked."
            )

        return PassResult()


@register_validator(name="scope_validator", data_type="string")
class ScopeValidator(Validator):
    def validate(self, value: str, metadata: Dict[str, Any] = {}) -> ValidationResult:
        structured_llm = llm.with_structured_output(ScopeModel)
        prompt = f"""
        You are a scope validation agent for an HR Knowledge Base (RAG) system.
        Your task is to determine whether the user's request can be answered using the HR policy document and company guidelines.

        Supported HR Operations & Topics:

        1. Employee Benefits & Leave Policies
            * Annual, sick, casual, maternity, and paternity leave rules
            * Health insurance, medical coverage, and wellness benefits
            * Holidays and office closure schedules

        2. Payroll, Compensation & Expenses
            * Salary structure, pay slips, and payout schedules
            * Tax forms (Form 16/W-2) and declaration rules
            * Travel, food, and internet allowance reimbursements

        3. Company Policies & Workplace Guidelines
            * Work hours, remote work / hybrid work policy, and attendance
            * Code of conduct, POSH policy, and ethics guidelines
            * Onboarding, probation period, and exit / notice period procedures

        Important:
            * Questions asking about leave balances, policy rules, expense submission steps, or HR procedures are VALID_SCOPE.
            * Do not mark a request as OUT_OF_SCOPE just because it contains employee details, dates, or personal context.
            * Focus on evaluating the user's intent — if it pertains to HR, employment policies, or workplace guidelines, it is in scope.

        If the user's request is related to any supported HR operations or company policies, respond ONLY with:

        VALID_SCOPE

        If the user's request is unrelated to HR or company policies (e.g., coding help, banking, general trivia, weather), respond ONLY with:

        OUT_OF_SCOPE

        User prompt:
        {value}
        """
        response = structured_llm.invoke(prompt)

        if getattr(response, "label", None) == "OUT_OF_SCOPE":
            return FailResult(
                error_message="The request received from the user is OUT OF SCOPE"
            )

        return PassResult()


def handle_injection_failure(value: str, fail_result: FailResult):
    raise PromptInjectionException(fail_result.error_message)


def handle_scope_failure(value: str, fail_result: FailResult):
    raise OutOfScopeException(fail_result.error_message)


guard = Guard().use(
    PromptInjectionValidator(on_fail=handle_injection_failure),  # type: ignore
    ScopeValidator(on_fail=handle_scope_failure),  # type: ignore
)


def validate_input(text: str) -> Dict[str, Any]:
    """Validates user input for prompt injection first, then evaluates HR scope alignment."""
    try:
        guard.validate(text)
        return {
            "is_valid": True,
            "status": "VALID_SCOPE",
            "message": text,
            "error": None,
        }
    except PromptInjectionException as e:
        return {
            "is_valid": False,
            "status": "PROMPT_INJECTION",
            "message": text,
            "error": str(e),
        }
    except OutOfScopeException as e:
        return {
            "is_valid": False,
            "status": "OUT_OF_SCOPE",
            "message": text,
            "error": str(e),
        }
    except Exception as e:
        return {
            "is_valid": False,
            "status": "ERROR",
            "message": text,
            "error": str(e),
        }


# Custom On-Fail Handlers
def handle_faithfulness_failure(value: str, fail_result: FailResult):
    raise HallucinationException(
        fail_result.error_message
    )

def handle_toxicity_failure(value: str, fail_result: FailResult):
    raise ToxicOutputException(
        fail_result.error_message 
    )

def handle_schema_failure(value: str, fail_result: FailResult):
    raise FormatValidationException(
        fail_result.error_message 
    )


import re
from guardrails.validators import FailResult, PassResult, Validator, register_validator


import re
from guardrails.validators import FailResult, PassResult, Validator, register_validator


@register_validator(name="faithfulness_validator", data_type="string")
class FaithfulnessValidator(Validator):
    def __init__(self, threshold: float = 0.6, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold

    def _normalize_text(self, text: str) -> str:
        """Fixes kerning/character tracking (e.g. 'd r e s s') while preserving word boundaries."""
        # 1. Remove markdown symbols and placeholder brackets
        text = re.sub(r"[#*_\-\[\]]", " ", text)

        # 2. Collapse single character spacing (e.g., "d r e s s" -> "dress")
        # Finds single letters separated by single spaces and joins them
        text = re.sub(r"(?<=\b[a-zA-Z])\s+(?=[a-zA-Z]\b)", "", text)

        # 3. Collapse multiple whitespace chars/newlines into a single space
        return re.sub(r"\s+", " ", text).strip().lower()

    def validate(self, value: str, metadata: dict = None) -> FailResult | PassResult:
        metadata = metadata or {}
        raw_sources = metadata.get("retrieved_context") or metadata.get("sources") or []

        if not raw_sources:
            return FailResult(error_message="No context sources provided for validation.")

        if isinstance(raw_sources, str):
            raw_context = raw_sources
        else:
            raw_context = " ".join(raw_sources)

        # Normalize context to restore spaces
        context_text = self._normalize_text(raw_context)

        # Split output response into sentences
        sentences = [s.strip() for s in re.split(r"[.!?]", value) if s.strip()]
        if not sentences:
            return PassResult()

        supported_sentences = 0
        for sentence in sentences:
            # Extract keywords (> 3 characters)
            words = [w for w in re.findall(r"\b\w+\b", sentence.lower()) if len(w) > 3]
            if not words:
                supported_sentences += 1
                continue

            # Check matching against normalized context
            matched_words = sum(1 for word in words if word in context_text)
            match_ratio = matched_words / len(words)

            if match_ratio >= 0.5:
                supported_sentences += 1

        overall_score = supported_sentences / len(sentences)

        if overall_score < self.threshold:
            return FailResult(
                error_message=f"Hallucination / ungrounded content detected. Grounding ratio ({overall_score:.2f}) < threshold ({self.threshold})."
            )
        return PassResult()


@register_validator(name="toxic_language_validator", data_type="string")
class ToxicLanguageValidator(Validator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define blocked keywords / profanity list
        self.toxic_keywords = {
            "abuse", "toxic", "profanity", "hate", "harass",
            "exploit", "malicious", "offensive", "threat"
        }

    def validate(self, value: str, metadata: dict = None) -> FailResult | PassResult:
        # Extract individual words from the output
        words = set(re.findall(r"\b\w+\b", value.lower()))
        detected_terms = words.intersection(self.toxic_keywords)

        if detected_terms:
            return FailResult(
                error_message=f"Toxic language detected: {', '.join(detected_terms)}"
            )
        return PassResult()

    
# Guard configuration with custom output validators
output_guard = Guard().use(
    FaithfulnessValidator(on_fail=handle_faithfulness_failure),  
    ToxicLanguageValidator(on_fail=handle_toxicity_failure)         
)

def vaidate_output(llm_res : str, retrieved_docs : str, user_query : str):
    validated_output = output_guard.validate(
        llm_res,
        metadata={"retrieved_context": retrieved_docs, "user_query": user_query}
    )

    return validated_output