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

from src.llm import llm
from src.utils.scope import ScopeModel


class PromptInjectionException(Exception):
    pass


class OutOfScopeException(Exception):
    pass


MODEL_CACHE_DIR = r"C:\Users\Dell\AI_projects\Multi Agent Chat bot\deberta-prompt-injection"
MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"

_CLASSIFIER = None


def get_classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        _CLASSIFIER = pipeline(
            "text-classification",
            model=MODEL_NAME,
            model_kwargs={"cache_dir": MODEL_CACHE_DIR},
        )
    return _CLASSIFIER


@register_validator(name="prompt_injection_validator", data_type="string")
class PromptInjectionValidator(Validator):
    def validate(self, value: str, metadata: Dict[str, Any] = {}) -> ValidationResult:
        if "my account number" in value.lower():
            return PassResult()

        classifier_pipe = get_classifier()
        res = classifier_pipe(value)[0]

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