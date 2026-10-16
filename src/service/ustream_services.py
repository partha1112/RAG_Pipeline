from datetime import timedelta

import httpx
from aiobreaker import CircuitBreaker, CircuitBreakerError
from fastapi import HTTPException

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

auth_breaker = CircuitBreaker(
    fail_max=3,
    timeout_duration=timedelta(seconds=30),
)

@auth_breaker
async def fetch_user_data(user_id: int):
    url = "http://127.0.0.1:8002/get_user_data"
    params = {"user_id": user_id}

    async with httpx.AsyncClient(timeout=3.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

async def fetch_user_data_with_fallback(user_id: int):
    try:
        return await fetch_user_data(user_id)
    except CircuitBreakerError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Authorization service temporarily unavailable",
                "circuit_open": True,
            },
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Authorization service is unavailable",
                "circuit_open": False,
            },
        ) from error

@retry(
    retry=retry_if_exception_type((
        TimeoutError,
        ConnectionError,
        HTTPException
    )),
    wait=wait_random_exponential(
        multiplier=1,
        max=10,
    ),
    stop=stop_after_attempt(4),
    reraise=True,
)


async def fetch_user_data_with_retry(user_id: str):
    return await fetch_user_data_with_fallback(user_id)