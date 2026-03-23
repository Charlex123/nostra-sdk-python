"""
NostraHealthAI SDK

Official Python SDK for NostraHealthAI Medical AI Platform

Example:
    >>> from nostra_healthai import NostraHealthAI
    >>> client = NostraHealthAI(api_key='your-firebase-token')
    >>> response = client.chat(message='What are the symptoms of high blood pressure?')
    >>> print(response['response'])
"""

from .client import NostraHealthAI, NostraHealthAIError
from .types import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    Analysis,
    JobStatus,
    PotentialCondition,
    LabInterpretation,
    Recommendation,
    AISubscriptionPlan,
    UserAISubscription,
    AIUsageResponse,
)

try:
    from .async_client import AsyncNostraHealthAI
except ImportError:
    pass  # httpx not installed

__version__ = "2.0.0"
__all__ = [
    "NostraHealthAI",
    "NostraHealthAIError",
    "AsyncNostraHealthAI",
    "ChatRequest",
    "ChatResponse",
    "ModelInfo",
    "Analysis",
    "JobStatus",
    "PotentialCondition",
    "LabInterpretation",
    "Recommendation",
    "AISubscriptionPlan",
    "UserAISubscription",
    "AIUsageResponse",
]
