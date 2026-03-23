"""
Unit tests for NostraHealthAI Python SDK.
Tests SDK construction, module initialization, and error handling.

Run with: python3 tests/test_sdk.py
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock requests module before importing client
sys.modules['requests'] = MagicMock()

from nostra_healthai.client import (
    NostraHealthAI,
    NostraHealthAIError,
    SkinInfectionModule,
    EyeDiagnosisModule,
    WoundHealingModule,
    DrugVerificationModule,
    FHIRModule,
    SubscriptionModule,
)
from nostra_healthai.types import (
    AISubscriptionPlan,
    UserAISubscription,
    AIUsageResponse,
    SubscriptionPurchaseRequest,
    SubscriptionChangeRequest,
)


class TestClientInitialization(unittest.TestCase):
    """Test NostraHealthAI client construction."""

    def test_creates_client_with_api_key(self):
        client = NostraHealthAI(api_key="test-api-key")
        self.assertIsNotNone(client)
        self.assertEqual(client.api_key, "test-api-key")

    def test_default_base_url(self):
        client = NostraHealthAI(api_key="test-api-key")
        self.assertEqual(client.base_url, "https://www.api.nostrahealth.com")

    def test_custom_base_url(self):
        client = NostraHealthAI(api_key="test-api-key", base_url="https://custom.api.com")
        self.assertEqual(client.base_url, "https://custom.api.com")

    def test_strips_trailing_slash_from_base_url(self):
        client = NostraHealthAI(api_key="test-api-key", base_url="https://custom.api.com/")
        self.assertEqual(client.base_url, "https://custom.api.com")

    def test_custom_timeout(self):
        client = NostraHealthAI(api_key="test-api-key", timeout=30)
        self.assertEqual(client.timeout, 30)

    def test_default_timeout(self):
        client = NostraHealthAI(api_key="test-api-key")
        self.assertEqual(client.timeout, 60)


class TestModuleInitialization(unittest.TestCase):
    """Test that all modules are properly initialized."""

    def setUp(self):
        self.client = NostraHealthAI(api_key="test-api-key")

    def test_skin_module_initialized(self):
        self.assertIsNotNone(self.client.skin)
        self.assertIsInstance(self.client.skin, SkinInfectionModule)

    def test_eye_module_initialized(self):
        self.assertIsNotNone(self.client.eye)
        self.assertIsInstance(self.client.eye, EyeDiagnosisModule)

    def test_wound_module_initialized(self):
        self.assertIsNotNone(self.client.wound)
        self.assertIsInstance(self.client.wound, WoundHealingModule)

    def test_drug_module_initialized(self):
        self.assertIsNotNone(self.client.drug)
        self.assertIsInstance(self.client.drug, DrugVerificationModule)

    def test_fhir_module_initialized(self):
        self.assertIsNotNone(self.client.fhir)
        self.assertIsInstance(self.client.fhir, FHIRModule)

    def test_subscription_module_initialized(self):
        self.assertIsNotNone(self.client.subscriptions)
        self.assertIsInstance(self.client.subscriptions, SubscriptionModule)


class TestMimeTypeDetection(unittest.TestCase):
    """Test MIME type detection from file extensions."""

    def test_png(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.png")), "image/png")

    def test_jpg(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.jpg")), "image/jpeg")

    def test_jpeg(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.jpeg")), "image/jpeg")

    def test_gif(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.gif")), "image/gif")

    def test_webp(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.webp")), "image/webp")

    def test_pdf(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.pdf")), "application/pdf")

    def test_mp3(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.mp3")), "audio/mpeg")

    def test_wav(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.wav")), "audio/wav")

    def test_webm(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.webm")), "audio/webm")

    def test_m4a(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.m4a")), "audio/mp4")

    def test_unknown_extension(self):
        self.assertEqual(NostraHealthAI._get_mime_type(Path("test.xyz")), "application/octet-stream")


class TestErrorHandling(unittest.TestCase):
    """Test error/exception handling."""

    def test_error_with_message(self):
        err = NostraHealthAIError("Test error")
        self.assertEqual(str(err), "Test error")
        self.assertIsNone(err.status_code)
        self.assertIsNone(err.response)

    def test_error_with_status_code(self):
        err = NostraHealthAIError("Test error", status_code=400)
        self.assertEqual(str(err), "Test error")
        self.assertEqual(err.status_code, 400)

    def test_error_with_response(self):
        resp = {"error": "Bad request", "details": "Missing field"}
        err = NostraHealthAIError("Test error", status_code=400, response=resp)
        self.assertEqual(err.response, resp)
        self.assertEqual(err.response["details"], "Missing field")

    def test_error_is_exception(self):
        err = NostraHealthAIError("Test error")
        self.assertIsInstance(err, Exception)


class TestSubscriptionTypes(unittest.TestCase):
    """Test that subscription types can be constructed."""

    def test_subscription_plan_type(self):
        plan: AISubscriptionPlan = {
            "tier": "premium",
            "name": "Premium Plan",
            "price": 29.99,
        }
        self.assertEqual(plan["tier"], "premium")

    def test_user_subscription_type(self):
        sub: UserAISubscription = {
            "tier": "standard",
            "status": "active",
            "startDate": "2024-01-01",
        }
        self.assertEqual(sub["tier"], "standard")

    def test_usage_response_type(self):
        usage: AIUsageResponse = {
            "subscription": {"tier": "basic"},
            "limits": {"daily": 100},
            "usage": {"daily": 50},
            "usagePercentages": {"daily": 50.0},
            "remainingRequests": {"daily": 50},
            "resetTimes": {"daily": "2024-01-02T00:00:00Z"},
        }
        self.assertEqual(usage["subscription"]["tier"], "basic")


class TestModuleMethods(unittest.TestCase):
    """Test that all module methods exist."""

    def setUp(self):
        self.client = NostraHealthAI(api_key="test-api-key")

    def test_skin_methods(self):
        self.assertTrue(callable(getattr(self.client.skin, 'analyze', None)))
        self.assertTrue(callable(getattr(self.client.skin, 'get_job_status', None)))
        self.assertTrue(callable(getattr(self.client.skin, 'wait_for_completion', None)))
        self.assertTrue(callable(getattr(self.client.skin, 'get_all_analyses', None)))
        self.assertTrue(callable(getattr(self.client.skin, 'get_analysis', None)))
        self.assertTrue(callable(getattr(self.client.skin, 'delete_analysis', None)))
        self.assertTrue(callable(getattr(self.client.skin, 'get_supported_conditions', None)))

    def test_eye_methods(self):
        self.assertTrue(callable(getattr(self.client.eye, 'analyze', None)))
        self.assertTrue(callable(getattr(self.client.eye, 'wait_for_completion', None)))
        self.assertTrue(callable(getattr(self.client.eye, 'get_supported_conditions', None)))

    def test_wound_methods(self):
        self.assertTrue(callable(getattr(self.client.wound, 'analyze', None)))
        self.assertTrue(callable(getattr(self.client.wound, 'create_profile', None)))
        self.assertTrue(callable(getattr(self.client.wound, 'get_timeline', None)))
        self.assertTrue(callable(getattr(self.client.wound, 'get_reference_data', None)))

    def test_drug_methods(self):
        self.assertTrue(callable(getattr(self.client.drug, 'verify', None)))
        self.assertTrue(callable(getattr(self.client.drug, 'batch_verify', None)))
        self.assertTrue(callable(getattr(self.client.drug, 'get_stats', None)))

    def test_fhir_methods(self):
        self.assertTrue(callable(getattr(self.client.fhir, 'get_patient_summary', None)))
        self.assertTrue(callable(getattr(self.client.fhir, 'get_observations', None)))
        self.assertTrue(callable(getattr(self.client.fhir, 'get_conditions', None)))
        self.assertTrue(callable(getattr(self.client.fhir, 'export_bundle', None)))

    def test_subscription_methods(self):
        self.assertTrue(callable(getattr(self.client.subscriptions, 'get_plans', None)))
        self.assertTrue(callable(getattr(self.client.subscriptions, 'get_plan_details', None)))
        self.assertTrue(callable(getattr(self.client.subscriptions, 'get_my_subscription', None)))
        self.assertTrue(callable(getattr(self.client.subscriptions, 'purchase', None)))
        self.assertTrue(callable(getattr(self.client.subscriptions, 'change', None)))
        self.assertTrue(callable(getattr(self.client.subscriptions, 'cancel', None)))
        self.assertTrue(callable(getattr(self.client.subscriptions, 'get_ai_usage', None)))

    def test_top_level_methods(self):
        self.assertTrue(callable(getattr(self.client, 'chat', None)))
        self.assertTrue(callable(getattr(self.client, 'audio_chat', None)))
        self.assertTrue(callable(getattr(self.client, 'get_conversations', None)))
        self.assertTrue(callable(getattr(self.client, 'get_conversation_messages', None)))
        self.assertTrue(callable(getattr(self.client, 'analyze_file', None)))
        self.assertTrue(callable(getattr(self.client, 'get_job_status', None)))
        self.assertTrue(callable(getattr(self.client, 'wait_for_job_completion', None)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
