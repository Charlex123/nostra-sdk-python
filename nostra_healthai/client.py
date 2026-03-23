"""
NostraHealthAI SDK Client

Official Python client for interacting with NostraHealthAI Medical AI Platform
"""

import time
from typing import Optional, Dict, Any, Union, BinaryIO, List
import requests
from pathlib import Path

from .types import (
    ChatRequest, ChatResponse, JobStatus, Conversation, ChatMessage,
    SkinInfectionAnalysis, SupportedSkinConditions,
    EyeDiagnosisAnalysis, SupportedEyeConditions,
    WoundHealingAnalysis, WoundProfile, WoundTimeline, WoundReferenceData,
    DrugVerificationResult, DrugVerificationStats,
    FHIRRecord, FHIRPatientSummary, FHIRBundle, FHIRSearchParams,
    AISubscriptionPlan, UserAISubscription, AIUsageResponse,
    SubscriptionPurchaseRequest, SubscriptionChangeRequest,
)


class NostraHealthAIError(Exception):
    """NostraHealthAI API Error"""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class NostraHealthAI:
    """
    NostraHealthAI SDK Client

    Official SDK for interacting with NostraHealthAI Medical AI Platform

    Args:
        api_key: Firebase authentication token
        base_url: API base URL (default: https://www.api.nostrahealth.com)
        timeout: Request timeout in seconds (default: 60)

    Example:
        >>> client = NostraHealthAI(api_key='your-api-key')
        >>>
        >>> # Chat with the AI
        >>> response = client.chat(message='What does high cholesterol mean?')
        >>> print(response['response'])
        >>>
        >>> # Skin Analysis
        >>> job_id = client.skin.analyze(file='./skin_image.jpg')
        >>> result = client.skin.wait_for_completion(job_id)
        >>>
        >>> # Eye Diagnosis
        >>> job_id = client.eye.analyze(file='./eye_image.jpg')
        >>> result = client.eye.wait_for_completion(job_id)
        >>>
        >>> # FHIR Records
        >>> summary = client.fhir.get_patient_summary()
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://www.api.nostrahealth.com",
        timeout: int = 60,
    ):
        """Initialize NostraHealthAI client"""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

        # Initialize sub-modules
        self.skin = SkinInfectionModule(self)
        self.eye = EyeDiagnosisModule(self)
        self.wound = WoundHealingModule(self)
        self.drug = DrugVerificationModule(self)
        self.fhir = FHIRModule(self)
        self.subscriptions = SubscriptionModule(self)

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        files: Optional[Dict] = None,
        data: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request to API"""
        url = f"{self.base_url}{endpoint}"

        try:
            headers = self.session.headers.copy()
            if files:
                headers.pop("Content-Type", None)

            response = self.session.request(
                method=method,
                url=url,
                json=json,
                files=files,
                data=data,
                timeout=self.timeout,
                headers=headers,
                **kwargs,
            )

            response.raise_for_status()
            result = response.json()

            if not result.get("success", True):
                raise NostraHealthAIError(
                    result.get("error", "Request failed"),
                    response.status_code,
                    result,
                )

            return result

        except requests.exceptions.HTTPError as e:
            error_data = e.response.json() if e.response.content else {}
            raise NostraHealthAIError(
                error_data.get("error", str(e)),
                e.response.status_code,
                error_data,
            )
        except requests.exceptions.RequestException as e:
            raise NostraHealthAIError(str(e))

    @staticmethod
    def _get_mime_type(file_path: Path) -> str:
        """Determine MIME type from file extension"""
        extension = file_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".webm": "audio/webm",
            ".m4a": "audio/mp4",
        }
        return mime_types.get(extension, "application/octet-stream")

    def _prepare_file(self, file: Union[str, Path, BinaryIO], field_name: str = "file") -> Dict:
        """Prepare file for upload"""
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            return {field_name: (file_path.name, open(file_path, "rb"), self._get_mime_type(file_path))}
        else:
            return {field_name: file}

    # =========================================================================
    # MEDICAL CHAT
    # =========================================================================

    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Send a message to the medical AI assistant

        Args:
            message: User message to send
            conversation_id: Optional conversation ID to continue existing conversation

        Returns:
            Chat response with AI message and model info
        """
        request_data: ChatRequest = {"message": message}
        if conversation_id:
            request_data["conversationId"] = conversation_id

        data = self._request("POST", "/api/v1/ai/chat", json=request_data)
        return data["data"]

    def audio_chat(
        self,
        audio_file: Union[str, Path, BinaryIO],
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send an audio message for voice-based conversation

        Args:
            audio_file: Audio file path or file object
            conversation_id: Optional conversation ID

        Returns:
            Dict with transcription, response, and audio URL
        """
        files = self._prepare_file(audio_file, "audio")
        data_dict = {}
        if conversation_id:
            data_dict["conversationId"] = conversation_id

        result = self._request("POST", "/api/v1/ai/audio-chat", files=files, data=data_dict)
        return result["data"]

    def get_conversations(self) -> List[Conversation]:
        """Get all conversations for the current user"""
        data = self._request("GET", "/api/v1/ai/conversations")
        return data["data"]

    def get_conversation_messages(self, conversation_id: str) -> List[ChatMessage]:
        """Get messages for a specific conversation"""
        data = self._request("GET", f"/api/v1/ai/conversations/{conversation_id}/messages")
        return data["data"]

    # =========================================================================
    # MEDICAL FILE ANALYSIS
    # =========================================================================

    def analyze_file(self, file_path: Union[str, Path, BinaryIO]) -> str:
        """
        Analyze a medical file (image, lab report, etc.)

        Args:
            file_path: Path to file or file-like object

        Returns:
            Job ID for polling the analysis result
        """
        files = self._prepare_file(file_path)
        data = self._request("POST", "/api/v1/ai/analyze", files=files)
        return data["jobId"]

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status and results of a file analysis job"""
        data = self._request("GET", f"/api/v1/ai/job/{job_id}")
        return {
            "jobId": data["jobId"],
            "status": data["status"],
            "progress": data["progress"],
            "metadata": data["metadata"],
            "data": data.get("data"),
            "error": data.get("error"),
        }

    def wait_for_job_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        max_attempts: int = 60,
    ) -> JobStatus:
        """
        Poll a file analysis job until completion

        Args:
            job_id: Job ID returned from analyze_file
            poll_interval: Polling interval in seconds (default: 2.0)
            max_attempts: Maximum polling attempts (default: 60)

        Returns:
            Completed job status with results
        """
        attempts = 0

        while attempts < max_attempts:
            status = self.get_job_status(job_id)

            if status["status"] == "completed":
                return status

            if status["status"] == "failed":
                error_msg = status.get("error", "Unknown error")
                raise NostraHealthAIError(f"Job failed: {error_msg}")

            time.sleep(poll_interval)
            attempts += 1

        raise NostraHealthAIError(f"Job polling timeout after {max_attempts} attempts")

    def get_user_jobs(self) -> list:
        """Get all analysis jobs for the current user"""
        data = self._request("GET", "/api/v1/ai/jobs")
        return data["jobs"]


# =============================================================================
# SKIN INFECTION MODULE
# =============================================================================

class SkinInfectionModule:
    """Skin infection analysis module"""

    def __init__(self, client: NostraHealthAI):
        self._client = client

    def analyze(
        self,
        file: Union[str, Path, BinaryIO],
        affected_area: Optional[str] = None,
        duration: Optional[str] = None,
        symptoms: Optional[List[str]] = None,
        previous_treatments: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None,
        medical_history: Optional[List[str]] = None,
    ) -> str:
        """
        Analyze a skin image for infections and conditions

        Returns:
            Job ID for polling results
        """
        files = self._client._prepare_file(file)
        data = {}

        if affected_area:
            data["affectedArea"] = affected_area
        if duration:
            data["duration"] = duration
        if symptoms:
            data["symptoms"] = ",".join(symptoms)
        if previous_treatments:
            data["previousTreatments"] = ",".join(previous_treatments)
        if allergies:
            data["allergies"] = ",".join(allergies)
        if medical_history:
            data["medicalHistory"] = ",".join(medical_history)

        result = self._client._request("POST", "/api/v1/ai/skin-infections", files=files, data=data)
        return result["jobId"]

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a skin analysis job"""
        return self._client._request("GET", f"/api/v1/ai/skin-infections/job/{job_id}")

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        max_attempts: int = 60,
    ) -> JobStatus:
        """Wait for a skin analysis job to complete"""
        attempts = 0

        while attempts < max_attempts:
            status = self.get_job_status(job_id)

            if status["status"] == "completed":
                return status
            if status["status"] == "failed":
                raise NostraHealthAIError(f"Skin analysis failed: {status.get('error')}")

            time.sleep(poll_interval)
            attempts += 1

        raise NostraHealthAIError("Skin analysis timeout")

    def get_all_analyses(self) -> List[SkinInfectionAnalysis]:
        """Get all skin analyses for the current user"""
        data = self._client._request("GET", "/api/v1/ai/skin-infections")
        return data["data"]

    def get_analysis(self, analysis_id: str) -> SkinInfectionAnalysis:
        """Get a specific skin analysis by ID"""
        data = self._client._request("GET", f"/api/v1/ai/skin-infections/{analysis_id}")
        return data["data"]

    def delete_analysis(self, analysis_id: str) -> None:
        """Delete a skin analysis"""
        self._client._request("DELETE", f"/api/v1/ai/skin-infections/{analysis_id}")

    def get_supported_conditions(self) -> SupportedSkinConditions:
        """Get list of supported skin conditions"""
        data = self._client._request("GET", "/api/v1/ai/skin-infections/conditions")
        return data["data"]


# =============================================================================
# EYE DIAGNOSIS MODULE
# =============================================================================

class EyeDiagnosisModule:
    """Eye diagnosis analysis module"""

    def __init__(self, client: NostraHealthAI):
        self._client = client

    def analyze(
        self,
        file: Union[str, Path, BinaryIO],
        eye_side: Optional[str] = None,
        symptoms: Optional[List[str]] = None,
        symptom_duration: Optional[str] = None,
        pain_level: Optional[int] = None,
        vision_changes: Optional[List[str]] = None,
        medical_history: Optional[List[str]] = None,
        current_medications: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None,
        family_history: Optional[List[str]] = None,
        last_eye_exam: Optional[str] = None,
        wearing_corrective_lenses: Optional[bool] = None,
        lens_type: Optional[str] = None,
    ) -> str:
        """
        Analyze an eye image for conditions

        Returns:
            Job ID for polling results
        """
        files = self._client._prepare_file(file)
        data = {}

        if eye_side:
            data["eyeSide"] = eye_side
        if symptoms:
            data["symptoms"] = ",".join(symptoms)
        if symptom_duration:
            data["symptomDuration"] = symptom_duration
        if pain_level is not None:
            data["painLevel"] = str(pain_level)
        if vision_changes:
            data["visionChanges"] = ",".join(vision_changes)
        if medical_history:
            data["medicalHistory"] = ",".join(medical_history)
        if current_medications:
            data["currentMedications"] = ",".join(current_medications)
        if allergies:
            data["allergies"] = ",".join(allergies)
        if family_history:
            data["familyHistory"] = ",".join(family_history)
        if last_eye_exam:
            data["lastEyeExam"] = last_eye_exam
        if wearing_corrective_lenses is not None:
            data["wearingCorrectiveLenses"] = str(wearing_corrective_lenses).lower()
        if lens_type:
            data["lensType"] = lens_type

        result = self._client._request("POST", "/api/v1/ai/eye-diagnosis", files=files, data=data)
        return result["jobId"]

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of an eye diagnosis job"""
        return self._client._request("GET", f"/api/v1/ai/eye-diagnosis/job/{job_id}")

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        max_attempts: int = 60,
    ) -> JobStatus:
        """Wait for an eye diagnosis job to complete"""
        attempts = 0

        while attempts < max_attempts:
            status = self.get_job_status(job_id)

            if status["status"] == "completed":
                return status
            if status["status"] == "failed":
                raise NostraHealthAIError(f"Eye diagnosis failed: {status.get('error')}")

            time.sleep(poll_interval)
            attempts += 1

        raise NostraHealthAIError("Eye diagnosis timeout")

    def get_all_analyses(self, limit: Optional[int] = None, last_doc_id: Optional[str] = None) -> Dict:
        """Get all eye diagnoses for the current user"""
        params = {}
        if limit:
            params["limit"] = limit
        if last_doc_id:
            params["lastDocId"] = last_doc_id

        endpoint = "/api/v1/ai/eye-diagnosis"
        if params:
            endpoint += "?" + "&".join(f"{k}={v}" for k, v in params.items())

        return self._client._request("GET", endpoint)

    def get_analysis(self, analysis_id: str) -> EyeDiagnosisAnalysis:
        """Get a specific eye diagnosis by ID"""
        data = self._client._request("GET", f"/api/v1/ai/eye-diagnosis/{analysis_id}")
        return data["data"]

    def delete_analysis(self, analysis_id: str) -> None:
        """Delete an eye diagnosis"""
        self._client._request("DELETE", f"/api/v1/ai/eye-diagnosis/{analysis_id}")

    def get_supported_conditions(self) -> SupportedEyeConditions:
        """Get list of supported eye conditions"""
        data = self._client._request("GET", "/api/v1/ai/eye-diagnosis/conditions")
        return data["data"]


# =============================================================================
# WOUND HEALING MODULE
# =============================================================================

class WoundHealingModule:
    """Wound healing analysis and tracking module"""

    def __init__(self, client: NostraHealthAI):
        self._client = client

    def analyze(
        self,
        file: Union[str, Path, BinaryIO],
        wound_profile_id: Optional[str] = None,
        wound_type: Optional[str] = None,
        body_location: Optional[str] = None,
        pain_level: Optional[int] = None,
        symptoms: Optional[List[str]] = None,
        current_treatments: Optional[List[str]] = None,
        recent_changes: Optional[str] = None,
    ) -> str:
        """
        Analyze a wound image

        Returns:
            Job ID for polling results
        """
        files = self._client._prepare_file(file)
        data = {}

        if wound_profile_id:
            data["woundProfileId"] = wound_profile_id
        if wound_type:
            data["woundType"] = wound_type
        if body_location:
            data["bodyLocation"] = body_location
        if pain_level is not None:
            data["painLevel"] = str(pain_level)
        if symptoms:
            data["symptoms"] = ",".join(symptoms)
        if current_treatments:
            data["currentTreatments"] = ",".join(current_treatments)
        if recent_changes:
            data["recentChanges"] = recent_changes

        result = self._client._request("POST", "/api/v1/ai/wound-healing/analyze", files=files, data=data)
        return result["jobId"]

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a wound analysis job"""
        return self._client._request("GET", f"/api/v1/ai/wound-healing/job/{job_id}")

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        max_attempts: int = 60,
    ) -> JobStatus:
        """Wait for a wound analysis job to complete"""
        attempts = 0

        while attempts < max_attempts:
            status = self.get_job_status(job_id)

            if status["status"] == "completed":
                return status
            if status["status"] == "failed":
                raise NostraHealthAIError(f"Wound analysis failed: {status.get('error')}")

            time.sleep(poll_interval)
            attempts += 1

        raise NostraHealthAIError("Wound analysis timeout")

    def get_all_analyses(self, limit: Optional[int] = None) -> List[WoundHealingAnalysis]:
        """Get all wound analyses for the current user"""
        endpoint = "/api/v1/ai/wound-healing/analyses"
        if limit:
            endpoint += f"?limit={limit}"

        data = self._client._request("GET", endpoint)
        return data["data"]

    def get_analysis(self, analysis_id: str) -> WoundHealingAnalysis:
        """Get a specific wound analysis by ID"""
        data = self._client._request("GET", f"/api/v1/ai/wound-healing/analyses/{analysis_id}")
        return data["data"]

    def delete_analysis(self, analysis_id: str) -> None:
        """Delete a wound analysis"""
        self._client._request("DELETE", f"/api/v1/ai/wound-healing/analyses/{analysis_id}")

    # --- Wound Profiles ---

    def create_profile(
        self,
        name: str,
        wound_type: str,
        body_location: str,
        body_location_details: Optional[str] = None,
        etiology: Optional[str] = None,
        wound_onset_date: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> WoundProfile:
        """Create a new wound profile for tracking"""
        payload = {
            "name": name,
            "woundType": wound_type,
            "bodyLocation": body_location,
        }
        if body_location_details:
            payload["bodyLocationDetails"] = body_location_details
        if etiology:
            payload["etiology"] = etiology
        if wound_onset_date:
            payload["woundOnsetDate"] = wound_onset_date
        if notes:
            payload["notes"] = notes

        data = self._client._request("POST", "/api/v1/ai/wound-healing/profiles", json=payload)
        return data["data"]

    def get_profiles(self) -> Dict:
        """Get all wound profiles for the current user"""
        data = self._client._request("GET", "/api/v1/ai/wound-healing/profiles")
        return data["data"]

    def get_profile(self, profile_id: str) -> Dict:
        """Get a specific wound profile with recent analyses"""
        data = self._client._request("GET", f"/api/v1/ai/wound-healing/profiles/{profile_id}")
        return data["data"]

    def update_profile(self, profile_id: str, **updates) -> WoundProfile:
        """Update a wound profile"""
        data = self._client._request("PUT", f"/api/v1/ai/wound-healing/profiles/{profile_id}", json=updates)
        return data["data"]

    def archive_profile(self, profile_id: str) -> None:
        """Archive a wound profile (mark as healed)"""
        self._client._request("POST", f"/api/v1/ai/wound-healing/profiles/{profile_id}/archive")

    def delete_profile(self, profile_id: str) -> None:
        """Delete a wound profile and all associated analyses"""
        self._client._request("DELETE", f"/api/v1/ai/wound-healing/profiles/{profile_id}")

    def get_timeline(self, profile_id: str) -> Optional[WoundTimeline]:
        """Get wound healing timeline for a profile"""
        data = self._client._request("GET", f"/api/v1/ai/wound-healing/profiles/{profile_id}/timeline")
        return data["data"]

    def get_reference_data(self) -> WoundReferenceData:
        """Get reference data (wound types and body locations)"""
        data = self._client._request("GET", "/api/v1/ai/wound-healing/reference-data")
        return data["data"]


# =============================================================================
# DRUG VERIFICATION MODULE
# =============================================================================

class DrugVerificationModule:
    """Drug verification module"""

    def __init__(self, client: NostraHealthAI):
        self._client = client

    def verify(
        self,
        drug_name: Optional[str] = None,
        manufacturer: Optional[str] = None,
        batch_number: Optional[str] = None,
        ndc: Optional[str] = None,
        barcode: Optional[str] = None,
        image: Optional[Union[str, Path, BinaryIO]] = None,
    ) -> DrugVerificationResult:
        """
        Verify a drug's authenticity

        Returns:
            Drug verification result
        """
        files = None
        data = {}

        if drug_name:
            data["drugName"] = drug_name
        if manufacturer:
            data["manufacturer"] = manufacturer
        if batch_number:
            data["batchNumber"] = batch_number
        if ndc:
            data["ndc"] = ndc
        if barcode:
            data["barcode"] = barcode

        if image:
            files = self._client._prepare_file(image, "image")

        result = self._client._request("POST", "/api/v1/ai/drug-verification/verify", files=files, data=data)
        return result["data"]

    def batch_verify(self, drugs: List[Dict]) -> List[DrugVerificationResult]:
        """Verify multiple drugs in batch"""
        result = self._client._request("POST", "/api/v1/ai/drug-verification/verify-batch", json={"drugs": drugs})
        return result["data"]

    def get_verifications(self) -> List[DrugVerificationResult]:
        """Get all drug verifications for the current user"""
        data = self._client._request("GET", "/api/v1/ai/drug-verification/verifications")
        return data["data"]

    def get_verification(self, verification_id: str) -> DrugVerificationResult:
        """Get a specific drug verification by ID"""
        data = self._client._request("GET", f"/api/v1/ai/drug-verification/verifications/{verification_id}")
        return data["data"]

    def get_stats(self) -> DrugVerificationStats:
        """Get drug verification statistics"""
        data = self._client._request("GET", "/api/v1/ai/drug-verification/verifications/stats")
        return data["data"]

    def delete_verification(self, verification_id: str) -> None:
        """Delete a drug verification record"""
        self._client._request("DELETE", f"/api/v1/ai/drug-verification/verifications/{verification_id}")


# =============================================================================
# FHIR MODULE
# =============================================================================

class FHIRModule:
    """FHIR (HL7 R4) module for healthcare interoperability"""

    def __init__(self, client: NostraHealthAI):
        self._client = client

    def get_patient_summary(self) -> FHIRPatientSummary:
        """Get comprehensive patient summary (all FHIR resources)"""
        data = self._client._request("GET", "/api/v1/fhir/patient/summary")
        return data["data"]

    def get_observations(self, category: Optional[str] = None) -> List[FHIRRecord]:
        """Get observations (vital signs, lab results)"""
        endpoint = "/api/v1/fhir/observations"
        if category:
            endpoint += f"?category={category}"

        data = self._client._request("GET", endpoint)
        return data["data"]

    def get_conditions(self) -> List[FHIRRecord]:
        """Get conditions (diagnoses)"""
        data = self._client._request("GET", "/api/v1/fhir/conditions")
        return data["data"]

    def get_medications(self) -> List[FHIRRecord]:
        """Get medication statements"""
        data = self._client._request("GET", "/api/v1/fhir/medications")
        return data["data"]

    def get_allergies(self) -> List[FHIRRecord]:
        """Get allergy intolerances"""
        data = self._client._request("GET", "/api/v1/fhir/allergies")
        return data["data"]

    def get_documents(self, doc_type: Optional[str] = None) -> List[FHIRRecord]:
        """Get document references (prescriptions, medical records)"""
        endpoint = "/api/v1/fhir/documents"
        if doc_type:
            endpoint += f"?type={doc_type}"

        data = self._client._request("GET", endpoint)
        return data["data"]

    def search(self, params: FHIRSearchParams) -> List[FHIRRecord]:
        """Search FHIR records with advanced filters"""
        data = self._client._request("POST", "/api/v1/fhir/search", json=params)
        return data["data"]

    def get_by_resource_type(self, resource_type: str, limit: Optional[int] = None) -> List[FHIRRecord]:
        """Get FHIR records by resource type"""
        endpoint = f"/api/v1/fhir/{resource_type}"
        if limit:
            endpoint += f"?limit={limit}"

        data = self._client._request("GET", endpoint)
        return data["data"]

    def get_all_records(self, limit: Optional[int] = None, last_doc_id: Optional[str] = None) -> Dict:
        """Get all FHIR records for the current user"""
        params = {}
        if limit:
            params["limit"] = limit
        if last_doc_id:
            params["lastDocId"] = last_doc_id

        endpoint = "/api/v1/fhir"
        if params:
            endpoint += "?" + "&".join(f"{k}={v}" for k, v in params.items())

        return self._client._request("GET", endpoint)

    def get_record(self, record_id: str) -> FHIRRecord:
        """Get a specific FHIR record by ID"""
        data = self._client._request("GET", f"/api/v1/fhir/{record_id}")
        return data["data"]

    def update_record(self, record_id: str, resource: Dict) -> FHIRRecord:
        """Update a FHIR record"""
        data = self._client._request("PUT", f"/api/v1/fhir/{record_id}", json={"resource": resource})
        return data["data"]

    def delete_record(self, record_id: str) -> None:
        """Delete a FHIR record (soft delete)"""
        self._client._request("DELETE", f"/api/v1/fhir/{record_id}")

    def archive_record(self, record_id: str) -> None:
        """Archive a FHIR record"""
        self._client._request("POST", f"/api/v1/fhir/{record_id}/archive")

    def export_bundle(self, resource_types: Optional[List[str]] = None) -> FHIRBundle:
        """Export patient data as FHIR Bundle"""
        endpoint = "/api/v1/fhir/export"
        if resource_types:
            endpoint += f"?types={','.join(resource_types)}"

        data = self._client._request("GET", endpoint)
        return data["data"]


# =============================================================================
# SUBSCRIPTION MODULE
# =============================================================================

class SubscriptionModule:
    """AI subscription management module"""

    def __init__(self, client: NostraHealthAI):
        self._client = client

    def get_plans(self) -> Dict[str, Any]:
        """Get all available subscription plans"""
        return self._client._request("GET", "/api/v1/ai/subscriptions/plans")

    def get_plan_details(self, tier: str) -> AISubscriptionPlan:
        """Get details for a specific subscription plan"""
        data = self._client._request("GET", f"/api/v1/ai/subscriptions/plans/{tier}")
        return data["plan"]

    def get_my_subscription(self) -> UserAISubscription:
        """Get current user's subscription"""
        data = self._client._request("GET", "/api/v1/ai/subscriptions/my-subscription")
        return data["subscription"]

    def purchase(
        self,
        tier: str,
        payment_method_id: Optional[str] = None,
        billing_cycle: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Purchase a subscription

        Args:
            tier: Subscription tier (free, basic, standard, premium, premium_plus)
            payment_method_id: Payment method ID (required for paid tiers)
            billing_cycle: 'monthly' or 'yearly'

        Returns:
            Dict with subscription, transaction, and message
        """
        payload: Dict[str, Any] = {"tier": tier}
        if payment_method_id:
            payload["paymentMethodId"] = payment_method_id
        if billing_cycle:
            payload["billingCycle"] = billing_cycle

        return self._client._request("POST", "/api/v1/ai/subscriptions/purchase", json=payload)

    def change(
        self,
        new_tier: str,
        payment_method_id: Optional[str] = None,
        billing_cycle: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Change (upgrade or downgrade) subscription

        Args:
            new_tier: New subscription tier
            payment_method_id: Payment method ID for upgrades
            billing_cycle: 'monthly' or 'yearly'

        Returns:
            Dict with updated subscription and message
        """
        payload: Dict[str, Any] = {"newTier": new_tier}
        if payment_method_id:
            payload["paymentMethodId"] = payment_method_id
        if billing_cycle:
            payload["billingCycle"] = billing_cycle

        return self._client._request("PUT", "/api/v1/ai/subscriptions/change", json=payload)

    def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel subscription"""
        payload = {}
        if reason:
            payload["reason"] = reason

        self._client._request("DELETE", "/api/v1/ai/subscriptions/cancel", json=payload)

    def get_ai_usage(self) -> AIUsageResponse:
        """Get AI usage statistics"""
        return self._client._request("GET", "/api/v1/ai/subscriptions/ai-usage")
