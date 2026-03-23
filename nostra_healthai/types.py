"""Type definitions for NostraHealthAI SDK"""

from typing import TypedDict, List, Dict, Optional, Literal, Any
from datetime import datetime


# =============================================================================
# COMMON TYPES
# =============================================================================

class ModelInfo(TypedDict, total=False):
    """Model information returned by the API"""
    name: str
    version: str
    provider: Optional[str]
    modelsUsed: Optional[List[str]]
    aggregatedConfidence: Optional[float]


class RateLimitInfo(TypedDict):
    """Rate limit information"""
    remainingRequests: int
    resetTime: str
    retryAfter: Optional[int]


class JobMetadata(TypedDict, total=False):
    """Job metadata"""
    createdAt: datetime
    startedAt: Optional[datetime]
    completedAt: Optional[datetime]


class JobStatus(TypedDict, total=False):
    """Job status for async operations"""
    jobId: str
    status: Literal['pending', 'processing', 'completed', 'failed']
    progress: int
    metadata: JobMetadata
    data: Optional[Dict]
    error: Optional[str]


class PotentialCondition(TypedDict, total=False):
    """Potential medical condition"""
    name: str
    probability: Literal['high', 'medium', 'low']
    supportingEvidence: List[str]
    icd10Code: Optional[str]
    requiresImmediate: Optional[bool]


class Recommendation(TypedDict, total=False):
    """Medical recommendation"""
    type: str
    description: str
    urgency: Literal['immediate', 'high', 'medium', 'low']
    reasoning: Optional[str]


# =============================================================================
# CHAT TYPES
# =============================================================================

class ChatMessage(TypedDict):
    """Medical chat message"""
    id: str
    conversationId: str
    userId: str
    role: Literal['user', 'assistant']
    content: str
    timestamp: datetime


class ChatRequest(TypedDict, total=False):
    """Chat request parameters"""
    message: str
    conversationId: Optional[str]


class ChatResponse(TypedDict):
    """Chat response from the API"""
    conversationId: str
    message: ChatMessage
    response: str
    modelInfo: Optional[ModelInfo]


class Conversation(TypedDict):
    """Conversation metadata"""
    id: str
    userId: str
    title: Optional[str]
    createdAt: datetime
    updatedAt: datetime
    messageCount: int


# =============================================================================
# MEDICAL ANALYSIS TYPES
# =============================================================================

class LabInterpretation(TypedDict):
    """Laboratory interpretation"""
    parameter: str
    value: str
    unit: str
    referenceRange: str
    status: Literal['normal', 'borderline', 'abnormal', 'critical']
    interpretation: str
    clinicalSignificance: str


class DetailedSummary(TypedDict, total=False):
    """Detailed summary of analysis"""
    clinical: str
    findings: List[str]
    abnormalities: List[str]
    criticalValues: List[str]
    trends: Optional[List[str]]


class ModelMetadata(TypedDict):
    """Model metadata"""
    modelsUsed: List[str]
    modelsFailed: List[str]
    aggregatedConfidence: float
    individualConfidences: Dict[str, float]


class Analysis(TypedDict):
    """Medical analysis result"""
    id: str
    subjectId: str
    userId: str
    summary: str
    detailedSummary: DetailedSummary
    potentialConditions: List[PotentialCondition]
    laboratoryInterpretation: List[LabInterpretation]
    recommendations: List[Recommendation]
    riskLevel: Literal['low', 'medium', 'high']
    riskFactors: List[str]
    confidence: float
    disclaimer: str
    dataSources: List[Dict]
    modelMetadata: ModelMetadata
    createdAt: datetime


# =============================================================================
# SKIN INFECTION TYPES
# =============================================================================

class SkinInfectionRequest(TypedDict, total=False):
    """Skin infection analysis request"""
    file: Any  # Path, file object, or bytes
    affectedArea: Optional[str]
    duration: Optional[str]
    symptoms: Optional[List[str]]
    previousTreatments: Optional[List[str]]
    allergies: Optional[List[str]]
    medicalHistory: Optional[List[str]]


class SkinCondition(TypedDict, total=False):
    """Identified skin condition"""
    name: str
    confidence: float
    icd10Code: Optional[str]
    category: Literal['fungal', 'bacterial', 'viral', 'parasitic', 'inflammatory']
    severity: Optional[Literal['mild', 'moderate', 'severe']]
    description: Optional[str]


class SkinInfectionAnalysis(TypedDict):
    """Skin infection analysis result"""
    id: str
    userId: str
    imageUrl: str
    identifiedConditions: List[SkinCondition]
    primaryDiagnosis: Optional[SkinCondition]
    differentialDiagnoses: List[SkinCondition]
    recommendations: List[Recommendation]
    treatmentSuggestions: List[str]
    urgencyLevel: Literal['low', 'medium', 'high', 'emergency']
    followUpRequired: bool
    disclaimer: str
    modelInfo: Optional[ModelInfo]
    createdAt: datetime


class SupportedSkinConditionCategory(TypedDict):
    """Category of skin conditions"""
    name: str
    conditions: List[Dict[str, str]]  # {id, name, icd10}


class SupportedSkinConditions(TypedDict):
    """All supported skin conditions"""
    fungal: SupportedSkinConditionCategory
    bacterial: SupportedSkinConditionCategory
    viral: SupportedSkinConditionCategory
    parasitic: SupportedSkinConditionCategory
    inflammatory: SupportedSkinConditionCategory


# =============================================================================
# EYE DIAGNOSIS TYPES
# =============================================================================

class EyeDiagnosisRequest(TypedDict, total=False):
    """Eye diagnosis analysis request"""
    file: Any  # Path, file object, or bytes
    eyeSide: Optional[Literal['left', 'right', 'both']]
    symptoms: Optional[List[str]]
    symptomDuration: Optional[str]
    painLevel: Optional[int]
    visionChanges: Optional[List[str]]
    medicalHistory: Optional[List[str]]
    currentMedications: Optional[List[str]]
    allergies: Optional[List[str]]
    familyHistory: Optional[List[str]]
    lastEyeExam: Optional[str]
    wearingCorrectiveLenses: Optional[bool]
    lensType: Optional[str]


class EyeCondition(TypedDict, total=False):
    """Identified eye condition"""
    name: str
    confidence: float
    icd10Code: Optional[str]
    category: Literal['refractive', 'infection', 'inflammation', 'degenerative', 'vascular', 'neurological', 'structural']
    severity: Optional[Literal['mild', 'moderate', 'severe']]
    affectedArea: Optional[str]
    description: Optional[str]


class EyeDiagnosisAnalysis(TypedDict):
    """Eye diagnosis analysis result"""
    id: str
    userId: str
    imageUrl: str
    eyeSide: Literal['left', 'right', 'both']
    identifiedConditions: List[EyeCondition]
    primaryDiagnosis: Optional[EyeCondition]
    differentialDiagnoses: List[EyeCondition]
    visualAcuityEstimate: Optional[str]
    recommendations: List[Recommendation]
    treatmentSuggestions: List[str]
    urgencyLevel: Literal['low', 'medium', 'high', 'emergency']
    followUpRequired: bool
    referralRecommended: bool
    disclaimer: str
    modelInfo: Optional[ModelInfo]
    createdAt: datetime


class SupportedEyeConditionCategory(TypedDict):
    """Category of eye conditions"""
    description: str
    conditions: List[Dict[str, str]]  # {id, name, icd10}


class SupportedEyeConditions(TypedDict):
    """All supported eye conditions"""
    refractive: SupportedEyeConditionCategory
    infection: SupportedEyeConditionCategory
    inflammation: SupportedEyeConditionCategory
    degenerative: SupportedEyeConditionCategory
    vascular: SupportedEyeConditionCategory
    neurological: SupportedEyeConditionCategory
    structural: SupportedEyeConditionCategory


# =============================================================================
# WOUND HEALING TYPES
# =============================================================================

class WoundAnalysisRequest(TypedDict, total=False):
    """Wound analysis request"""
    file: Any  # Path, file object, or bytes
    woundProfileId: Optional[str]
    woundType: Optional[str]
    bodyLocation: Optional[str]
    painLevel: Optional[int]
    symptoms: Optional[List[str]]
    currentTreatments: Optional[List[str]]
    recentChanges: Optional[str]


class WoundProfileRequest(TypedDict, total=False):
    """Wound profile creation request"""
    name: str
    woundType: str
    bodyLocation: str
    bodyLocationDetails: Optional[str]
    etiology: Optional[str]
    woundOnsetDate: Optional[datetime]
    notes: Optional[str]


class WoundProfile(TypedDict):
    """Wound profile for tracking"""
    id: str
    userId: str
    name: str
    woundType: str
    bodyLocation: str
    bodyLocationDetails: Optional[str]
    etiology: Optional[str]
    woundOnsetDate: Optional[datetime]
    notes: Optional[str]
    status: Literal['active', 'archived']
    analysisCount: int
    createdAt: datetime
    updatedAt: datetime


class WoundMeasurement(TypedDict, total=False):
    """Wound measurement data"""
    length: Optional[float]
    width: Optional[float]
    depth: Optional[float]
    area: Optional[float]
    unit: Literal['cm', 'mm', 'inches']


class TissueAnalysis(TypedDict, total=False):
    """Wound tissue composition analysis"""
    granulation: Optional[float]
    epithelial: Optional[float]
    slough: Optional[float]
    necrotic: Optional[float]


class WoundHealingAnalysis(TypedDict):
    """Wound healing analysis result"""
    id: str
    userId: str
    woundProfileId: Optional[str]
    imageUrl: str
    woundType: str
    bodyLocation: str
    measurements: Optional[WoundMeasurement]
    healingStage: Literal['inflammatory', 'proliferative', 'maturation', 'chronic']
    healingProgress: Literal['improving', 'stable', 'declining', 'unknown']
    tissueAnalysis: TissueAnalysis
    infectionSigns: List[str]
    infectionRisk: Literal['low', 'medium', 'high']
    recommendations: List[Recommendation]
    careInstructions: List[str]
    warningSignsToWatch: List[str]
    nextAnalysisRecommended: Optional[datetime]
    disclaimer: str
    modelInfo: Optional[ModelInfo]
    createdAt: datetime


class WoundTimelineEntry(TypedDict):
    """Single entry in wound timeline"""
    id: str
    date: datetime
    healingProgress: str
    measurements: Optional[WoundMeasurement]
    infectionRisk: str


class WoundTimeline(TypedDict):
    """Wound healing timeline"""
    profileId: str
    analyses: List[WoundTimelineEntry]
    overallTrend: Literal['improving', 'stable', 'declining', 'insufficient_data']


class WoundReferenceData(TypedDict):
    """Reference data for wound types and body locations"""
    woundTypes: List[Dict[str, str]]  # {id, name}
    bodyLocations: Dict[str, List[str]]  # {head: [...], torso: [...], ...}


# =============================================================================
# DRUG VERIFICATION TYPES
# =============================================================================

class DrugVerificationRequest(TypedDict, total=False):
    """Drug verification request"""
    drugName: Optional[str]
    manufacturer: Optional[str]
    batchNumber: Optional[str]
    ndc: Optional[str]
    barcode: Optional[str]
    image: Optional[Any]  # Path, file object, or bytes


class DrugInfo(TypedDict, total=False):
    """Verified drug information"""
    name: str
    genericName: Optional[str]
    manufacturer: str
    ndc: Optional[str]
    activeIngredients: Optional[List[str]]
    dosageForm: Optional[str]
    strength: Optional[str]


class BatchInfo(TypedDict, total=False):
    """Drug batch information"""
    batchNumber: str
    expirationDate: Optional[datetime]
    manufacturingDate: Optional[datetime]
    isRecalled: bool
    recallInfo: Optional[str]


class VerificationDetails(TypedDict):
    """Verification check details"""
    ndcVerified: bool
    manufacturerVerified: bool
    packagingVerified: bool
    recallChecked: bool


class DrugVerificationResult(TypedDict):
    """Drug verification result"""
    id: str
    userId: str
    verificationStatus: Literal['verified', 'unverified', 'suspicious', 'counterfeit']
    drugInfo: Optional[DrugInfo]
    batchInfo: Optional[BatchInfo]
    verificationDetails: VerificationDetails
    warnings: List[str]
    recommendations: List[str]
    confidence: float
    disclaimer: str
    createdAt: datetime


class DrugVerificationStats(TypedDict):
    """Drug verification statistics"""
    totalVerifications: int
    verified: int
    unverified: int
    suspicious: int
    counterfeit: int
    lastVerificationDate: Optional[datetime]


# =============================================================================
# FHIR TYPES
# =============================================================================

FHIRResourceType = Literal[
    'Patient',
    'Practitioner',
    'Observation',
    'Condition',
    'MedicationStatement',
    'DiagnosticReport',
    'DocumentReference',
    'AllergyIntolerance',
    'Encounter'
]


class FHIRMeta(TypedDict, total=False):
    """FHIR resource metadata"""
    versionId: Optional[str]
    lastUpdated: Optional[str]
    source: Optional[str]
    profile: Optional[List[str]]


class FHIRResource(TypedDict, total=False):
    """FHIR resource base"""
    resourceType: str
    id: Optional[str]
    meta: Optional[FHIRMeta]


class FHIRRecord(TypedDict):
    """FHIR record wrapper"""
    id: str
    userId: str
    resourceType: str
    resource: FHIRResource
    createdAt: datetime
    updatedAt: datetime
    visibility: Optional[Literal['private', 'shared', 'public']]
    status: Optional[Literal['active', 'archived', 'deleted']]


class FHIRSearchParams(TypedDict, total=False):
    """FHIR search parameters"""
    resourceType: Optional[str]
    category: Optional[str]
    status: Optional[Literal['active', 'archived']]
    startDate: Optional[datetime]
    endDate: Optional[datetime]
    limit: Optional[int]


class FHIRPatientSummary(TypedDict):
    """Comprehensive patient summary"""
    patient: FHIRResource
    observations: List[FHIRResource]
    conditions: List[FHIRResource]
    medications: List[FHIRResource]
    allergies: List[FHIRResource]
    documents: List[FHIRResource]


class FHIRBundleEntry(TypedDict, total=False):
    """FHIR Bundle entry"""
    resource: FHIRResource
    fullUrl: Optional[str]


class FHIRBundle(TypedDict):
    """FHIR Bundle resource"""
    resourceType: Literal['Bundle']
    type: Literal['searchset', 'collection']
    total: int
    entry: List[FHIRBundleEntry]


# =============================================================================
# SUBSCRIPTION TYPES
# =============================================================================

AISubscriptionTier = Literal['free', 'basic', 'standard', 'premium', 'premium_plus']


class AISubscriptionPlan(TypedDict, total=False):
    """AI subscription plan details"""
    tier: str
    name: str
    description: str
    price: float
    yearlyPrice: Optional[float]
    currency: str
    features: List[str]
    limits: Dict[str, Any]
    categoryLimits: Optional[Dict[str, Any]]
    isActive: bool


class UserAISubscription(TypedDict, total=False):
    """User's current AI subscription"""
    tier: str
    status: str
    startDate: str
    endDate: Optional[str]
    billingCycle: Optional[str]
    autoRenew: bool
    limits: Dict[str, Any]
    usage: Dict[str, Any]
    cancelledAt: Optional[str]
    cancellationReason: Optional[str]
    previousTier: Optional[str]
    organizationId: Optional[str]


class SubscriptionPurchaseRequest(TypedDict, total=False):
    """Subscription purchase request"""
    tier: str
    paymentMethodId: Optional[str]
    billingCycle: Optional[str]


class SubscriptionChangeRequest(TypedDict, total=False):
    """Subscription change request"""
    newTier: str
    paymentMethodId: Optional[str]
    billingCycle: Optional[str]


class AIUsageResponse(TypedDict, total=False):
    """AI usage statistics response"""
    subscription: Dict[str, Any]
    limits: Dict[str, Any]
    usage: Dict[str, Any]
    sixHourUsage: Optional[Dict[str, Any]]
    usagePercentages: Dict[str, Any]
    remainingRequests: Dict[str, Any]
    resetTimes: Dict[str, Any]
    aiToolUsage: Optional[Dict[str, Any]]
    categoryUsage: Optional[Dict[str, Any]]
