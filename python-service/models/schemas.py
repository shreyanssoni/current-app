from pydantic import BaseModel
from typing import Optional, List, Any


# --- Stream/Task Schemas ---
class StreamLockRequest(BaseModel):
    userId: Optional[str] = None
    tasks: List[Any]


class StreamAddRequest(BaseModel):
    content: str
    type: str = "task"
    userId: Optional[str] = None
    in_stream: bool = False
    duration: Optional[int] = None


class StreamToggleRequest(BaseModel):
    id: str
    in_stream: bool


class StreamUpdateRequest(BaseModel):
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    content: Optional[str] = None
    status: Optional[str] = None
    json_attributes: Optional[dict] = None


class SmartRescheduleRequest(BaseModel):
    id: str
    userId: Optional[str] = None
    duration: int = 30


class TaskCompleteRequest(BaseModel):
    userId: Optional[str] = None


class TaskSuggestRequest(BaseModel):
    taskContext: Optional[str] = None
    task: Optional[str] = None
    resistance: Optional[int] = 50


class TaskScheduleRequest(BaseModel):
    date: str
    time: str


# --- Insight Schemas ---
class InsightCaptureRequest(BaseModel):
    content: str
    userId: Optional[str] = None
    type: str = "raw"
    status: str = "INBOX"


class InsightMorphRequest(BaseModel):
    action: str  # 'promote', 'save', or 'trash'


class InsightRescheduleRequest(BaseModel):
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    minutes: Optional[int] = None


class InsightDecomposeRequest(BaseModel):
    id: str
    userId: Optional[str] = None


# --- Library Schemas ---
class LibrarySparkRequest(BaseModel):
    content: str


# --- Mission Schemas ---
class MissionGenerateRequest(BaseModel):
    userId: Optional[str] = None


class MissionCascadeRequest(BaseModel):
    userId: Optional[str] = None
    minutes: int = 15


class MissionSwapRequest(BaseModel):
    locked_id: str
    dock_id: str


# --- Queue Schemas ---
class QueueMoveRequest(BaseModel):
    id: str
    targetZone: str


# --- Mindspace Schemas ---
class MindspaceChatRequest(BaseModel):
    query: str
    userId: Optional[str] = None


# --- AI Schemas ---
class GenerateRequest(BaseModel):
    prompt: str
    provider: Optional[str] = "groq"
    model: Optional[str] = None
    prompt_type: Optional[str] = None


class AnalyzeRequest(BaseModel):
    text: str


class CopilotIgniteRequest(BaseModel):
    task_text: str
    userId: str
