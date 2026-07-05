"""
api/database/models.py — MongoDB document models for the ALPR system.

Two top-level collections:
  recognition_sessions  — one per processed video / camera job
  recognition_records   — one per unique tracked vehicle (with plate OCR result)

Images (plate crops, vehicle thumbnails) are stored in external object storage
(Supabase Storage / S3). MongoDB holds metadata + public URLs only, keeping
documents well within the 16 MB Atlas limit even for long tracks.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from bson import ObjectId
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema


# ── ObjectId compatibility ────────────────────────────────────────────────────

class _ObjectIdPydanticAnnotation:
    """Make bson.ObjectId work with Pydantic v2."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.to_string_ser_schema(),
        )

    @staticmethod
    def _validate(value: Any) -> ObjectId:
        if isinstance(value, ObjectId):
            return value
        if ObjectId.is_valid(value):
            return ObjectId(value)
        raise ValueError(f"Invalid ObjectId: {value!r}")


PyObjectId = Annotated[ObjectId, _ObjectIdPydanticAnnotation]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Embedded documents ────────────────────────────────────────────────────────

class PlateFrame(BaseModel):
    """
    Metadata for a single plate-crop frame captured during vehicle tracking.

    Stored as an embedded sub-document inside RecognitionRecord — either as
    best_plate_frame (the highest-quality frame) or inside track_buffer
    (every buffered frame for this track).

    Images are stored as base64 JPEG inline (image_b64) and, when uploaded
    to external storage, also as a public URL (image_url). At least one of
    image_b64 or image_url must be set.
    """

    frame_index: int
    quality_score: float                 # Laplacian-variance focus score [0, 1]

    # Public Supabase Storage URL for this plate crop
    image_url: str | None = None

    # Enrichment fields — populated when available from the pipeline
    timestamp_ms: float | None = None
    plate_bbox: list[float] | None = None        # [x1, y1, x2, y2] normalised
    detection_confidence: float | None = None

    # Per-frame OCR output — set in single-frame prob_vote path
    ocr_text: str | None = None
    ocr_confidence: float | None = None


class RecognitionCluster(BaseModel):
    """One OCR cluster within a mixed track buffer."""

    cluster_index: int
    plate_text: str
    chars: list[tuple[str, float]] = Field(default_factory=list)
    best_plate_frame: PlateFrame
    track_buffer: list[PlateFrame] = Field(default_factory=list)
    plate_text_confidence: float = 0.0
    ocr_vote_summary: dict[str, int] = Field(default_factory=dict)
    ocr_method: Literal[
        "prob_vote",
        "segment_vote",
        "ocr_output_ctm",
        "single_frame_direct",
        "paddle_prob_vote",
        "paddle_segment_vote",
    ] = "ocr_output_ctm"
    frame_count: int = 0
    template: str | None = None


# ── Top-level collections ─────────────────────────────────────────────────────

class User(BaseModel):
    """Application user stored in MongoDB."""

    id: PyObjectId | None = Field(default=None, alias="_id")
    email: str
    name: str
    password_hash: str
    role: Literal["user", "admin"] = "user"
    is_active: bool = True
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}


class AuthSession(BaseModel):
    """Server-side session backing the HttpOnly auth cookie."""

    id: PyObjectId | None = Field(default=None, alias="_id")
    session_id: str
    user_id: str
    expires_at: datetime
    revoked: bool = False
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}


class RecognitionSession(BaseModel):
    """
    One video-processing job submitted via POST /upload.

    Collection: recognition_sessions
    Unique index: session_id
    """

    id: PyObjectId | None = Field(default=None, alias="_id")
    session_id: str                      # UUID hex — stable public identifier
    user_id: str | None = None           # FK → users._id for authenticated jobs
    source_filename: str
    source_type: Literal["video", "image_dir", "rtsp"] = "video"
    status: Literal["queued", "processing", "completed", "failed"] = "queued"
    total_records: int = 0               # vehicles finalised
    processed_frames: int = 0
    preprocess_mode: str = "none"
    ocr_backend: str = "default"
    error_message: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}


class RecognitionRecord(BaseModel):
    """
    One fully-tracked vehicle with its OCR-resolved licence plate.

    Collection: recognition_records
    Unique compound index: (session_id, track_id)

    Field mapping from user spec
    ────────────────────────────
    license_plate_img  → best_plate_frame   (PlateFrame with highest quality_score)
    track_buffer       → track_buffer       (all buffered PlateFrames for this track)
    value              → plate_text         (final OCR output after track voting)
    """

    id: PyObjectId | None = Field(default=None, alias="_id")

    # ── Identity ──────────────────────────────────────────────────────────────
    session_id: str                      # FK → RecognitionSession.session_id
    user_id: str | None = None           # FK → users._id for authenticated jobs
    track_id: int                        # assigned by SORT-family tracker
    vehicle_track_id: int | None = None  # raw vehicle tracker id for this result
    plate_track_id: int | None = None    # raw plate tracker id, when available
    vehicle_class: str                   # "car" | "motorcycle" | "bus" | "truck"

    # ── Key image fields ──────────────────────────────────────────────────────
    # The single clearest plate crop (highest quality_score in the buffer)
    best_plate_frame: PlateFrame

    # Every plate crop collected during this track (up to MAX_BUFFER frames)
    track_buffer: list[PlateFrame] = Field(default_factory=list)

    # Optional: public URL for the full-vehicle thumbnail (for UI display)
    vehicle_thumbnail_url: str | None = None

    # ── OCR result ────────────────────────────────────────────────────────────
    # Final licence-plate text produced by track-level OCR voting
    plate_text: str

    # Weighted-mean character probability across the winning decode
    plate_text_confidence: float

    # Raw vote tallies, e.g. {"51A12345": 8, "51A1234S": 2}
    ocr_vote_summary: dict[str, int] = Field(default_factory=dict)

    # Distinct OCR clusters split out of a mixed/reused track buffer.
    clusters: list[RecognitionCluster] = Field(default_factory=list)

    # Which OCR voting strategy produced the final result
    ocr_method: Literal[
        "prob_vote",
        "segment_vote",
        "ocr_output_ctm",
        "single_frame_direct",
        "paddle_prob_vote",
        "paddle_segment_vote",
    ] = "prob_vote"

    # ── Temporal metadata ─────────────────────────────────────────────────────
    first_seen_frame: int
    last_seen_frame: int

    # ── Audit timestamps (required by CLAUDE.md) ──────────────────────────────
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}


# ── Event-Monitor models ──────────────────────────────────────────────────


class EventVehicle(BaseModel):
    """One vehicle detected within an event's analysis window."""

    track_id: int
    vehicle_track_id: int | None = None
    plate_track_id: int | None = None
    plate_text: str
    plate_text_confidence: float
    chars: list[tuple[str, float]]
    vehicle_class: str
    plate_image_url: str | None = None
    vehicle_image_url: str | None = None
    ocr_method: Literal["segment_vote", "prob_vote", "ocr_output_ctm", "single_frame_direct"]
    ocr_frames: int
    first_seen_frame: int
    last_seen_frame: int


class MonitorEvent(BaseModel):
    """A user-marked event: a short window pulled out of a live stream
    or uploaded video for fast ALPR analysis."""

    event_id: str
    session_id: str
    source_type: Literal["live", "upload"]
    source_ref: str
    marked_at: datetime
    window_start_sec: float
    window_end_sec: float
    duration_sec: float
    status: Literal["processing", "completed", "failed"]
    vehicles: list[EventVehicle] = Field(default_factory=list)
    total_vehicles: int = 0
    processing_ms: int | None = None
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
