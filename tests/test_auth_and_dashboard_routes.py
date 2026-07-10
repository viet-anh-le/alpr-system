from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from bson import ObjectId
from fastapi import FastAPI

from api.database.models import AuthSession, PlateFrame, RecognitionRecord, RecognitionSession, User

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def auth_client(monkeypatch):
    from api import auth

    users_by_email: dict[str, User] = {}
    users_by_id: dict[str, User] = {}
    sessions: dict[str, AuthSession] = {}

    async def create_user(user: User) -> str:
        user.id = ObjectId()
        users_by_email[user.email] = user
        users_by_id[str(user.id)] = user
        return str(user.id)

    async def get_user_by_email(email: str) -> User | None:
        return users_by_email.get(email)

    async def get_user_by_id(user_id: str) -> User | None:
        return users_by_id.get(user_id)

    async def create_auth_session(session: AuthSession) -> str:
        session.id = ObjectId()
        sessions[session.session_id] = session
        return str(session.id)

    async def get_auth_session(session_id: str) -> AuthSession | None:
        return sessions.get(session_id)

    async def revoke_auth_session(session_id: str) -> None:
        if session_id in sessions:
            sessions[session_id].revoked = True

    monkeypatch.setattr(auth, "is_db_configured", lambda: True)
    monkeypatch.setattr(auth, "create_user", create_user)
    monkeypatch.setattr(auth, "get_user_by_email", get_user_by_email)
    monkeypatch.setattr(auth, "get_user_by_id", get_user_by_id)
    monkeypatch.setattr(auth, "create_auth_session", create_auth_session)
    monkeypatch.setattr(auth, "get_auth_session", get_auth_session)
    monkeypatch.setattr(auth, "revoke_auth_session", revoke_auth_session)

    app = FastAPI()
    app.include_router(auth.router)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    try:
        yield client
    finally:
        await client.aclose()


async def test_register_login_me_and_logout(auth_client):
    resp = await auth_client.post("/auth/register", json={
        "name": "Le Anh",
        "email": "anh@example.com",
        "password": "strongpass123",
    })
    assert resp.status_code == 200
    assert resp.json()["user"]["email"] == "anh@example.com"
    assert "password" not in resp.text

    csrf = resp.json()["csrf_token"]
    me = await auth_client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["name"] == "Le Anh"

    logout = await auth_client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
    assert logout.status_code == 200
    assert logout.json()["ok"] is True

    after_logout = await auth_client.get("/auth/me")
    assert after_logout.status_code == 401

    login = await auth_client.post("/auth/login", json={
        "email": "anh@example.com",
        "password": "strongpass123",
    })
    assert login.status_code == 200


async def test_register_rejects_duplicate_email(auth_client):
    payload = {"name": "User", "email": "dup@example.com", "password": "strongpass123"}
    assert (await auth_client.post("/auth/register", json=payload)).status_code == 200
    duplicate = await auth_client.post("/auth/register", json=payload)
    assert duplicate.status_code == 409


async def test_login_rejects_wrong_password(auth_client):
    await auth_client.post("/auth/register", json={
        "name": "User",
        "email": "wrong@example.com",
        "password": "strongpass123",
    })
    resp = await auth_client.post("/auth/login", json={
        "email": "wrong@example.com",
        "password": "bad-password",
    })
    assert resp.status_code == 401


async def test_protected_alpr_routes_require_auth(monkeypatch):
    from api import auth, main

    monkeypatch.setattr(auth, "is_db_configured", lambda: True)
    main.app.dependency_overrides.clear()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        assert (await client.get("/sessions")).status_code == 401
        assert (await client.get("/records/nope/1")).status_code == 401
        assert (await client.get("/jobs/nope/preprocessed-video")).status_code == 401


async def test_sessions_and_records_are_filtered_by_user(monkeypatch):
    from api import main
    from api.auth import get_current_user

    user = User(
        id=ObjectId(),
        email="owner@example.com",
        name="Owner",
        password_hash="hash",
    )
    owned_session = RecognitionSession(
        session_id="job_owned",
        user_id=str(user.id),
        source_filename="video.mp4",
        status="completed",
        total_records=1,
    )
    owned_record = RecognitionRecord(
        session_id="job_owned",
        user_id=str(user.id),
        track_id=7,
        vehicle_class="car",
        best_plate_frame=PlateFrame(frame_index=1, quality_score=0.9, image_url=None),
        track_buffer=[],
        plate_text="30A12345",
        plate_text_confidence=0.95,
        first_seen_frame=1,
        last_seen_frame=10,
    )

    async def current_user_override():
        return user

    async def list_sessions_for_user(user_id: str, limit: int = 50):
        assert user_id == str(user.id)
        return [owned_session]

    async def get_session_for_user(session_id: str, user_id: str):
        if session_id == "job_owned" and user_id == str(user.id):
            return owned_session
        return None

    async def get_records_for_session_for_user(session_id: str, user_id: str):
        assert session_id == "job_owned"
        assert user_id == str(user.id)
        return [owned_record]

    async def get_record_by_track_for_user(session_id: str, track_id: int, user_id: str):
        if session_id == "job_owned" and track_id == 7 and user_id == str(user.id):
            return owned_record
        return None

    monkeypatch.setattr("api.database.mongodb.is_db_configured", lambda: True)
    monkeypatch.setattr("api.database.mongodb.list_sessions_for_user", list_sessions_for_user)
    monkeypatch.setattr("api.database.mongodb.get_session_for_user", get_session_for_user)
    monkeypatch.setattr(
        "api.database.mongodb.get_records_for_session_for_user",
        get_records_for_session_for_user,
    )
    monkeypatch.setattr(
        "api.database.mongodb.get_record_by_track_for_user",
        get_record_by_track_for_user,
    )

    main.app.dependency_overrides[get_current_user] = current_user_override
    transport = httpx.ASGITransport(app=main.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            sessions = await client.get("/sessions")
            assert sessions.status_code == 200
            assert sessions.json()["items"][0]["session_id"] == "job_owned"

            records = await client.get("/sessions/job_owned/records")
            assert records.status_code == 200
            assert records.json()["items"][0]["plate_text"] == "30A12345"

            direct = await client.get("/records/job_owned/7")
            assert direct.status_code == 200

            missing = await client.get("/sessions/other_user_job")
            assert missing.status_code == 404
    finally:
        main.app.dependency_overrides.clear()
