from __future__ import annotations

import logging
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Response, status
from pydantic import BaseModel, Field

from api.core.config import (
    AUTH_COOKIE_NAME,
    AUTH_COOKIE_SECURE,
    AUTH_SECRET_KEY,
    AUTH_SESSION_TTL_HOURS,
    CSRF_COOKIE_NAME,
)
from api.database.models import AuthSession, User
from api.database.mongodb import (
    create_auth_session,
    create_user,
    get_auth_session,
    get_user_by_email,
    get_user_by_id,
    is_db_configured,
    revoke_auth_session,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_JWT_ALG = "HS256"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SECRET = AUTH_SECRET_KEY or secrets.token_urlsafe(32)

if not AUTH_SECRET_KEY:
    logger.warning("AUTH_SECRET_KEY is not set; using an ephemeral development secret.")


class AuthPayload(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)
    name: str | None = Field(default=None, max_length=100)


class LoginPayload(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_email(email: str) -> str:
    value = email.strip().lower()
    if not _EMAIL_RE.match(value):
        raise HTTPException(status_code=400, detail="Email không hợp lệ")
    return value


def _validate_password(password: str) -> None:
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Mật khẩu phải có ít nhất 8 ký tự")
    if password.strip() != password:
        raise HTTPException(status_code=400, detail="Mật khẩu không được bắt đầu hoặc kết thúc bằng khoảng trắng")


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def _public_user(user: User) -> dict:
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "role": user.role,
    }


def _ensure_db() -> None:
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")


def _create_token(user_id: str, session_id: str, expires_at: datetime) -> str:
    return jwt.encode(
        {"sub": user_id, "sid": session_id, "exp": expires_at},
        _SECRET,
        algorithm=_JWT_ALG,
    )


def _set_session_cookie(response: Response, token: str, expires_at: datetime) -> None:
    max_age = max(0, int((expires_at - _now()).total_seconds()))
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        max_age=max_age,
        expires=expires_at,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(AUTH_COOKIE_NAME, path="/", samesite="lax")


def _set_csrf_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        CSRF_COOKIE_NAME,
        token,
        max_age=AUTH_SESSION_TTL_HOURS * 3600,
        httponly=False,
        secure=AUTH_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


async def _issue_session(response: Response, user: User) -> dict:
    if user.id is None:
        raise HTTPException(status_code=500, detail="User id missing")

    session_id = uuid.uuid4().hex
    user_id = str(user.id)
    expires_at = _now() + timedelta(hours=AUTH_SESSION_TTL_HOURS)
    await create_auth_session(AuthSession(
        session_id=session_id,
        user_id=user_id,
        expires_at=expires_at,
    ))
    token = _create_token(user_id, session_id, expires_at)
    _set_session_cookie(response, token, expires_at)
    csrf_token = secrets.token_urlsafe(32)
    _set_csrf_cookie(response, csrf_token)
    return {"user": _public_user(user), "csrf_token": csrf_token}


async def get_current_user(
    session_cookie: str | None = Cookie(default=None, alias=AUTH_COOKIE_NAME),
) -> User:
    _ensure_db()
    if not session_cookie:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    try:
        payload = jwt.decode(session_cookie, _SECRET, algorithms=[_JWT_ALG])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session") from exc

    user_id = str(payload.get("sub") or "")
    session_id = str(payload.get("sid") or "")
    if not user_id or not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

    session = await get_auth_session(session_id)
    if (
        session is None
        or session.revoked
        or session.user_id != user_id
        or _as_aware_utc(session.expires_at) <= _now()
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    user = await get_user_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User disabled")
    return user


async def verify_csrf(
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
    csrf_cookie: str | None = Cookie(default=None, alias=CSRF_COOKIE_NAME),
) -> None:
    if not csrf_header or not csrf_cookie or not secrets.compare_digest(csrf_header, csrf_cookie):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")


async def get_current_user_with_csrf(
    current_user: User = Depends(get_current_user),
    _csrf: None = Depends(verify_csrf),
) -> User:
    return current_user


@router.post("/register")
async def register(payload: AuthPayload, response: Response) -> dict:
    _ensure_db()
    email = _normalize_email(payload.email)
    _validate_password(payload.password)
    if await get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email đã được đăng ký")

    name = (payload.name or email.split("@", 1)[0]).strip()
    if not name:
        raise HTTPException(status_code=400, detail="Tên không hợp lệ")

    user = User(email=email, name=name, password_hash=_hash_password(payload.password))
    user_id = await create_user(user)
    created = await get_user_by_id(user_id)
    if created is None:
        raise HTTPException(status_code=500, detail="Không tạo được tài khoản")
    return await _issue_session(response, created)


@router.post("/login")
async def login(payload: LoginPayload, response: Response) -> dict:
    _ensure_db()
    email = _normalize_email(payload.email)
    user = await get_user_by_email(email)
    if user is None or not user.is_active or not _verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng")
    return await _issue_session(response, user)


@router.post("/logout")
async def logout(
    response: Response,
    current_user: User = Depends(get_current_user_with_csrf),
    session_cookie: str | None = Cookie(default=None, alias=AUTH_COOKIE_NAME),
) -> dict:
    del current_user
    if session_cookie:
        try:
            payload = jwt.decode(session_cookie, _SECRET, algorithms=[_JWT_ALG])
            session_id = str(payload.get("sid") or "")
            if session_id:
                await revoke_auth_session(session_id)
        except jwt.PyJWTError:
            pass
    _clear_session_cookie(response)
    response.delete_cookie(CSRF_COOKIE_NAME, path="/", samesite="lax")
    return {"ok": True}


@router.get("/me")
async def me(current_user: User = Depends(get_current_user)) -> dict:
    return {"user": _public_user(current_user)}


@router.get("/csrf")
async def csrf(response: Response, current_user: User = Depends(get_current_user)) -> dict:
    del current_user
    token = secrets.token_urlsafe(32)
    _set_csrf_cookie(response, token)
    return {"csrf_token": token}
