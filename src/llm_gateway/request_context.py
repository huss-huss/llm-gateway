"""Request ID propagation (contextvar + ASGI middleware)."""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_VALID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def current_request_id() -> str | None:
    return request_id_var.get()


class RequestIDMiddleware:
    """Accepts a well-formed incoming ``X-Request-ID`` or generates one; echoes it back."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = None
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                incoming = value.decode("latin-1")
                break
        rid = incoming if incoming and _VALID.match(incoming) else f"req_{uuid.uuid4().hex}"
        token = request_id_var.set(rid)
        scope.setdefault("state", {})["request_id"] = rid

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)
