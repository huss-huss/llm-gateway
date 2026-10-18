"""Structured JSON logging with the current request ID on every record."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from llm_gateway.request_context import current_request_id

_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = getattr(record, "request_id", None) or current_request_id()
        if rid:
            out["request_id"] = rid
        for key, value in vars(record).items():
            if key not in _RESERVED and key not in out:
                out[key] = value
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JSONFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "httpx2", "httpcore2", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
