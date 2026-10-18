"""``llm-gateway`` command line: serve, migrate, manage keys, run evals."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from llm_gateway.settings import Settings


def migrations_dir() -> Path:
    candidates = [
        os.environ.get("GATEWAY_MIGRATIONS_DIR"),
        Path(__file__).resolve().parents[2] / "migrations",  # source checkout
        Path.cwd() / "migrations",
    ]
    for c in candidates:
        if c and (Path(c) / "env.py").exists():
            return Path(c)
    raise SystemExit("could not find the migrations directory; set GATEWAY_MIGRATIONS_DIR")


def db_upgrade(url: str | None = None, revision: str = "head") -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(migrations_dir()))
    cfg.set_main_option("sqlalchemy.url", url or Settings().database_url)
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, revision)


async def _with_state(fn):
    from llm_gateway.state import build_state

    state = await build_state(Settings())
    try:
        return await fn(state)
    finally:
        await state.aclose()


def cmd_keys(args: argparse.Namespace) -> int:
    async def run(state):
        if args.keys_cmd == "create":
            raw, row = await state.keys.create(args.name, args.tenant, args.rpm, args.budget)
            print(
                json.dumps(
                    {
                        "id": row.id,
                        "name": row.name,
                        "tenant": row.tenant,
                        "prefix": row.prefix,
                        "key": raw,
                    },
                    indent=2,
                )
            )
            print("Store this key now; it cannot be shown again.", file=sys.stderr)
        elif args.keys_cmd == "list":
            for k in await state.keys.list():
                status = "revoked" if k.revoked_at else "active"
                print(f"{k.id}\t{k.prefix}...\t{k.tenant}\t{k.name}\t{status}")
        elif args.keys_cmd == "revoke":
            row = await state.keys.revoke(args.id)
            if row is None:
                print(f"key {args.id} not found", file=sys.stderr)
                return 1
            print(f"revoked key {row.id} ({row.prefix}...)")
        return 0

    return asyncio.run(_with_state(run))


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "llm_gateway.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_config=None,
        access_log=False,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-gateway")
    sub = p.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the HTTP server")
    serve.add_argument("--host", default=os.environ.get("GATEWAY_HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("GATEWAY_PORT", "8080")))
    serve.set_defaults(func=cmd_serve)

    db = sub.add_parser("db", help="database migrations")
    db_sub = db.add_subparsers(dest="db_cmd", required=True)
    up = db_sub.add_parser("upgrade", help="apply migrations (alembic upgrade)")
    up.add_argument("revision", nargs="?", default="head")
    up.add_argument("--url", default=None)
    up.set_defaults(func=lambda a: db_upgrade(a.url, a.revision) or 0)

    keys = sub.add_parser("keys", help="manage API keys")
    keys_sub = keys.add_subparsers(dest="keys_cmd", required=True)
    create = keys_sub.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--tenant", required=True)
    create.add_argument("--rpm", type=int, default=None, help="requests per minute limit")
    create.add_argument("--budget", type=float, default=None, help="monthly budget in USD")
    keys_sub.add_parser("list")
    revoke = keys_sub.add_parser("revoke")
    revoke.add_argument("id", type=int)
    keys.set_defaults(func=cmd_keys)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
