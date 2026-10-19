"""``llm-gateway`` command line: serve, migrate, manage keys, run and compare evals."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from llm_gateway.settings import Settings, load_dotenv_file


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


def cmd_eval(args: argparse.Namespace) -> int:
    from llm_gateway.evals.runner import EvalRunner, format_report, store_report
    from llm_gateway.evals.suite import load_suite

    suite = load_suite(args.suite)

    async def run(state):
        runner = EvalRunner(
            state.router,
            state.catalog,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
        )
        report = await runner.run(suite, args.model)
        if not args.no_store:
            await store_report(state.sessionmaker, report)
        return report

    report = asyncio.run(_with_state(run))
    if args.json:
        print(
            json.dumps(
                {**report.summary(), "results": [vars(r) for r in report.results]},
                indent=2,
                default=str,
            )
        )
    else:
        print(format_report(report))
    return 0 if report.pass_rate >= args.min_pass_rate else 1


def cmd_eval_diff(args: argparse.Namespace) -> int:
    from llm_gateway.db.session import make_engine, make_sessionmaker
    from llm_gateway.evals.diff import RunNotFoundError, diff_runs, format_diff, load_run

    async def run():
        engine = make_engine(args.url or Settings().database_url)
        try:
            sm = make_sessionmaker(engine)
            return diff_runs(
                await load_run(sm, args.run_a, args.suite),
                await load_run(sm, args.run_b, args.suite),
            )
        finally:
            await engine.dispose()

    try:
        d = asyncio.run(run())
    except RunNotFoundError as exc:
        print(f"eval diff: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        for w in d.warnings:  # also in the payload; stderr keeps them visible in CI logs
            print(f"warning: {w}", file=sys.stderr)
        print(json.dumps(d.to_dict(args.fail_on), indent=2))
    else:
        print(format_diff(d, args.fail_on))
    return 1 if d.gate_failures(args.fail_on) else 0


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

    ev = sub.add_parser("eval", help="run prompt eval suites")
    ev_sub = ev.add_subparsers(dest="eval_cmd", required=True)
    run = ev_sub.add_parser("run", help="run a suite against a model alias")
    run.add_argument("suite", help="path to a suite YAML file")
    run.add_argument("--model", required=True, help="alias or provider/model")
    run.add_argument("--judge-model", default="smart", help="model for llm_judge assertions")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--no-store", action="store_true", help="do not save results to the DB")
    run.add_argument("--json", action="store_true", help="print machine-readable results")
    run.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="exit non-zero below this pass rate (0-1), for CI gates",
    )
    run.set_defaults(func=cmd_eval)

    diff = ev_sub.add_parser(
        "diff",
        help="compare two stored runs case by case",
        description="Compare two stored eval runs. RUN is a run id, 'latest', or 'previous'. "
        "Exit codes: 0 gate passed, 1 gate failed, 2 run not found.",
    )
    diff.add_argument("run_a", metavar="RUN_A", help="baseline run")
    diff.add_argument("run_b", metavar="RUN_B", help="candidate run")
    diff.add_argument("--format", choices=["text", "json"], default="text")
    diff.add_argument(
        "--fail-on",
        choices=["regression", "pass-rate", "never"],
        default="regression",
        help="regression: any regressed case or a lower pass rate (default); "
        "pass-rate: only a lower pass rate; never: always exit 0",
    )
    diff.add_argument(
        "--suite", default=None, help="resolve latest/previous among runs of this suite only"
    )
    diff.add_argument("--url", default=None, help="database URL (default: GATEWAY_DATABASE_URL)")
    diff.set_defaults(func=cmd_eval_diff)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv_file()
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
