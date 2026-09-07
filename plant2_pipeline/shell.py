"""Running subprocesses so that failures are impossible to miss.

The one rule: a non-zero exit stops the pipeline. There is no ``try/except``
here and none anywhere else in this package. Every stage below either produces
its output or raises with the command that failed printed above it.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str | Path], *, env: dict[str, str] | None = None,
        cwd: Path | None = None, log: Path | None = None) -> None:
    """Run a command, streaming its output. Raise on a non-zero exit.

    `env` is *added* to the current environment rather than replacing it, and
    is echoed so a reader of the log can see exactly which result-changing
    variables were in force.
    """
    argv = [str(c) for c in cmd]
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    banner = ["$ " + " ".join(shlex.quote(a) for a in argv)]
    if env:
        banner.insert(0, "  env: " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))
    if cwd:
        banner.insert(0, f"  cwd: {cwd}")
    print("\n".join(banner), flush=True)

    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(banner) + "\n")
            handle.flush()
            code = subprocess.call(argv, env=full_env, cwd=cwd,
                                   stdout=handle, stderr=subprocess.STDOUT)
        print(f"  -> exit {code}   log: {log}", flush=True)
    else:
        code = subprocess.call(argv, env=full_env, cwd=cwd)

    if code != 0:
        raise SystemExit(f"Command failed with exit {code}: {' '.join(argv)}")


def run_background(cmd: list[str | Path], *, env: dict[str, str] | None = None,
                   cwd: Path | None = None, log: Path) -> subprocess.Popen:
    """Start a command detached and return the Popen handle.

    Returns the handle, not the pid, so `wait_for` can reap the child. Waiting
    on a bare pid does not work here: an exited child that has not been reaped
    stays a zombie, `/proc/<pid>` still exists for it, and a wait loop built on
    that check never finishes — which is exactly how the first full run hung
    after the dump's 23 shards had all completed.
    """
    argv = [str(c) for c in cmd]
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    handle.write("$ " + " ".join(shlex.quote(a) for a in argv) + "\n")
    handle.flush()
    proc = subprocess.Popen(argv, env=full_env, cwd=cwd,
                            stdout=handle, stderr=subprocess.STDOUT,
                            start_new_session=True, stdin=subprocess.DEVNULL)
    print(f"  started pid {proc.pid}   log: {log}", flush=True)
    return proc


def wait_for(procs: list[subprocess.Popen]) -> None:
    """Block until every child has exited, reaping each one.

    `Popen.wait()` both blocks and reaps, so no zombie is left behind and the
    loop actually terminates.
    """
    failed = [f"exit {p.returncode}: {p.args[1] if len(p.args) > 1 else p.args}"
              for p in procs if (p.wait(), p.returncode)[1] != 0]
    if failed:
        raise SystemExit("Background command(s) failed:\n  " + "\n  ".join(failed))


def announce(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def require(path: Path, what: str) -> Path:
    """Assert a produced artefact exists, naming it if it does not."""
    if not path.exists():
        raise SystemExit(f"Expected {what} at {path}, but it was not created.")
    return path
