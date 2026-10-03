"""Create the secrets the Docker stack needs, in a directory (a Docker volume).

    python -m scripts.generate_secrets /var/lib/bankai/secrets

For each secret:

1. a value in the environment (e.g. ``OPERATOR_API_KEY`` from ``.env``) wins;
2. otherwise an existing file is kept, so secrets survive restarts;
3. otherwise a new random value is generated.

Files are written with mode 0600 and the values are never printed. The services read
them through ``SECRETS_DIR`` (see ``app.core.config.get_settings``). Nothing secret is
in the repository or baked into the image.
"""

import argparse
import os
import secrets
import sys
from collections.abc import Mapping
from pathlib import Path

SECRETS = ("mcp_auth_secret", "api_session_secret", "operator_api_key")
MIN_LENGTH = 32  # HS256 key strength; the application enforces the same minimum


class SecretError(ValueError):
    """A secret given in the environment is unusable."""


def ensure_secrets(directory: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """Make sure every secret file exists; return where each value came from."""
    directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, str] = {}
    for name in SECRETS:
        path = directory / name
        given = environ.get(name.upper(), "").strip()  # "KEY=" in a copied .env.example means unset
        if given:
            if len(given) < MIN_LENGTH:
                raise SecretError(f"{name.upper()} must be at least {MIN_LENGTH} characters")
            _write_private(path, given)
            report[name] = "from the environment"
        elif path.is_file() and len(path.read_text(encoding="utf-8").strip()) >= MIN_LENGTH:
            report[name] = "kept"
        else:
            _write_private(path, secrets.token_urlsafe(32))
            report[name] = "generated"
    return report


def _write_private(path: Path, value: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(value)
    path.chmod(0o600)  # O_CREAT's mode does not apply to a file that already existed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the secrets for the Docker stack.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=os.environ.get("SECRETS_DIR"),
        help="where to write them (default: $SECRETS_DIR)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.directory:
        print("error: give a directory or set SECRETS_DIR", file=sys.stderr)
        return 2
    try:
        report = ensure_secrets(Path(args.directory), os.environ)
    except SecretError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, source in report.items():
        print(f"secret {name}: {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
