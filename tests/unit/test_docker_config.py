"""The container setup, checked without Docker.

These rules are easy to break with a one-line edit and hard to notice afterwards: a
published MCP port, a secret typed into the compose file, a root user, an environment
variable with a typo that the application silently ignores. ``tests/docker`` runs the
real stack; this file runs everywhere, in a second.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.core.config import Settings
from tests.conftest import PROJECT_ROOT

COMPOSE_FILES = ["docker-compose.yml", "docker-compose.ollama.yml", "docker-compose.voice.yml"]
APP_SERVICES = ["init", "mcp-server", "backend"]
SETTINGS = {name.upper() for name in Settings.model_fields}
COMPOSE_ONLY = {"SECRETS_DIR"}  # read by get_settings itself, not a field


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((PROJECT_ROOT / name).read_text(encoding="utf-8"))
    return data


BASE = load("docker-compose.yml")["services"]


def environment(service: dict[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in (service.get("environment") or {}).items()}


# --- docker-compose.yml -------------------------------------------------------------------------


def test_the_services_of_the_specification_exist() -> None:
    assert {"backend", "database", "mcp-server"} <= set(BASE)


@pytest.mark.parametrize("name", COMPOSE_FILES)
def test_no_secret_is_written_in_a_compose_file(name: str) -> None:
    """Secrets are generated at start-up or come from .env - never typed into the file."""
    for service, definition in load(name)["services"].items():
        for key, value in environment(definition).items():
            if re.search(r"(SECRET|PASSWORD|API_KEY|TOKEN)$", key):
                assert value.startswith("${"), f"{name}: {service}.{key} is a literal value"


def test_only_backend_and_database_are_published_and_only_on_localhost() -> None:
    published = {name: svc["ports"] for name, svc in BASE.items() if svc.get("ports")}

    assert set(published) == {"backend", "database"}  # the MCP tool server stays internal
    assert all(str(port).startswith("127.0.0.1:") for ports in published.values() for port in ports)


@pytest.mark.parametrize("name", APP_SERVICES)
def test_application_containers_are_hardened(name: str) -> None:
    service = BASE[name]
    assert service["read_only"] is True
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["cap_drop"] == ["ALL"]
    assert "secrets:/var/lib/bankai/secrets" in service["volumes"]


@pytest.mark.parametrize("name", COMPOSE_FILES)
def test_every_environment_variable_is_a_real_setting(name: str) -> None:
    """A typo (MCP_ALLOWED_HOST) would be silently ignored by the application."""
    for service in APP_SERVICES:
        definition = load(name)["services"].get(service, {})
        unknown = set(environment(definition)) - SETTINGS - COMPOSE_ONLY
        assert not unknown, f"{name}: {service} sets unknown variables {sorted(unknown)}"


def test_the_backend_uses_the_mcp_server_over_http() -> None:
    backend, mcp = environment(BASE["backend"]), environment(BASE["mcp-server"])
    assert backend["MCP_TRANSPORT"] == "http"
    assert backend["MCP_SERVER_URL"] == "http://mcp-server:8001/mcp"
    assert "mcp-server:*" in mcp["MCP_ALLOWED_HOSTS"]  # its Host header passes the DNS-rebinding check


def test_start_up_order() -> None:
    def waits_for(service: str) -> dict[str, str]:
        return {name: rule["condition"] for name, rule in BASE[service]["depends_on"].items()}

    assert waits_for("init") == {"database": "service_healthy"}
    assert waits_for("mcp-server") == {"init": "service_completed_successfully"}
    assert waits_for("backend") == {"init": "service_completed_successfully", "mcp-server": "service_healthy"}


def test_the_init_job_is_safe_to_run_on_every_start() -> None:
    command = " ".join(BASE["init"]["command"])
    for step in ["generate_secrets", "alembic upgrade head", "seed_db --if-empty", "ingest_knowledge_base"]:
        assert step in command
    assert BASE["init"]["restart"] == "no"


def test_the_default_mode_needs_no_llm_and_no_key() -> None:
    backend = environment(BASE["backend"])
    assert (backend["LLM_PROVIDER"], backend["EMBEDDING_PROVIDER"]) == ("none", "hashing")


def test_the_add_ons_change_only_what_they_are_for() -> None:
    ollama = load("docker-compose.ollama.yml")["services"]
    assert all(
        set(environment(s)) <= {k for k in SETTINGS if k.startswith(("LLM_", "EMBEDDING_"))}
        for s in ollama.values()
    )
    voice = load("docker-compose.voice.yml")["services"]
    assert set(voice) == {"backend"} and voice["backend"]["volumes"] == ["./models:/models:ro"]


# --- Dockerfile and build context -------------------------------------------------------------

DOCKERFILE = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_the_image_runs_as_an_unprivileged_user() -> None:
    users = re.findall(r"^USER\s+(\S+)", DOCKERFILE, re.MULTILINE)
    assert users and users[-1] not in {"root", "0"}


def test_the_image_has_a_health_check_and_a_pinned_base() -> None:
    assert re.search(r"^HEALTHCHECK ", DOCKERFILE, re.MULTILINE)
    bases = re.findall(r"^FROM (\S+)", DOCKERFILE, re.MULTILINE)
    assert len(bases) == 2 and len(set(bases)) == 1  # builder and runtime: the same Python
    assert re.fullmatch(r"python:3\.12\.\d+-slim-\w+", bases[0])  # an exact version, never :latest
    assert not re.search(r"^ENV .*(SECRET|PASSWORD|API_KEY)", DOCKERFILE, re.MULTILINE)


def test_the_build_context_is_an_allowlist() -> None:
    rules = [
        line.strip()
        for line in (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rules[0] == "*"  # exclude everything first ...
    allowed = {rule[1:].rstrip("/") for rule in rules if rule.startswith("!")}
    # ... then only what the image needs: never .env, .git, tests or the speech models.
    assert allowed == {
        "app",
        "scripts",
        "alembic",
        "alembic.ini",
        "knowledge_base",
        "frontend",
        "requirements.txt",
        "requirements-voice.txt",
    }


def test_every_setting_is_documented_in_env_example() -> None:
    """Specification: document the environment variables."""
    documented = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert not [name for name in sorted(SETTINGS) if name not in documented]


def test_everything_the_image_needs_at_run_time_is_a_runtime_dependency() -> None:
    """Found by building the image: the evaluation runner needs aiosqlite, which used to be
    a development dependency only - the dashboard's "Run evaluation" would have crashed."""
    runtime = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^aiosqlite==", runtime, re.MULTILINE)
    assert "sqlite+aiosqlite" in (PROJECT_ROOT / "app/evaluation/runner.py").read_text(encoding="utf-8")


def test_no_compose_file_is_missing() -> None:
    assert all(Path(PROJECT_ROOT / name).is_file() for name in COMPOSE_FILES)
