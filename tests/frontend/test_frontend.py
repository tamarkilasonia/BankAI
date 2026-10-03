"""The web UI: how it is served, what its code may not do, and that it matches the API.

* serving + security headers (Content-Security-Policy etc.);
* no HTML injection sinks in the JavaScript (everything untrusted goes through textContent);
* a contract test: every API call in the JavaScript exists in the OpenAPI schema;
* every element the JavaScript looks up exists in index.html;
* the JavaScript unit tests and syntax checks (node), and an end-to-end run in headless
  Chrome (marked ``ui``; skipped where Chrome or Node are missing).
"""

import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from app.api.main import FRONTEND_HEADERS, create_app
from tests.conftest import PROJECT_ROOT

FRONTEND = PROJECT_ROOT / "frontend"
SCRIPTS = sorted((FRONTEND / "js").glob("*.js"))
NODE = shutil.which("node")


async def get(path: str) -> httpx.Response:
    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        return await client.get(path)


# --- serving -------------------------------------------------------------------------------------


@pytest.mark.usefixtures("reset_app_state")
async def test_the_root_opens_the_ui() -> None:
    response = await get("/")
    assert response.status_code == 307 and response.headers["location"] == "/ui/"


@pytest.mark.usefixtures("reset_app_state")
@pytest.mark.parametrize(
    ("path", "content_type"),
    [("/ui/", "text/html"), ("/ui/styles.css", "text/css"), ("/ui/js/main.js", "javascript")],
)
async def test_ui_files_are_served_with_security_headers(path: str, content_type: str) -> None:
    response = await get(path)

    assert response.status_code == 200 and content_type in response.headers["content-type"]
    for header, value in FRONTEND_HEADERS.items():
        assert response.headers[header] == value
    csp = response.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp


@pytest.mark.usefixtures("reset_app_state")
async def test_api_docs_are_not_restricted_by_the_ui_policy() -> None:
    """Swagger UI loads from a CDN; the strict UI policy must not break /docs."""
    assert "content-security-policy" not in (await get("/docs")).headers


def test_the_page_promises_mock_data() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert "MOCK" in html and "<script>" not in html and "style=" not in html  # nothing inline (CSP)


# --- the code -------------------------------------------------------------------------------------

# Any mention counts ("{ innerHTML: x }" via Object.assign is as dangerous as ".innerHTML = x").
SINKS = re.compile(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function\(|srcdoc")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_html_injection_sinks(script: Path) -> None:
    """Agent answers and evaluation inputs (attack strings!) must never be parsed as HTML."""
    assert not SINKS.findall(script.read_text(encoding="utf-8"))


CALL = re.compile(r"""request\(\s*"(GET|POST)",\s*[`"]([^`"]+)[`"]""")


def test_every_api_call_exists_in_the_openapi_schema() -> None:
    schema_paths = create_app().openapi()["paths"]
    by_shape = {re.sub(r"\{[^}]+\}", "{}", path): methods for path, methods in schema_paths.items()}
    calls = [
        (method, re.sub(r"\$\{[^}]+\}", "{}", path.split("?")[0]))
        for script in SCRIPTS
        for method, path in CALL.findall(script.read_text(encoding="utf-8"))
    ]

    assert len(calls) >= 9
    for method, path in calls:
        assert path in by_shape, f"{method} {path} is not an API endpoint"
        assert method.lower() in by_shape[path], f"{method} {path}: wrong method"


def test_every_element_the_code_uses_exists() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    page_ids = set(re.findall(r'\bid="([^"]+)"', html))
    used = {
        i for script in SCRIPTS for i in re.findall(r'byId\("([^"]+)"\)', script.read_text(encoding="utf-8"))
    }
    assert used and used <= page_ids, used - page_ids


# --- node: syntax and unit tests -------------------------------------------------------------------


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_javascript_syntax(script: Path) -> None:
    result = subprocess.run([NODE, "--check", str(script)], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_javascript_unit_tests() -> None:
    result = subprocess.run(  # noqa: S603
        [NODE, "--test", str(PROJECT_ROOT / "tests/frontend/format.test.mjs")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- end to end in a real browser --------------------------------------------------------------------

CHROME_CANDIDATES = [
    os.environ.get("CHROME", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]
CHROME = next((c for c in CHROME_CANDIDATES if c and Path(c).exists()), None)
OPERATOR_KEY = "ui-test-operator-key-0123456789abcdef"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


@pytest.fixture
def live_server(tmp_path: Path) -> Iterator[str]:
    """uvicorn on a seeded SQLite bank (offline agent), with one evaluation run already stored."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'ui.db'}"
    env = {
        **os.environ,
        "DATABASE_URL": url,
        "OPERATOR_API_KEY": OPERATOR_KEY,
        "VECTOR_STORE": "memory",
        "LLM_PROVIDER": "none",
        "LOG_LEVEL": "WARNING",
    }
    seed = (
        "import asyncio, sys\n"
        "from sqlalchemy.ext.asyncio import async_sessionmaker\n"
        "import app.models\n"
        "from app.core.config import Settings\n"
        "from app.database.base import Base\n"
        "from app.database.seed import seed_database\n"
        "from app.database.session import create_engine\n"
        "from app.evaluation.dataset import load_cases, select_cases\n"
        "from app.evaluation.runner import evaluate\n"
        "async def main():\n"
        f"    engine = create_engine({url!r})\n"
        "    async with engine.begin() as conn:\n"
        "        await conn.run_sync(Base.metadata.create_all)\n"
        "    sessions = async_sessionmaker(engine, expire_on_commit=False)\n"
        "    async with sessions() as session:\n"
        "        await seed_database(session)\n"
        "    cases = select_cases(load_cases(), categories=['GENERAL', 'SAFETY'])\n"
        "    offline = Settings(_env_file=None, llm_provider='none')\n"
        "    await evaluate(offline, cases=cases, results_store=sessions)\n"
        "    await engine.dispose()\n"
        "asyncio.run(main())\n"
    )
    subprocess.run([sys.executable, "-c", seed], env=env, check=True, cwd=PROJECT_ROOT)  # noqa: S603
    port = _free_port()
    server = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "uvicorn", "app.api.main:app", "--port", str(port)],
        env=env,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code in {200, 503}:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        yield base
    finally:
        server.terminate()
        server.wait(timeout=10)


@pytest.fixture
def browser() -> str:
    """Chrome, or a skip - a failure when REQUIRE_BROWSER=1 (CI), so the test never vanishes silently."""
    if CHROME is None or NODE is None:
        if os.getenv("REQUIRE_BROWSER") == "1":
            pytest.fail("Chrome and node are required (REQUIRE_BROWSER=1)")
        pytest.skip("needs Chrome and node")
    return CHROME


@pytest.mark.ui
def test_end_to_end_in_headless_chrome(browser: str, live_server: str) -> None:
    result = subprocess.run(  # noqa: S603
        [NODE, str(PROJECT_ROOT / "tests/frontend/ui_smoke.mjs"), live_server, OPERATOR_KEY],
        capture_output=True,
        text=True,
        check=False,
        # SLOW_FIRST_LOAD: the page arrives slowly, as on a busy CI machine, where the test once
        # read the page before it had loaded ("page error: Uncaught") - it must wait instead
        env={**os.environ, "CHROME": browser, "SLOW_FIRST_LOAD": "1"},
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED (6 steps)" in result.stdout
