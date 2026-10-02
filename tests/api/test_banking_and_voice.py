"""GET /accounts|/transactions/{user_id} (authorization) and the voice endpoints (fake engines)."""

import base64
import uuid

import pytest

from app.voice.errors import SpeechUnavailableError
from app.voice.stt import Transcription
from tests.api.conftest import Api
from tests.fakes import wav_bytes
from tests.oracle import SEED

# --- accounts and transactions -------------------------------------------------------------------------


async def test_own_accounts(api: Api, nino_id: str) -> None:
    response = await api.client.get(f"/accounts/{nino_id}", headers=await api.login())

    body = response.json()
    mine = sorted((a for a in SEED.accounts if str(a.user_id) == nino_id), key=lambda a: a.account_number)
    assert response.status_code == 200
    assert [a["account_number"] for a in body["accounts"]] == [a.account_number for a in mine]
    assert body["totals_by_currency"]["USD"] == str(
        next(a.balance for a in mine if a.currency.value == "USD")
    )
    assert body["notice"] == "MOCK data"


@pytest.mark.parametrize("path", ["/accounts/{id}", "/transactions/{id}"])
async def test_another_customers_data_is_forbidden_and_unknown_looks_the_same(
    api: Api, giorgi_id: str, path: str
) -> None:
    headers = await api.login("CUST-0001")

    foreign = await api.client.get(path.format(id=giorgi_id), headers=headers)
    unknown = await api.client.get(path.format(id=uuid.uuid4()), headers=headers)

    assert foreign.status_code == unknown.status_code == 403
    assert foreign.json()["error"]["message"] == unknown.json()["error"]["message"]
    assert "Giorgi" not in foreign.text


async def test_own_transactions_with_filters(api: Api, nino_id: str) -> None:
    response = await api.client.get(
        f"/transactions/{nino_id}",
        params={
            "period": "custom",
            "start_date": "2026-08-01",
            "end_date": "2026-08-31",
            "category": "groceries",
            "limit": 3,
        },
        headers=await api.login(),
    )

    body = response.json()
    assert response.status_code == 200
    assert body["period"]["start_date"] == "2026-08-01" and len(body["transactions"]) == 3
    assert {t["category"] for t in body["transactions"]} == {"groceries"}
    assert body["truncated"] is True


@pytest.mark.parametrize(
    ("params", "code"),
    [
        ({"period": "custom"}, "rule_violation"),  # custom needs dates
        ({"category": "jewellery"}, "invalid_request"),
        ({"limit": 1000}, "invalid_request"),
    ],
)
async def test_bad_transaction_queries(api: Api, nino_id: str, params: dict[str, object], code: str) -> None:
    response = await api.client.get(f"/transactions/{nino_id}", params=params, headers=await api.login())
    assert response.status_code == 422 and response.json()["error"]["code"] == code


async def test_banking_data_needs_a_session(api: Api, nino_id: str) -> None:
    assert (await api.client.get(f"/accounts/{nino_id}")).status_code == 401


# --- voice --------------------------------------------------------------------------------------------


def heard(text: str, confidence: float = 0.8, language: str = "en") -> Transcription:
    return Transcription(text, language, 0.98, confidence, 1.5, "fake-stt", 5.0)


async def test_transcribe(api: Api) -> None:
    api.stt._results += [heard("What's my balance?"), heard("Ramdeni", 0.3, "ka")]
    headers = await api.login()

    clear = await api.client.post(
        "/voice/transcribe", files={"audio": ("q.wav", wav_bytes(), "audio/wav")}, headers=headers
    )
    unclear = await api.client.post(
        "/voice/transcribe", files={"audio": ("q.wav", wav_bytes())}, data={"language": "ka"}, headers=headers
    )

    assert clear.status_code == 200
    assert clear.json() | {"model": "-"} == {
        "text": "What's my balance?",
        "language": "en",
        "language_probability": 0.98,
        "confidence": 0.8,
        "duration_s": 1.5,
        "understood": True,
        "reason": None,
        "model": "-",
    }
    assert unclear.json()["understood"] is False and unclear.json()["reason"] == "low_confidence"
    assert api.stt.calls[1]["language"] == "ka"


@pytest.mark.parametrize(
    ("audio", "status", "code"),
    [
        (b"", 400, "invalid_audio"),
        (b"%PDF-1.7" + b"\x00" * 100, 415, "unsupported_audio"),
        (b"OggS" + b"\x00" * 300_000, 413, "audio_too_large"),
        (wav_bytes(31.0, rate=2000), 413, "audio_too_long"),  # small file, long recording
    ],
    ids=["empty", "pdf", "too-large", "too-long"],
)
async def test_bad_audio(api: Api, audio: bytes, status: int, code: str) -> None:
    response = await api.client.post(
        "/voice/transcribe", files={"audio": ("x", audio)}, headers=await api.login()
    )

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert api.stt.calls == []  # rejected before the recognizer


async def test_synthesize(api: Api) -> None:
    headers = await api.login()
    response = await api.client.post(
        "/voice/synthesize",
        json={"text": "Balance: ****0001 has 5.00 GEL", "language": "en"},
        headers=headers,
    )

    assert response.status_code == 200 and response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"
    assert api.tts.spoken == [("Balance: ending in 0001 has 5.00 GEL.", "en")]
    too_long = await api.client.post("/voice/synthesize", json={"text": "x" * 501}, headers=headers)
    assert too_long.status_code == 422 and too_long.json()["error"]["code"] == "text_too_long"
    bad_language = await api.client.post(
        "/voice/synthesize", json={"text": "hi", "language": "fr"}, headers=headers
    )
    assert bad_language.status_code == 422


async def test_voice_chat_full_turn(api: Api) -> None:
    api.stt._results += [heard("Transfer 50 GEL to my friend"), heard("yes")]
    headers = await api.login()

    first = (
        await api.client.post("/voice/chat", files={"audio": ("q.wav", wav_bytes())}, headers=headers)
    ).json()
    second = (
        await api.client.post(
            "/voice/chat",
            files={"audio": ("q.wav", wav_bytes())},
            data={"conversation_id": first["conversation_id"]},
            headers=headers,
        )
    ).json()

    assert first["transcription"]["text"] == "Transfer 50 GEL to my friend"
    assert first["chat"]["pending_confirmation"]["summary"] == 'transfer 50.00 GEL to "my friend"'
    assert (
        base64.b64decode(first["audio_base64"])[:4] == b"RIFF" and first["audio_content_type"] == "audio/wav"
    )
    assert second["chat"]["action_request"]["status"] == "recorded_mock_not_executed"


async def test_voice_chat_not_understood(api: Api) -> None:
    api.stt._results.append(heard("Ramdeni tanxa", 0.3, "ka"))
    body = (
        await api.client.post(
            "/voice/chat", files={"audio": ("q.wav", wav_bytes())}, headers=await api.login()
        )
    ).json()

    assert body["chat"] is None and body["transcription"]["reason"] == "low_confidence"
    assert body["response"].startswith("ბოდიში")  # asked again, in Georgian


async def test_voice_disabled_or_failing(api: Api) -> None:
    headers = await api.login()
    api.services.stt, api.services.tts = None, None
    no_stt = await api.client.post("/voice/chat", files={"audio": ("q.wav", wav_bytes())}, headers=headers)
    no_tts = await api.client.post("/voice/synthesize", json={"text": "hi"}, headers=headers)
    assert (no_stt.status_code, no_tts.status_code) == (503, 503)
    assert no_stt.json()["error"]["code"] == "voice_unavailable"

    api.services.stt = api.stt
    api.stt._results.append(SpeechUnavailableError("Speech recognition is unreachable."))
    down = await api.client.post(
        "/voice/transcribe", files={"audio": ("q.wav", wav_bytes())}, headers=headers
    )
    assert down.status_code == 503 and down.json()["error"]["message"] == "Speech recognition is unreachable."


async def test_voice_needs_a_session(api: Api) -> None:
    response = await api.client.post("/voice/transcribe", files={"audio": ("q.wav", wav_bytes())})
    assert response.status_code == 401
