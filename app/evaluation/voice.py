"""Voice round trip: text -> TTS -> STT -> text, measured.

Synthetic speech (Piper) is cleaner than a real caller on a phone, so these numbers
are an optimistic bound - but they are measured, repeatable, and they answer two
concrete questions:

1. How well does the speech recognizer understand English and Georgian banking
   questions? (word error rate, WER; character error rate, CER; script check)
2. Does the "act only on confident transcripts" rule (``STT_MIN_CONFIDENCE``)
   separate good transcripts from bad ones? A *false accept* is a bad transcript
   the assistant would act on (dangerous); a *false reject* is a good one it would
   ask to repeat (annoying).
"""

import re
import statistics
import unicodedata
from collections.abc import Sequence
from functools import partial

from pydantic import BaseModel

from app.evaluation.dataset import load_cases
from app.evaluation.schemas import EvalCategory
from app.voice.pipeline import VoicePipeline
from app.voice.stt import SpeechToText
from app.voice.tts import TextToSpeech

# Georgian banking questions (the dataset is mostly English). Translations:
GEORGIAN = (
    "რამდენი თანხა მაქვს ანგარიშზე?",  # How much money do I have in the account?
    "მაჩვენე ბოლო ტრანზაქციები.",  # Show me the latest transactions.
    "რა დავხარჯე ამ თვეში?",  # What did I spend this month?
    "რა არის დოლარის კურსი?",  # What is the dollar exchange rate?
    "როგორ დავბლოკო ბარათი?",  # How do I block the card?
    "რა ღირს საერთაშორისო გადარიცხვა?",  # What does an international transfer cost?
    "რამდენი დავხარჯე რესტორნებში?",  # How much did I spend in restaurants?
    "მინდა სესხის აღება.",  # I want to take a loan.
    "რა არის ჩემი ანგარიშის ნომერი?",  # What is my account number?
    "გადარიცხე ასი ლარი დედაჩემს.",  # Transfer one hundred lari to my mother.
    "გამარჯობა, რით შეგიძლია დამეხმარო?",  # Hello, how can you help me?
    "რა არის იპოთეკის საპროცენტო განაკვეთი?",  # What is the mortgage interest rate?
)
_SPOKEN_CATEGORIES = {
    EvalCategory.ACCOUNT, EvalCategory.TRANSACTION, EvalCategory.SPENDING, EvalCategory.CALCULATION,
    EvalCategory.RAG, EvalCategory.GENERAL, EvalCategory.VOICE,
}  # fmt: skip
_GEORGIAN_LETTER = re.compile(f"[{chr(0x10A0)}-{chr(0x10FF)}]")
GOOD_WER = 0.1  # at most ~1 wrong word in 10: the meaning survives
BAD_WER = 0.25  # a quarter of the words wrong: acting on it is a gamble


def english_sentences(every: int = 4) -> list[str]:
    """Plain English questions from the evaluation dataset (deterministic sample)."""
    texts = [
        c.input
        for c in load_cases()
        if c.category in _SPOKEN_CATEGORIES
        and c.input.isascii()
        and 8 <= len(c.input) <= 120
        and '"' not in c.input
    ]
    return texts[::every]


_SMALL_NUMBERS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty"
)
_NUMBER_WORDS = {word: str(number) for number, word in enumerate(_SMALL_NUMBERS.split())}


def normalize(text: str) -> list[str]:
    """Usual WER normalization: case, punctuation, apostrophes and small number words don't count."""
    text = (
        unicodedata.normalize("NFKC", text)
        .lower()
        .replace("'", "")
        .replace("\N{RIGHT SINGLE QUOTATION MARK}", "")
    )
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,000 -> 1000
    text = re.sub(r"[^\w\s]", " ", text)
    return [_NUMBER_WORDS.get(word, word) for word in text.split()]


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    return edit_distance(ref, normalize(hypothesis)) / max(len(ref), 1)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = "".join(normalize(reference))
    return edit_distance(ref, "".join(normalize(hypothesis))) / max(len(ref), 1)


class RoundTrip(BaseModel):
    language: str
    text: str
    heard: str
    detected_language: str | None
    confidence: float
    wer: float
    cer: float
    acted_on: bool  # would the voice pipeline pass it to the agent?
    stt_ms: float


class LanguageStats(BaseModel):
    samples: int
    mean_wer: float
    mean_cer: float
    exact: int  # transcripts identical after normalization
    acted_on: int
    false_accepts: int  # acted on although WER > BAD_WER
    false_rejects: int  # not acted on although WER <= GOOD_WER
    right_script: int  # Georgian only: transcripts written in Georgian letters
    mean_confidence: float
    mean_stt_ms: float


class VoiceReport(BaseModel):
    stt: str
    tts_voices: dict[str, str]
    min_confidence: float
    samples: list[RoundTrip]
    languages: dict[str, LanguageStats]
    threshold_window: tuple[float, float] | None  # (highest bad confidence, lowest good confidence)


def _stats(samples: list[RoundTrip], language: str) -> LanguageStats:
    return LanguageStats(
        samples=len(samples),
        mean_wer=round(statistics.fmean(s.wer for s in samples), 3),
        mean_cer=round(statistics.fmean(s.cer for s in samples), 3),
        exact=sum(s.wer == 0 for s in samples),
        acted_on=sum(s.acted_on for s in samples),
        false_accepts=sum(s.acted_on and s.wer > BAD_WER for s in samples),
        false_rejects=sum(not s.acted_on and s.wer <= GOOD_WER for s in samples),
        right_script=sum(bool(_GEORGIAN_LETTER.search(s.heard)) for s in samples) if language == "ka" else 0,
        mean_confidence=round(statistics.fmean(s.confidence for s in samples), 3),
        mean_stt_ms=round(statistics.fmean(s.stt_ms for s in samples), 1),
    )


async def run_round_trips(
    tts: TextToSpeech,
    stt: SpeechToText,
    *,
    min_confidence: float,
    english: Sequence[str] | None = None,
    georgian: Sequence[str] = GEORGIAN,
    voices: dict[str, str] | None = None,
) -> VoiceReport:
    # The same decision the live voice pipeline makes (no agent needed to evaluate it).
    decide = partial(
        VoicePipeline(None, stt, None, min_confidence=min_confidence).why_not_understood,  # type: ignore[arg-type]
        language_hint=None,
    )
    samples = []
    for language, texts in (
        ("en", english if english is not None else english_sentences()),
        ("ka", georgian),
    ):
        for text in texts:
            speech = await tts.synthesize(text, language=language)
            heard = await stt.transcribe(speech.audio)
            samples.append(
                RoundTrip(
                    language=language,
                    text=text,
                    heard=heard.text,
                    detected_language=heard.language,
                    confidence=heard.confidence,
                    wer=round(word_error_rate(text, heard.text), 3),
                    cer=round(character_error_rate(text, heard.text), 3),
                    acted_on=decide(heard) is None,
                    stt_ms=heard.latency_ms,
                )
            )
    good = [s.confidence for s in samples if s.wer <= GOOD_WER]
    bad = [s.confidence for s in samples if s.wer > BAD_WER]
    window = (max(bad), min(good)) if good and bad else None
    return VoiceReport(
        stt=getattr(stt, "name", type(stt).__name__),
        tts_voices=voices or {},
        min_confidence=min_confidence,
        samples=samples,
        languages={lang: _stats([s for s in samples if s.language == lang], lang) for lang in ("en", "ka")},
        threshold_window=window,
    )


def to_markdown(report: VoiceReport) -> str:
    lines = [
        "# Voice round trip (TTS -> STT)",
        "",
        f"* Speech recognition: `{report.stt}`; voices: "
        + ", ".join(f"`{v}` ({lang})" for lang, v in report.tts_voices.items()),
        f"* Acting threshold `STT_MIN_CONFIDENCE` = {report.min_confidence}",
        "* Synthetic speech is cleaner than real callers: read these numbers as an optimistic bound.",
        "",
        "| | English | Georgian |",
        "|---|---|---|",
    ]
    en, ka = report.languages["en"], report.languages["ka"]
    rows = [
        ("Samples", en.samples, ka.samples),
        ("Mean word error rate", f"{en.mean_wer:.1%}", f"{ka.mean_wer:.1%}"),
        ("Mean character error rate", f"{en.mean_cer:.1%}", f"{ka.mean_cer:.1%}"),
        ("Exact transcripts", f"{en.exact}/{en.samples}", f"{ka.exact}/{ka.samples}"),
        ("Written in Georgian script", "-", f"{ka.right_script}/{ka.samples}"),
        ("Mean confidence", en.mean_confidence, ka.mean_confidence),
        ("Acted on (passed to the agent)", f"{en.acted_on}/{en.samples}", f"{ka.acted_on}/{ka.samples}"),
        (f"False accepts (acted on, WER > {BAD_WER:.0%})", en.false_accepts, ka.false_accepts),
        (f"False rejects (asked again, WER <= {GOOD_WER:.0%})", en.false_rejects, ka.false_rejects),
        ("Mean recognition time", f"{en.mean_stt_ms / 1000:.2f} s", f"{ka.mean_stt_ms / 1000:.2f} s"),
    ]
    lines += [f"| {name} | {a} | {b} |" for name, a, b in rows]
    if report.threshold_window:
        low, high = report.threshold_window
        lines += ["", f"Highest confidence of a bad transcript: {low:.3f}; lowest of a good one: {high:.3f}."]
    lines += [
        "",
        "## Every sample",
        "",
        "| Lang | Said | Heard | WER | Confidence | Acted on |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {s.language} | {s.text} | {s.heard} | {s.wer:.2f} | {s.confidence:.2f} | "
        f"{'yes' if s.acted_on else 'no'} |"
        for s in report.samples
    ]
    return "\n".join(lines) + "\n"
