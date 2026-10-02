"""Voice: speech-to-text in front of the agent, text-to-speech behind it.

    audio ─► validate (format, size, length) ─► STT ─► confident? ─► agent ─► text answer ─► TTS ─► audio
                                                        └─ no ─► "please repeat or type" (nothing is done)

* ``audio``    - format detection by content (magic bytes, not the file name) and limits.
* ``stt``      - ``SpeechToText`` interface; local Whisper and OpenAI-compatible providers.
* ``tts``      - ``TextToSpeech`` interface; local Piper and OpenAI-compatible providers.
* ``speech``   - turning a written answer into something pleasant to listen to.
* ``pipeline`` - one voice turn end to end, with graceful degradation.
* ``factory``  - providers from settings (``none`` = voice disabled).

No module outside ``app/voice`` knows which provider is used: swapping Whisper for a
cloud service is a configuration change.
"""
