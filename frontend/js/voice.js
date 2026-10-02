// Recording a spoken question in the browser and playing spoken answers.
// MediaRecorder needs a secure context: https, or http://localhost.

import { extensionFor, pickRecordingType } from "./format.js";

const MAX_RECORDING_MS = 30_000;

export function canRecord() {
  return Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
}

export class Recorder {
  #recorder = null;
  #stream = null;
  #chunks = [];
  #timer = null;
  #done = null;

  get recording() {
    return this.#recorder?.state === "recording";
  }

  /** Ask for the microphone and start recording. `onAutoStop` fires when the time limit stops it. */
  async start(onAutoStop) {
    this.#stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = pickRecordingType((type) => MediaRecorder.isTypeSupported(type));
    this.#recorder = new MediaRecorder(this.#stream, mimeType ? { mimeType } : undefined);
    this.#chunks = [];
    this.#recorder.ondataavailable = (event) => event.data.size && this.#chunks.push(event.data);
    this.#done = new Promise((resolve) => {
      this.#recorder.onstop = () => {
        const type = this.#recorder.mimeType || mimeType || "audio/webm";
        resolve({ blob: new Blob(this.#chunks, { type }), extension: extensionFor(type) });
      };
    });
    this.#recorder.start();
    this.#timer = setTimeout(() => onAutoStop?.(), MAX_RECORDING_MS);
  }

  /** Stop and return { blob, extension }. */
  async stop() {
    clearTimeout(this.#timer);
    if (this.recording) this.#recorder.stop();
    this.#stream?.getTracks().forEach((track) => track.stop());
    return this.#done;
  }
}

/** base64 WAV from the API -> an object URL an <audio> element can play. */
export function audioUrlFromBase64(base64, type = "audio/wav") {
  const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type }));
}
