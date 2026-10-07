"""Voice pipeline worker — runs the VAD/STT/NLU chain off the flight loop."""

import os
import tempfile
import threading

import cv2

from control.voice_model import (
    describe_scene,
    listen_for_utterance_vad,
    resolve_command,
    speak,
    transcribe,
)


class VoiceWorker:
    """Voice pipeline in a background thread — never blocks the flight loop.

    Runs the full listen_for_utterance_vad → transcribe → resolve_command
    chain.  describe_scene commands are handled entirely here (capture
    latest frame from the existing frame buffer, call VLM, speak result).
    All other commands are pushed into the shared queue for the main
    loop to drain non-blockingly via from_voice() + merge_commands().
    """

    def __init__(self, frame_read, pose_worker, reporter_worker,
                 reporter_toggle, stop_event, cmd_queue):
        self._frame_read = frame_read
        self._pose_worker = pose_worker
        self._reporter_worker = reporter_worker  # None => no reporter guard
        self._reporter_toggle = reporter_toggle  # O-key bypass switch
        self._stop = stop_event          # shared threading.Event
        self._queue = cmd_queue          # shared queue.Queue
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="voice-worker"
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _person_present(self):
        """VAD gate — always listen, so takeoff/stop work with nobody in frame.

        Person-presence is NOT used to gate the microphone anymore; instead
        it filters commands AFTER transcription (see _run): only takeoff /
        land / hover (stop) / emergency are allowed without ANY detection.
        Everything else needs BOTH a person (pose gate) AND the reporter
        (mic holder from ReporterTracker) to be in frame.  Returns False
        when the stop event is set so listen_for_utterance_vad drains its
        queue and the outer while-loop can exit cleanly.
        """
        return not self._stop.is_set()

    def _run(self):
        while not self._stop.is_set():
            audio = listen_for_utterance_vad(
                person_present_fn=self._person_present
            )
            if self._stop.is_set():
                break
            if audio is None:
                continue

            transcript = transcribe(audio)
            print(f"[voice] transcript: \"{transcript}\"")

            if not transcript:
                continue

            command, path = resolve_command(transcript)
            print(f"[voice] path={path} -> {command}")

            action = command.get("action")

            # takeoff / land / hover (stop) / emergency are allowed with
            # NO detection at all.  Everything else needs BOTH the pose
            # gate (person present) AND the reporter gate (the identified
            # mic holder is still in frame).  If the reporter tracker
            # failed to start, fall back to the pose-only gate.
            if action not in ("takeoff", "land", "hover", "emergency"):
                if not self._pose_worker.is_person_present():
                    print(
                        "[voice] no person in frame — "
                        "ignoring (only takeoff/stop work without pose)"
                    )
                    continue
                if (self._reporter_toggle.enabled
                        and self._reporter_worker is not None
                        and not self._reporter_worker.is_reporter_present()):
                    print(
                        "[voice] reporter (mic holder) not in frame — "
                        "ignoring (only takeoff/stop work without "
                        "detection)"
                    )
                    continue

            if action == "describe_scene":
                # Reuse the main loop's frame buffer — no second
                # cv2.VideoCapture.  Save to a temp file so
                # describe_scene() can read it via its data-URL path.
                frame = self._frame_read.frame
                if frame is not None:
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                    )
                    try:
                        cv2.imwrite(tmp.name, frame)
                        description = describe_scene(tmp.name)
                        if description:
                            print(
                                f"[voice] scene: \"{description}\""
                            )
                            speak(description)
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except OSError:
                            pass
                else:
                    print(
                        "[voice] no frame available "
                        "for scene description"
                    )
            else:
                self._queue.put(command)
