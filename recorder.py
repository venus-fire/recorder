#!/usr/bin/env python3
"""Digital voice recorder ingest pipeline.

Plug the recorder in -> detect it -> mount -> copy new recordings to
~/Recordings -> unmount -> transcribe on the GPU with faster-whisper ->
write raw transcript notes into the Obsidian vault -> summarize each note
through the Fireworks API -> pull the action items that concern the subject
into the todo note -> run the vault maintenance loop -> remount, delete the
ingested files from the recorder, unmount again.

Everything is idempotent and resumable: state lives in
~/.local/share/recorder/state.db keyed by the sha256 of each recording, so a
crash, an OOM kill, or an unplug mid-run costs at most the work in flight.
Nothing is deleted from the recorder until its transcript note exists and the
local copy still hashes to what was recorded.

Settings come from DEFAULTS below, overridden by ~/.config/recorder/config.toml.
See config.example.toml and README.md.

Commands:
    recorder watch                 long-running watcher (systemd service)
    recorder ingest                one pass, everything
    recorder copy                  one pass, copy phase only
    recorder transcribe [--all]    transcribe pending (--all = redo everything)
    recorder summarize  [--all]    summarize pending (--all = redo everything)
    recorder todo [--all] [--only STEM]
                                   pull action items into the todo note
    recorder delete                delete verified+noted files off the device
    recorder status                show pipeline state
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

DEFAULTS: dict = {
    "device_label": "WS-852",
    "device_id_glob": "usb-OLYMPUS_DVR*",
    "audio_dir": "~/Recordings",
    "vault": "~/Documents/OBSIDIAN",
    "transcripts_subdir": "Transcripts",
    "model": "~/.local/share/recorder/models/whisper-large-v3-turbo",
    "compute_type": "float16",
    "beam_size": 5,
    # VAD decides what whisper is allowed to see, and off is the measured
    # default. Silero scores the quiet far end of a phone call below its
    # threshold and drops it, so a 6m26s call arrives as her half only: one
    # recording lost 4x its content that way, and the todo pass then had
    # nothing to work with. Speech is still MEASURED with VAD (see
    # measure_speech_seconds) so the no-speech gate and the banner keep
    # working; it is simply not used to cut the audio.
    "vad": False,
    "language": "",  # empty = auto-detect per file
    # Vocab the transcriber should expect, so names and jargon come out right
    # instead of phonetically. Keep it to a sentence or two: it biases every
    # decode. On a nearly silent file whisper can echo this back as if it were
    # speech, which is what min_speech_seconds exists to catch.
    # Set your own in config.toml, e.g. "Voice notes. Sam Rivera, Northside
    # Clinic, CompTIA A+, Portland Oregon."
    "initial_prompt": "Voice notes and recorded conversations.",
    # Who the pipeline is acting for. Appears in the prompts that decide whose
    # action items these are.
    "subject": "the user",
    "summarize": True,
    "summary_model": "accounts/fireworks/routers/deepseek-flash-latest",
    "summary_base_url": "https://api.fireworks.ai/inference/v1",
    "summary_max_tokens": 6000,
    # "date" prefixes each summary filename with its recording date so
    # content-derived names still sort chronologically. "none" for bare names.
    "summary_name_prefix": "date",
    "fireworks_env_file": "~/.hermes/.env",
    "delete_from_device": True,
    "notify": True,
    "notify_target": "simplex",
    # The hermes CLI, for the message and the agent fallback. On PATH by
    # default; give an absolute path if it is not, which matters for the
    # systemd unit, whose PATH is not your shell's.
    "hermes_bin": "hermes",
    # Desktop notifications (KDE/libnotify). The final "safe to unplug" one
    # always fires; the per-phase progress ones are behind their own switch.
    "desktop_notify": True,
    "desktop_notify_progress": True,
    "desktop_icon": "audio-input-microphone",
    "poll_seconds": 3,
    # Absolute paths to the media tools if they are not on PATH.
    "ffmpeg": "",
    "ffprobe": "",
    "min_speech_seconds": 2.0,
    "log_max_bytes": 2_000_000,
    # Inline tags written into each note body. The vault's smart-autocorrect
    # plugin (syncFrontmatterTags) rebuilds the frontmatter `tags` field from
    # these, so they must be present or the field gets deleted.
    "tag_note": "transcript",
    "tag_voice": "voice-note",
    "tag_summary": "summary",
    # Action items: a headless Hermes run reads each new summary and edits the
    # todo note. Deciding what concerns her needs judgement, so this is an agent
    # run rather than a parser. Only recordings ingested from now on qualify;
    # the old backlog is marked handled.
    "todo_extract": True,
    "todo_note": "~/Documents/OBSIDIAN/To_Do/GENERAL_TODO.md",
    # "api" = one Fireworks call that returns JSON, and the pipeline inserts the
    # items itself. "agent" = the headless Hermes run, kept as a fallback for
    # when the extraction needs to look something up. The API path is ~1/10th
    # the wall clock: the agent spends its time on framework startup and tool
    # round trips, and it wandered badly when given the run of the machine.
    "todo_mode": "api",
    # Not the summarizer's model. That router reasons before it answers and
    # `max_tokens` covers the reasoning, so on this task it repeatedly spent the
    # whole budget thinking and returned nothing, at 50s a try. Measured on the
    # same prompt: gpt-oss-120b answered in 3s with 8 items; the reasoning
    # router took 48-50s and returned 2, or empty. nemotron-lightning, glm-5p3,
    # glm-5p3-fast and deepseek-v4p1-flash all returned empty.
    "todo_model": "accounts/fireworks/models/gpt-oss-120b",
    "todo_max_tokens": 8000,
    # Required, not a tuning knob. Measured on one real recording: low returns
    # JSON in 20s using 3k reasoning tokens; medium and the unset default both
    # spent the entire 8k budget thinking and returned empty content after 150s.
    # The budget caps the reasoning, so raising it makes things worse: 16k
    # produced 16k of reasoning and then needed 32k.
    "todo_reasoning": "low",
    # When the API path comes back empty, hand the recording to the slow agent
    # rather than losing the items. Layered on purpose: this router is not
    # deterministic, so the same prompt answers in 20s once and blows the budget
    # the next time.
    "todo_fallback_agent": True,
    # On PATH by default. Give an absolute path if hermes is not.
    "todo_agent": "hermes",
    # file only: terminal let the agent wander into git, ls and the vault
    # scripts. Nothing it needs lives outside read/write/patch/search_files.
    "todo_toolsets": "file",
    "todo_skills": "obsidian",
    "todo_max_turns": 12,
    "todo_run_budget": 180,
    "todo_timeout": 300,
}

CONFIG_FILE = Path("~/.config/recorder/config.toml").expanduser()
HOME = Path.home()
BASE = Path("~/.local/share/recorder").expanduser()
LOG_FILE = BASE / "recorder.log"
DB_FILE = BASE / "state.db"
LOCK_FILE = BASE / "ingest.lock"
STAGING = BASE / "staging"  # holdout for files mid-verify, same filesystem as audio_dir

SKIP_DIRS = {"lost.dir", "system", "ol_manual", "system volume information"}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("rb") as fh:
            cfg.update(tomllib.load(fh))
    for key in ("audio_dir", "vault", "model", "fireworks_env_file"):
        cfg[key] = str(Path(os.path.expanduser(str(cfg[key]))))
    return cfg


CFG = load_config()


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    try:
        BASE.mkdir(parents=True, exist_ok=True)
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > CFG["log_max_bytes"]:
            LOG_FILE.replace(LOG_FILE.with_suffix(".log.1"))
        with LOG_FILE.open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# desktop notifications
# --------------------------------------------------------------------------

# KDE on Wayland. notify-send reaches the session bus from the user unit
# because systemd --user carries DBUS_SESSION_BUS_ADDRESS and WAYLAND_DISPLAY
# (verified with systemd-run --user). Failures are logged, never fatal: a
# notification that cannot be shown must not break an ingest.
def desktop_notify(title: str, body: str = "", urgency: str = "normal",
                   progress: bool = False, icon: str = "") -> None:
    if not CFG["desktop_notify"]:
        return
    if progress and not CFG["desktop_notify_progress"]:
        return
    exe = shutil.which("notify-send")
    if not exe:
        log("desktop-notify: notify-send not found")
        return
    cmd = [exe, "-a", "Recorder", "-u", urgency,
           "-i", icon or CFG["desktop_icon"], title]
    if body:
        cmd.append(body)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if res.returncode != 0:
            log(f"desktop-notify: exit={res.returncode} {(res.stderr or '').strip()[:150]}")
    except Exception as exc:
        log(f"desktop-notify: failed {exc}")


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


# --------------------------------------------------------------------------
# cuda runtime
# --------------------------------------------------------------------------

def ensure_cuda_env() -> None:
    """ctranslate2 needs the pip-installed cublas/cudnn on the loader path.

    LD_LIBRARY_PATH has to be set before the process starts, so this re-execs
    once with it patched. When there is no script path to re-exec (imported
    from stdin or -c), it preloads the same libraries with RTLD_GLOBAL
    instead, which the loader will then reuse.
    """
    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    dirs = [site / "nvidia" / "cublas" / "lib", site / "nvidia" / "cudnn" / "lib"]
    if not dirs[0].is_dir():
        return
    wanted = [str(d) for d in dirs]
    current = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if all(w in current for w in wanted):
        return

    script = sys.argv[0] if sys.argv else ""
    if script and os.path.isfile(script):
        os.environ["LD_LIBRARY_PATH"] = ":".join(wanted + [p for p in current if p])
        os.execv(sys.executable, [sys.executable] + sys.argv)

    import ctypes

    for lib in ("libcublasLt.so.12", "libcublas.so.12"):
        path = dirs[0] / lib
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    for path in sorted(dirs[1].glob("libcudnn*.so.9")):
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


# Called at import, not just in main(): importing this module from another
# script must get a working CUDA environment too, and this has to happen
# before ctranslate2 is imported.
ensure_cuda_env()


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    sha256        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    rel_path      TEXT NOT NULL,
    size          INTEGER NOT NULL,
    duration      REAL,
    recorded_date TEXT,
    copied_at     TEXT,
    audio_path    TEXT,
    note_path     TEXT,
    transcribed_at TEXT,
    language      TEXT,
    speech_seconds REAL,
    summary_path  TEXT,
    summarized_at TEXT,
    transcript    TEXT,
    todo_at       TEXT,
    todo_added    TEXT,
    no_speech_at  TEXT,
    deleted_at    TEXT,
    error         TEXT
);
"""


def db() -> sqlite3.Connection:
    BASE.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    # Migrate older databases that predate a column.
    have = {row[1] for row in conn.execute("PRAGMA table_info(recordings)")}
    for column in ("transcript", "todo_at", "todo_added", "no_speech_at"):
        if column not in have:
            conn.execute(f"ALTER TABLE recordings ADD COLUMN {column} TEXT")
    return conn


# --------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------

def device_node() -> Path | None:
    by_label = Path("/dev/disk/by-label") / CFG["device_label"]
    if by_label.exists():
        try:
            return by_label.resolve()
        except OSError:
            return None
    for link in sorted(Path("/dev/disk/by-id").glob(CFG["device_id_glob"] + "-part*")):
        try:
            return link.resolve()
        except OSError:
            continue
    return None


def mountpoint(dev: Path) -> Path | None:
    out = subprocess.run(
        ["/usr/bin/findmnt", "-n", "-o", "TARGET", "--source", str(dev)],
        capture_output=True, text=True,
    )
    target = out.stdout.strip().splitlines()
    return Path(target[0]) if target else None


def udev_settle(seconds: float = 1.5) -> None:
    subprocess.run(["/usr/bin/udevadm", "settle", "--timeout", str(int(seconds))],
                   capture_output=True)


def mount_device(dev: Path, timeout: float = 20.0) -> Path | None:
    mp = mountpoint(dev)
    if mp:
        return mp
    udev_settle()
    for attempt in range(3):
        res = subprocess.run(["/usr/bin/udisksctl", "mount", "-b", str(dev)],
                             capture_output=True, text=True)
        if res.returncode == 0 or "already mounted" in (res.stderr + res.stdout):
            break
        log(f"mount attempt {attempt + 1} failed: {(res.stderr or res.stdout).strip()}")
        time.sleep(2)
    deadline = time.time() + timeout
    while time.time() < deadline:
        mp = mountpoint(dev)
        if mp and mp.is_dir():
            return mp
        time.sleep(0.5)
    return None


def unmount_device(dev: Path) -> bool:
    if not mountpoint(dev):
        return True
    res = subprocess.run(["/usr/bin/udisksctl", "unmount", "-b", str(dev)],
                         capture_output=True, text=True)
    ok = res.returncode == 0 or not mountpoint(dev)
    if not ok:
        log(f"unmount failed: {(res.stderr or res.stdout).strip()}")
    return ok


def collect_recordings(root: Path) -> list[Path]:
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".mp3", ".wav", ".m4a"):
            continue
        if any(part.lower() in SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        found.append(path)
    return found


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration(path: Path) -> float | None:
    res = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(res.stdout.strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------
# phase 1: copy
# --------------------------------------------------------------------------

def copy_phase() -> list[sqlite3.Row]:
    dev = device_node()
    if not dev:
        log("copy: no recorder attached")
        return []
    mp = mount_device(dev)
    if not mp:
        log("copy: could not mount the recorder")
        return []
    log(f"copy: recorder mounted at {mp}")

    audio_dir = Path(CFG["audio_dir"])
    audio_dir.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)

    conn = db()
    new_rows: list[sqlite3.Row] = []
    files = collect_recordings(mp)
    log(f"copy: {len(files)} audio files on the device")

    for src in files:
        rel = str(src.relative_to(mp))
        digest = sha256_of(src)
        if conn.execute("SELECT 1 FROM recordings WHERE sha256=?", (digest,)).fetchone():
            continue
        target = audio_dir / src.name
        if target.exists() and sha256_of(target) != digest:
            stem, suffix, n = target.stem, target.suffix, 1
            while target.exists():
                target = audio_dir / f"{stem}_{n}{suffix}"
                n += 1
        staged = STAGING / (src.name + ".part")
        shutil.copy2(src, staged)
        if sha256_of(staged) != digest:
            log(f"copy: HASH MISMATCH for {src.name}, discarded")
            staged.unlink(missing_ok=True)
            continue
        os.replace(staged, target)
        duration = probe_duration(target)
        recorded = recorded_date_from_name(src.name)
        conn.execute(
            """INSERT INTO recordings
               (sha256, name, rel_path, size, duration, recorded_date, copied_at, audio_path)
               VALUES (?,?,?,?,?,?,?,?)""",
            (digest, src.name, rel, src.stat().st_size, duration, recorded,
             datetime.now().strftime("%Y-%m-%d %H:%M"), str(target)),
        )
        conn.commit()
        log(f"copy: {src.name} -> {target} ({duration:.1f}s)" if duration
            else f"copy: {src.name} -> {target}")
        new_rows.append(conn.execute("SELECT * FROM recordings WHERE sha256=?", (digest,)).fetchone())

    if not unmount_device(dev):
        log("copy: recorder left mounted; unplug is not safe yet")
        desktop_notify("Recorder still mounted",
                       f"copied {plural(len(new_rows), 'recording')}, unmount failed",
                       urgency="critical", progress=True)
    else:
        log("copy: recorder unmounted")
        if new_rows:
            desktop_notify(f"Copied {plural(len(new_rows), 'recording')}",
                           "card unmounted, safe to unplug if you need it",
                           progress=True)
        else:
            desktop_notify("Recorder connected",
                           "no new recordings on the card", progress=True)
    return new_rows


def recorded_date_from_name(name: str) -> str | None:
    m = re.match(r"^(\d{2})(\d{2})(\d{2})_", name)
    if not m:
        return None
    yy, mm, dd = m.groups()
    return f"20{yy}-{mm}-{dd}"


# --------------------------------------------------------------------------
# phase 2: transcribe
# --------------------------------------------------------------------------

# ffmpeg and ffprobe decode the MP3s. Resolved from PATH, with an optional
# override in config.toml for a machine that keeps them somewhere unusual.
FFMPEG = str(CFG.get("ffmpeg") or "") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = str(CFG.get("ffprobe") or "") or shutil.which("ffprobe") or "ffprobe"


def decode_audio(path: Path, rate: int = 16000):
    import numpy as np

    res = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(res.stdout, dtype=np.float32)


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_duration(seconds: float | None) -> str:
    if not seconds:
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def paragraphs(segments: list[dict], min_gap: float = 1.2, max_len: int = 420) -> str:
    out: list[tuple[float, str]] = []
    start = None
    last_end = 0.0
    cur: list[str] = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if start is None:
            start, cur = seg["start"], [text]
        elif (seg["start"] - last_end) > min_gap or sum(len(c) + 1 for c in cur) > max_len:
            out.append((start, " ".join(cur)))
            start, cur = seg["start"], [text]
        else:
            cur.append(text)
        last_end = seg["end"]
    if cur and start is not None:
        out.append((start, " ".join(cur)))
    return "\n\n".join(f"[{fmt_ts(t)}] {body}" for t, body in out)


def measure_speech_seconds(audio) -> float | None:
    """How much of the file is speech, by the same VAD faster-whisper uses.

    Measured, not filtered: the number drives the no-speech gate and the
    "mostly non-speech" banner, while transcription itself sees the whole
    file. None means the measurement failed, and the caller falls back to the
    full duration so a broken VAD can never mark a real recording silent.
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        spans = get_speech_timestamps(audio, VadOptions())
        return sum(max(0, s["end"] - s["start"]) for s in spans) / 16000
    except Exception as exc:  # noqa: BLE001 - a measurement must not break an ingest
        log(f"vad: speech measurement failed: {exc}")
        return None


def transcribe_file(model, path: Path) -> dict:
    audio = decode_audio(path)
    duration = len(audio) / 16000
    kwargs = dict(
        beam_size=CFG["beam_size"],
        vad_filter=bool(CFG["vad"]),
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    if CFG["language"]:
        kwargs["language"] = CFG["language"]
    if CFG["initial_prompt"]:
        kwargs["initial_prompt"] = CFG["initial_prompt"]
    speech = measure_speech_seconds(audio)
    segments, info = model.transcribe(audio, **kwargs)
    segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    return {
        "segments": segs,
        "duration": duration,
        # Speech seconds come from the VAD measurement, not from
        # info.duration_after_vad: with vad_filter off the latter is the whole
        # file, and the gap between speech and duration is what tells a
        # mostly-music recording from a voice note.
        "speech_seconds": speech if speech is not None else duration,
        "language": info.language,
        "language_probability": info.language_probability,
        "model_speech_seconds": getattr(info, "duration_after_vad", None),
    }


def frontmatter(**fields) -> str:
    lines = ["---"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {quote(item)}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {quote(str(value))}")
    lines.append("---")
    return "\n".join(lines)


def quote(value: str) -> str:
    if value != value.strip() or ": " in value or value.startswith(("[", "{", "*", "&", "!", "|", ">", "@", "`", "#", "%")):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def write_raw_note(row, result: dict, body: str) -> Path:
    raw_dir = Path(CFG["vault"]) / CFG["transcripts_subdir"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(row["name"]).stem
    note = raw_dir / f"{stem}.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    recorded = row["recorded_date"] or now[:10]
    dur = fmt_duration(result["duration"])
    title = stem
    speech = result["speech_seconds"] or 0.0
    total = result["duration"] or 0.0
    meta = f"> {dur} · recorded {recorded} · audio `{row['audio_path']}`"
    # Say when a file is mostly non-speech, so a 5-minute note that reads like
    # one line explains itself. With vad off the rest was still transcribed, so
    # the wording says what was measured rather than what was filtered.
    if total > 30 and speech < total * 0.25:
        meta += (f"\n>\n> _Mostly non-speech audio: {fmt_duration(speech)} of speech "
                 f"detected in {dur}. Singing, music and noise are transcribed "
                 f"as heard, so expect lyrics and fragments from them._")
    description = f"Voice recording transcript, {dur}, recorded {recorded}"
    text = "\n".join([
        frontmatter(
            title=title,
            status="active",
            category=CFG["transcripts_subdir"],
            created=f"{recorded} 00:00",
            updated=now,
            description=description,
            generated="process/recorder-ingest",
            generated_at=now,
            recorded=recorded,
            duration=dur,
            language=result["language"],
            resource=f"file://{row['audio_path']}",
            sources=[f"Olympus WS-852 voice recorder • {row['name']}"],
        ),
        "",
        f"# {title}",
        "",
        meta,
        "",
        body,
        "",
        # Inline tags, not frontmatter tags: the vault's smart-autocorrect
        # plugin (syncFrontmatterTags) rewrites the frontmatter `tags` field
        # from the inline #tags in the body, and deletes it when there are
        # none. Writing them inline is what makes the plugin produce the
        # frontmatter field itself.
        f"#{CFG['tag_note']} #{CFG['tag_voice']}",
        "",
    ])
    note.write_text(text)
    return note


NO_SPEECH_MARKER = "_(no speech detected in this recording)_"


def echoes_initial_prompt(body: str) -> bool:
    """True when a near-silent file made whisper recite the initial prompt.

    Without VAD there is nothing to stop whisper from decoding a quiet file as
    the prompt it was given, which is how an 8 second silence became a note
    reciting the configured name, employer and exam names back as if she had
    said them. Compare word sets, since punctuation and case do not survive the
    trip.
    """
    prompt = str(CFG.get("initial_prompt") or "")
    words = [w for w in re.findall(r"[a-z0-9]+", body.lower()) if len(w) > 2]
    if not words or not prompt:
        return False
    pw = set(re.findall(r"[a-z0-9]+", prompt.lower()))
    return sum(1 for w in words if w in pw) / len(words) > 0.8


def transcribe_phase(redo_all: bool = False, limit: int | None = None,
                     confirmed: bool = False) -> int:
    conn = db()
    where = "audio_path IS NOT NULL" + ("" if redo_all else " AND transcribed_at IS NULL")
    rows = conn.execute(f"SELECT * FROM recordings WHERE {where} ORDER BY name").fetchall()
    if not rows:
        log("transcribe: nothing pending")
        return 0
    if limit:
        rows = rows[:limit]
    if redo_all and not confirmed:
        refuse_all("transcribe", len(rows), sum(r["duration"] or 0 for r in rows))
        return 0

    log(f"transcribe: {len(rows)} file(s) pending")
    from faster_whisper import WhisperModel

    started = time.time()
    model = WhisperModel(CFG["model"], device="cuda", compute_type=CFG["compute_type"], cpu_threads=2)
    log(f"transcribe: model loaded in {time.time() - started:.1f}s")
    pending_audio = sum(r["duration"] or 0 for r in rows)
    desktop_notify(f"Transcribing {plural(len(rows), 'recording')}",
                   f"{fmt_duration(pending_audio)} of audio on the GPU",
                   progress=True)

    done = 0
    for row in rows:
        path = Path(row["audio_path"])
        if not path.exists():
            conn.execute("UPDATE recordings SET error=? WHERE sha256=?", ("audio file missing", row["sha256"]))
            conn.commit()
            continue
        try:
            t0 = time.time()
            result = transcribe_file(model, path)
            body = paragraphs(result["segments"])
            gated = (result["speech_seconds"] < CFG["min_speech_seconds"]
                     or not body or echoes_initial_prompt(body))
            if gated:
                # No note for silence. The audio is kept in ~/Recordings and the
                # row records that the decision was made, which is what lets the
                # device copy be cleared; a note saying "no speech detected"
                # only clutters the transcript folder.
                conn.execute(
                    """UPDATE recordings SET transcribed_at=?, note_path=NULL, duration=?,
                       language=?, speech_seconds=?, transcript=?, no_speech_at=?, error=NULL
                       WHERE sha256=?""",
                    (datetime.now().strftime("%Y-%m-%d %H:%M"), result["duration"],
                     result["language"], result["speech_seconds"], NO_SPEECH_MARKER,
                     datetime.now().strftime("%Y-%m-%d %H:%M"), row["sha256"]),
                )
                conn.commit()
                done += 1
                log(f"transcribe: {row['name']} {fmt_duration(result['duration'])} "
                    f"speech={result['speech_seconds']:.1f}s -> no note, audio kept")
                continue
            note = write_raw_note(row, result, body)
            conn.execute(
                """UPDATE recordings SET transcribed_at=?, note_path=?, duration=?,
                   language=?, speech_seconds=?, transcript=?, no_speech_at=NULL, error=NULL
                   WHERE sha256=?""",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), str(note), result["duration"],
                 result["language"], result["speech_seconds"], body, row["sha256"]),
            )
            conn.commit()
            done += 1
            rtf = (time.time() - t0) / max(result["duration"], 0.001)
            log(f"transcribe: {row['name']} {fmt_duration(result['duration'])} "
                f"lang={result['language']} rtf={rtf:.3f} -> {note.name}")
        except Exception as exc:  # keep going; one bad file must not stop the batch
            conn.execute("UPDATE recordings SET error=? WHERE sha256=?", (str(exc)[:300], row["sha256"]))
            conn.commit()
            log(f"transcribe: FAILED {row['name']}: {exc}")
    log(f"transcribe: {done}/{len(rows)} done in {fmt_duration(time.time() - started)}")
    desktop_notify(f"Transcribed {plural(done, 'recording')}",
                   f"took {fmt_duration(time.time() - started)}", progress=True)
    return done


# --------------------------------------------------------------------------
# phase 3: summarize
# --------------------------------------------------------------------------

SUMMARY_SYSTEM = """You summarize voice-note transcripts for a personal Obsidian vault.
Rules:
- Be neutral, factual and concrete. Never moralize, never editorialize, never pathologize.
- Treat medical, psychiatric and relationship content plainly and only as it appears.
- Do not invent names, dates, amounts or decisions. If something is unclear, say so.
- Write in the third person about the speaker as "she" unless the transcript says otherwise.
- The recording is often a conversation between two or more people and the
  transcript does NOT label who is speaking. Recordings often open with a
  preamble saying who is talking or who the recording is for; when it does, use
  it to fix the attribution.
- Never assert who said or owns something when the transcript does not make it
  clear. Write "the speaker is not identified" rather than guessing. A confident
  wrong "she told the other person" is worse than an admitted gap, because
  downstream steps trust this summary.
- A second person telling her to do something is an action item for HER, not for
  them. "You need to ..." aimed at her is hers.
- No preamble, no closing offer of help, no meta commentary about being an AI."""

SUMMARY_INSTRUCTION = """Summarize the transcript below.

Output exactly this shape, nothing before it:

TITLE: <a specific human title for this recording, under 60 characters, no colon, no quotes, no slash, describes the actual content>
DESCRIPTION: <one line, under 80 characters, no colon-space, describes the recording>
KIND: <exactly one of: call, conversation, dictation, class, media>

## Summary
<2-4 sentences. If more than one person speaks, say who is who when the
recording makes it clear, and say the speakers are unidentified when it does not.>

## Key points
<ONLY what the summary above does not already say: exact dates, numbers,
names, providers, quotes worth keeping, and the facts a later reader would
need. Never restate a sentence the summary already made.>

## Decisions
<What was actually decided or agreed, by whom.>

## Action items
<Each one: the action, who has to do it, and the few words from the transcript
that show who. Use "unclear" for the owner when the transcript does not settle
it, and never assign an owner by guessing. Actions that someone asks of her are
hers.>

## Open questions
<An unresolved thing that matters>

Omit any section whose content would be empty. Never write "None stated",
"None", "N/A" or a placeholder bullet, and never keep a heading you have
nothing to put under. A five-section, two-section or one-section note is
correct when that is all there is.

Do not write a section for names or things mentioned: useful names belong in
Key points, everything else does not belong in the note.

Inline #tags belong at the very end of the body. Do not write frontmatter.

Transcript:
---
"""


def read_env_key(path: Path, key: str) -> str | None:
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def read_note_body(note: Path, attempts: int = 4) -> str:
    """Read a note's body, tolerating the vault plugin's rewrite window.

    `smart-autocorrect` rewrites a note seconds after it changes, and a read
    that lands inside that rewrite comes back empty. Retry briefly instead of
    believing an empty note. The state database's stored transcript is the
    preferred source; this is the fallback.
    """
    body = ""
    for _ in range(attempts):
        body = strip_frontmatter(note.read_text())
        if len(body.strip()) > 20:
            return body
        time.sleep(1.5)
    return body


def fireworks_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY") or read_env_key(
        Path(CFG["fireworks_env_file"]), "FIREWORKS_API_KEY")
    if not key:
        raise RuntimeError("no FIREWORKS_API_KEY found")
    return key


def fireworks_chat(messages: list[dict], max_tokens: int, label: str,
                   temperature: float = 0.2, timeout: int = 300,
                   reasoning_effort: str | None = None,
                   attempts: int = 3, model: str | None = None) -> str:
    """One chat completion against the configured router.

    The model behind this router reasons before it answers, and `max_tokens`
    covers the reasoning too. Two things follow, both measured on real
    recordings rather than guessed:

    - Cap the effort. `reasoning_effort: "low"` answered in 19s using ~3k
      reasoning tokens; medium and the unset default both spent the entire 8k
      budget thinking and returned empty content after 150s. The budget caps
      the reasoning, so raising it makes this worse, not better: 16k produced
      16k of reasoning and then needed 32k.
    - Retry the same request rather than escalating it. This router is not
      deterministic and the same prompt that answers in 19s one minute can blow
      the budget the next, so an empty reply is a coin flip, not a signal.

    `model` overrides the configured summarizer model. Callers doing structured
    extraction should pass one that does not reason its way past the budget.
    """
    key = fireworks_key()
    content = ""
    for attempt in range(1, max(1, attempts) + 1):
        payload = {
            "model": model or CFG["summary_model"],
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "messages": messages,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        req = urllib.request.Request(
            CFG["summary_base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        choice = data["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        usage = data.get("usage", {})
        log(f"{label}: api attempt {attempt}/{attempts} budget={max_tokens} "
            f"effort={reasoning_effort} finish={choice.get('finish_reason')} "
            f"tokens={usage.get('completion_tokens')} "
            f"reasoning={usage.get('completion_tokens_details', {}).get('reasoning_tokens')} "
            f"content={len(content)}c")
        if content:
            break
    return content


def call_summary_api(transcript: str) -> tuple[str, str, str, str]:
    """Ask the configured model for the header lines plus a markdown summary.

    The model behind this router is a reasoning model: it can spend the whole
    token budget on reasoning and return an empty `content`. So the budget
    starts generous and doubles once if the answer comes back empty.
    """
    key = os.environ.get("FIREWORKS_API_KEY") or read_env_key(
        Path(CFG["fireworks_env_file"]), "FIREWORKS_API_KEY")
    if not key:
        raise RuntimeError("no FIREWORKS_API_KEY found")
    if len(transcript) > 120_000:
        transcript = transcript[:120_000] + "\n\n[...transcript truncated...]"

    budget = int(CFG["summary_max_tokens"])
    content = ""
    for attempt in range(2):
        payload = {
            "model": CFG["summary_model"],
            "temperature": 0.2,
            "max_tokens": budget,
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": SUMMARY_INSTRUCTION + transcript + "\n---"},
            ],
        }
        req = urllib.request.Request(
            CFG["summary_base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        choice = data["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        usage = data.get("usage", {})
        log(f"summarize: api attempt {attempt + 1} budget={budget} "
            f"finish={choice.get('finish_reason')} tokens={usage.get('completion_tokens')} "
            f"reasoning={usage.get('completion_tokens_details', {}).get('reasoning_tokens')}")
        if content:
            break
        budget *= 2
    if not content:
        raise RuntimeError("model returned empty content at both token budgets")

    title = ""
    description = "Voice note transcript summary"
    kind = ""
    for line in content.splitlines()[:6]:
        if line.upper().startswith("TITLE:") and not title:
            title = line.split(":", 1)[1].strip().strip('"')
        elif line.upper().startswith("DESCRIPTION:") and description == "Voice note transcript summary":
            description = line.split(":", 1)[1].strip().strip('"')[:79] or description
        elif line.upper().startswith("KIND:") and not kind:
            kind = line.split(":", 1)[1].strip().lower()
    # Drop the header lines from the body.
    body_lines = []
    for line in content.splitlines():
        upper = line.upper()
        if (upper.startswith("TITLE:") or upper.startswith("DESCRIPTION:")
                or upper.startswith("KIND:")):
            continue
        body_lines.append(line)
    content = "\n".join(body_lines).lstrip("\n")
    return title, description, kind, content


def safe_title(title: str, fallback: str) -> str:
    """Turn a model-written title into a filesystem- and wikilink-safe name."""
    name = re.sub(r"\s+", " ", (title or "").strip())
    name = name.replace("•", "-").replace("—", "-")
    for char in '\\/:*?"<>|#[]^{}':
        name = name.replace(char, "")
    name = name.strip(" .-_")
    if len(name) > 70:
        cut = name[:70]
        name = cut[:cut.rfind(" ")].strip(" .-_") if " " in cut else cut
    return name or fallback


def write_summary_note(row, title: str, description: str, kind: str, body: str) -> Path:
    sum_dir = Path(CFG["vault"]) / CFG["transcripts_subdir"] / "summarized"
    sum_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(row["name"]).stem
    base = safe_title(title, stem)
    # The recording date stays as a prefix so content-named summaries still
    # sort in the order they were recorded. Set summary_name_prefix = "none"
    # in the config for bare content names.
    if CFG.get("summary_name_prefix", "date") == "date" and row["recorded_date"]:
        base = f"{row['recorded_date']} {base}"
    previous = row["summary_path"] if "summary_path" in row.keys() else None
    note = sum_dir / f"{base}.md"
    n = 2
    while note.exists() and str(note) != (previous or ""):
        note = sum_dir / f"{base} ({n}).md"
        n += 1
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    recorded = row["recorded_date"] or now[:10]
    text = "\n".join([
        frontmatter(
            title=note.stem,
            status="active",
            category=CFG["transcripts_subdir"],
            created=f"{recorded} 00:00",
            updated=now,
            description=description,
            generated=f"api/{CFG['summary_model'].rsplit('/', 1)[-1]}",
            kind=(kind or None),
            generated_at=now,
            recorded=recorded,
            duration=fmt_duration(row["duration"]),
            related=[f"[[{stem}]]"],
            sources=[f"Transcript note • [[{stem}]]"],
        ),
        "",
        f"# {note.stem}",
        "",
        f"> from [[{stem}]] · {fmt_duration(row['duration'])} · recorded {recorded}",
        "",
        body,
        "",
        f"#{CFG['tag_note']} #{CFG['tag_voice']} #{CFG['tag_summary']}",
        "",
    ])
    note.write_text(text)
    for stale in prune_stale_summaries(sum_dir, stem, note):
        log(f"summarize: removed stale summary {stale.name}")
    if previous and Path(previous) != note and Path(previous).exists():
        Path(previous).unlink()
        log(f"summarize: removed the previous name {Path(previous).name}")
    return note


def prune_stale_summaries(sum_dir: Path, stem: str, keep: Path) -> list[Path]:
    """Remove older summary notes for the same recording.

    A summary's name comes from the model, so re-running can land on slightly
    different wording and strand the previous file. Anything in the folder
    whose header links back to this recording's raw note, and that is not the
    file just written, is a stale copy of it.
    """
    marker = f"[[{stem}]]"
    removed: list[Path] = []
    for path in sorted(sum_dir.glob("*.md")):
        if path == keep or path.name == "index.md":
            continue
        try:
            head = path.read_text()[:1500]
        except OSError:
            continue
        if marker in head:
            try:
                path.unlink()
                removed.append(path)
            except OSError as exc:
                log(f"summarize: could not remove stale {path.name}: {exc}")
    return removed


def link_raw_note(raw_path: Path, summary_stem: str) -> None:
    """Point the raw transcript note at its summary.

    Without this the summary is only reachable by browsing the folder: nothing
    links to it, so LINT reports it as an orphan. Replaces an existing
    `related` block, because a re-summarize can rename the target.
    """
    if not raw_path.exists():
        return
    text = raw_path.read_text()
    if not text.startswith("---"):
        return
    end = text.find("\n---", 3)
    if end == -1:
        return
    head, tail = text[:end], text[end:]

    kept: list[str] = []
    lines = head.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("related:"):
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                i += 1
            continue
        kept.append(lines[i])
        i += 1
    kept += ["related:", f'  - "[[{summary_stem}]]"']
    raw_path.write_text("\n".join(kept) + tail)


def refuse_all(name: str, count: int, audio_seconds: float) -> None:
    """Say what a --all redo would cost, and change nothing.

    Every one of these phases makes a model call per recording, so a --all run
    over a full archive is a budget event, not a convenience. It has to be
    asked for twice, and --limit caps how many go through in one run.
    """
    print(f"{name} --all would reprocess {count} recording(s), "
          f"{fmt_duration(audio_seconds)} of audio, one model call each.")
    print("Nothing was changed.")
    print("Re-run with --yes to do it, or add --limit N to cap the run.")
    log(f"{name}: --all refused, confirmed=false, {count} recording(s) projected")


def summarize_phase(redo_all: bool = False, limit: int | None = None,
                    confirmed: bool = False) -> int:
    if not CFG["summarize"]:
        log("summarize: disabled")
        return 0
    conn = db()
    where = ("note_path IS NOT NULL AND speech_seconds IS NOT NULL "
             + ("" if redo_all else " AND summarized_at IS NULL"))
    rows = conn.execute(f"SELECT * FROM recordings WHERE {where} ORDER BY name").fetchall()
    if not rows:
        log("summarize: nothing pending")
        return 0
    if limit:
        rows = rows[:limit]
    if redo_all and not confirmed:
        refuse_all("summarize", len(rows), sum(r["duration"] or 0 for r in rows))
        return 0
    done = 0
    for row in rows:
        note = Path(row["note_path"]) if row["note_path"] else None
        if not note or not note.exists():
            continue
        if ((row["speech_seconds"] or 0) < CFG["min_speech_seconds"]
                or NO_SPEECH_MARKER in (row["transcript"] or "")):
            # There is nothing to summarize. Skipping on a --all too: it is how
            # twelve notes reading "no speech detected" collected six sections
            # of empty headings each, which is noise in the folder she browses.
            log(f"summarize: {row['name']} has no speech, skipped")
            conn.execute("UPDATE recordings SET summarized_at=? WHERE sha256=?",
                         (datetime.now().strftime("%Y-%m-%d %H:%M"), row["sha256"]))
            conn.commit()
            continue
        try:
            source = (row["transcript"] or "").strip() or read_note_body(note)
            title, description, kind, body = call_summary_api(source)
            out = write_summary_note(row, title, description, kind, body)
            link_raw_note(note, out.stem)
            conn.execute("UPDATE recordings SET summary_path=?, summarized_at=?, error=NULL WHERE sha256=?",
                         (str(out), datetime.now().strftime("%Y-%m-%d %H:%M"), row["sha256"]))
            conn.commit()
            done += 1
            log(f"summarize: {row['name']} -> {out.name}")
        except Exception as exc:
            log(f"summarize: FAILED {row['name']}: {exc}")
    log(f"summarize: {done}/{len(rows)} done")
    if done:
        desktop_notify(f"Summarized {plural(done, 'recording')}",
                       "notes written into Transcripts/summarized", progress=True)
    return done


# --------------------------------------------------------------------------
# phase 4: action items -> todo note
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# phase 4a: action items, one API call
# --------------------------------------------------------------------------

TODO_API_SYSTEM = """You maintain one person's task list from her voice recordings.

Rules:
- Be neutral, factual and concrete. Never moralize, never editorialize, never
  pathologize.
- Treat medical, psychiatric, relationship and sexual content plainly, exactly
  as it appears. It is her list, and filtering it is not your job.
- Never invent a task, a date, a name or an amount the recording does not contain.
- Output JSON only. No prose, no markdown fences, no commentary."""

TODO_API_INSTRUCTION = """Below is {subject}'s task list, then a recording of hers.

Add the action items from this recording that concern HER.

The recording is often a conversation and the transcript does not label who is
speaking. It may open with a preamble saying who is talking; if so, that settles
the attribution. Otherwise, a second person saying "you need to ..." is usually
talking to her, and those are her items, not theirs.

Include an item when:
- {subject} is the one who has to act, whoever is speaking.
- Someone in the recording asks her to do something.
- She says she is doing it too, even if she also assigns it to someone else.
- She has to chase someone who was supposed to act and might not.

Leave out what has nothing to do with her: what a scheduler, surgeon or provider
will do for her, and an errand two other people arrange between themselves.

Her list already has items. Never repeat one in any wording, including as a
subtask of an existing item. Match on the action, not the wording: "check in
periodically to make sure X isnt going crazy" and "make sure X isnt going crazy"
are the same item. If an existing item only needs a date or a detail from this
recording, put that in "updates" instead of adding a new item.

A chase is its own item. When she has to follow up with someone who was
supposed to act and might not, that is a new item with its own checkbox, not a
clause appended to an existing line. Measured without this rule, the chase was
emitted once in four runs, because the model treats it as a detail of the task
it belongs to.

Never add an item for something a provider said they would check, verify or
handle. "We'll verify that" and "I'll call the office" are his work, even when
insurance or a referral is the subject. Measured without this rule, a
CareOregon verification item appeared in four runs out of four, on a call where
the receptionist had already said he would verify it himself.

Keep each request as its own item. Two things she was asked to do are two items,
not one line joined with "and": merging them loses the ability to tick them off
separately.

Sections that exist: {sections}

Reply with JSON only, exactly this shape:

{{"items": [{{"section": "<one of the sections above>", "text": "<lowercase terse text, no trailing period>", "date": "<YYYY-MM-DD or null>", "priority": "<highest|high|null>", "subtasks": ["<optional subtask>"]}}], "updates": [{{"existing": "<text of the existing line, without the checkbox>", "append": "<what to add to that line>"}}]}}

Use "date" only when the recording gives a real date, and resolve relative dates
such as "Monday" against the recording date below. Use "priority" only for a
genuine deadline or urgency. If nothing concerns her reply
{{"items": [], "updates": []}}.

--- her task list ---
{todo}
--- end task list ---

Recording date: {recorded} ({weekday})
Today is: {today}

--- summary of the recording ---
{summary}
--- end summary ---

--- verbatim transcript ---
{transcript}
--- end transcript ---
"""

TODO_ITEM_RE = re.compile(r"^\s*- \[[ x]\]\s*(.+?)\s*$")
TODO_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*$")
# The Tasks plugin markers, with their optional date, stripped before comparing.
TODO_MARKER_RE = re.compile(r"\s*[📅⏫🔺✅]\s*\d{4}-\d{2}-\d{2}?|\s*[📅⏫🔺✅]")
# Where an item goes when the model names a section that does not exist.
TODO_CATCHALL = ("personal and errands", "general", "other", "medical and disability")


def item_key(text: str) -> str:
    """Comparison key for a task, so a reworded or re-marked line still matches.

    Strips the checkbox, the Tasks markers and their dates, the highlight
    markers and trailing punctuation, then lowercases and collapses whitespace.
    """
    t = TODO_ITEM_RE.match(text)
    if t:
        text = t.group(1)
    t = TODO_MARKER_RE.sub(" ", text)
    t = t.lower().replace("==", "")
    t = re.sub(r"[\s\u00a0]+", " ", t)
    t = re.sub(r"[\s!?.,;:]+$", "", t)
    return t.strip()


def todo_sections(lines: list[str]) -> list[str]:
    """The `## ` section names of the task area, in file order.

    The task area ends at the second H1, so a reference block's own `## `
    heading is never offered as somewhere to file a task.
    """
    stop = len(lines)
    h1s = [i for i, line in enumerate(lines) if line.startswith("# ")]
    if len(h1s) > 1:
        stop = h1s[1]
    return [line[3:].strip() for line in lines[:stop] if line.startswith("## ")]


def fallback_section(sections: list[str]) -> str:
    """A safe home for an item whose named section does not exist."""
    for name in TODO_CATCHALL:
        if name in sections:
            return name
    usable = [s for s in sections if s.strip().lower() != "done"]
    return usable[-1] if usable else sections[-1]


def parse_todo_json(content: str) -> dict:
    """Pull the JSON object out of a reply that may be fenced or chatty."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"no JSON object in the reply: {text[:200]!r}")


def render_todo_item(item: dict) -> list[str]:
    text = str(item.get("text") or "").strip().rstrip(".")
    if not text:
        return []
    date = str(item.get("date") or "").strip()
    priority = str(item.get("priority") or "").strip().lower()
    marker = {"highest": " 🔺", "high": " ⏫"}.get(priority, "")
    date_bit = f" 📅 {date}" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else ""
    lines = [f"- [ ] {text}{date_bit}{marker}"]
    for sub in item.get("subtasks") or []:
        sub = str(sub).strip().rstrip(".")
        if sub:
            lines.append(f"\t- [ ] {sub}")
    return lines


def apply_todo_json(todo_path: Path, payload: dict) -> list[str]:
    """Insert the model's items into the todo note, in place.

    Grouped by section and inserted bottom-up so earlier insertions cannot
    shift the line numbers of later ones. Returns the texts actually added, for
    the log and the digest.
    """
    lines = todo_path.read_text().splitlines()
    sections = todo_sections(lines)
    if not sections:
        raise RuntimeError(f"no '## ' sections found in {todo_path}")

    existing = {item_key(m.group(1))
                for m in (TODO_ITEM_RE.match(line) for line in lines) if m}

    grouped: dict[str, list[str]] = {}
    added: list[str] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section") or "").strip()
        if section not in sections:
            section = fallback_section(sections)
            log(f"todo: unknown section, filing under {section!r}")
        rendered = render_todo_item(item)
        if not rendered:
            continue
        key = item_key(str(item.get("text") or ""))
        if not key or key in existing:
            log(f"todo: already there, skipped: {rendered[0][:80]}")
            continue
        existing.add(key)
        grouped.setdefault(section, []).extend(rendered)
        added.append(item.get("text", "").strip())

    # "updates" appends a detail to an existing line rather than adding a
    # near-duplicate one. The append has to be idempotent: `todo --only` and
    # `todo --all` re-run this, and an unguarded append turned
    # "follow up on the sleep appointment" into a line that grew a new clause
    # on every pass.
    for update in payload.get("updates") or []:
        if not isinstance(update, dict):
            continue
        target = item_key(str(update.get("existing") or ""))
        extra = str(update.get("append") or "").strip()
        if not target or not extra:
            continue
        for i, line in enumerate(lines):
            m = TODO_ITEM_RE.match(line)
            if m and item_key(m.group(1)) == target:
                if item_key(extra) and item_key(extra) in item_key(m.group(1)):
                    log(f"todo: update already on the line, skipped: {extra[:60]}")
                    break
                lines[i] = f"{line.rstrip()} {extra}"
                log(f"todo: updated existing item with {extra[:60]}")
                break

    for section in sorted(grouped, key=lambda s: _section_start(lines, s), reverse=True):
        start = _section_start(lines, section)
        end = len(lines)
        for i in range(start + 1, len(lines)):
            if lines[i].startswith("## ") or lines[i].startswith("# "):
                end = i
                break
        insert_at = end
        while insert_at > start + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines[insert_at:insert_at] = grouped[section]

    text = "\n".join(lines)
    text = re.sub(r"^(updated:).*$", f"\\1 {datetime.now():%Y-%m-%d %H:%M}",
                  text, count=1, flags=re.M)
    if not text.endswith("\n"):
        text += "\n"
    todo_path.write_text(text)
    return added


def _section_start(lines: list[str], section: str) -> int:
    for i, line in enumerate(lines):
        if line.strip() == f"## {section}":
            return i
    return 0


def call_todo_api(row, summary: Path) -> tuple[list[str], dict]:
    """One API call in place of the agent run. Returns (added, payload)."""
    todo_path = todo_note_path()
    todo_text = todo_path.read_text()
    lines = todo_text.splitlines()
    sections = todo_sections(lines)

    recorded = row["recorded_date"] or "unknown"
    try:
        weekday = datetime.strptime(recorded, "%Y-%m-%d").strftime("%A")
    except ValueError:
        weekday = "unknown"
    try:
        summary_text = strip_frontmatter(summary.read_text()).strip()
    except OSError:
        summary_text = ""
    transcript = (row["transcript"] or "").strip()
    if len(transcript) > TODO_TRANSCRIPT_CHARS:
        transcript = (transcript[:TODO_TRANSCRIPT_CHARS]
                      + "\n[... transcript truncated for length ...]")

    prompt = TODO_API_INSTRUCTION.format(
        todo=todo_text,
        sections=", ".join(f'"{s}"' for s in sections),
        summary=summary_text or "(the summary note could not be read)",
        transcript=transcript or "(no transcript text was captured)",
        recorded=recorded,
        weekday=weekday,
        today=datetime.now().strftime("%Y-%m-%d %A"),
        subject=CFG["subject"],
    )
    content = fireworks_chat(
        [{"role": "system", "content": TODO_API_SYSTEM},
         {"role": "user", "content": prompt}],
        max_tokens=int(CFG["todo_max_tokens"]),
        label="todo",
        reasoning_effort=str(CFG["todo_reasoning"]) or None,
        model=str(CFG["todo_model"]),
    )
    if not content:
        raise RuntimeError("model returned empty content")
    payload = parse_todo_json(content)
    added = apply_todo_json(todo_path, payload)
    return added, payload


# --------------------------------------------------------------------------
# phase 4b: action items, headless agent (fallback)
# --------------------------------------------------------------------------

# Handed to a headless Hermes run, one recording at a time. It gets the file
# tools and the vault's own AGENTS.md (by running with the vault as cwd), so it
# can read the todo note, judge what concerns her, and write in her style.
#
# The summary is embedded rather than linked, and the prompt fences off the rest
# of the vault and the machine. Left open, the agent spent 20 API calls and 4
# minutes reading past transcripts, this pipeline's own source, the skill file
# describing it, and running git and LINT, none of which is its job.
TODO_PROMPT = """A recording from {subject}'s Olympus voice recorder was just \
transcribed and summarized. Everything you need is in this prompt and in her \
todo list.

Recorded: {recorded} ({weekday})
Today is: {today}
Her todo list: {todo_note}

--- summary of the recording ---
{summary}
--- end summary ---

--- verbatim transcript, for checking the summary ---
{transcript}
--- end transcript ---

Add the action items from this recording that concern HER.

The transcript merges whoever was in the room, so "you" and "me" are ambiguous
and the summary's owner labels are only a guess. The transcript is the ground
truth. Read it before deciding, and trust it over the summary when they differ.

Include an item when any of these is true:
- {subject} is the one who has to act, whoever is speaking.
- Someone in the recording asks her to do something. The transcript has no
  speaker labels, so a second person saying "you need to ..." is usually talking
  to her, and those are her items, not theirs.
- She says she is doing it too, even if she also assigns it to someone else.
- She has to chase someone who was supposed to act and might not.

Leave out what has nothing to do with her: something a third party will do for
her (a scheduler who will call, a surgeon who will check a record), or an errand
two other people arrange between themselves. She wants what she has to do, not a
transcript of everything said.

If the recording opens with a preamble naming who is speaking or who it is for,
that settles the attribution: follow it.

How:
1. Read her todo list. That is the only file you need. Open a note it links to
   only if this recording clearly bears on that note.
2. Add each new item under the section it belongs to: "medical and disability",
   "money", "school and work", "personal and errands", or "security".
3. Match the file's existing style exactly:
   - "- [ ] lowercase terse text" with no trailing period
   - a tab-indented "- [ ]" line for a subtask
   - when an item has a date, end the line with "📅 YYYY-MM-DD"; never write the
     date as prose, because her Tasks queries read that field
   - "🔺" or "⏫" only for a genuine priority; "==text==" around a phone number
   - resolve relative dates such as "Monday" against the recording date above
4. Never add an item that is already in the file in any wording, including as a
   subtask of an existing item. If an existing item needs a date or a detail
   from this recording, add that to the existing item rather than a new line.
5. Set the "updated:" timestamp in the todo file's frontmatter to now.
6. Change nothing else. Do not read the summary file, past transcripts, the
   recorder pipeline, or its skills. Do not edit or create any other note. Do
   not run git, and do not run the vault maintenance scripts, the pipeline runs
   those itself after you. Do not send messages.

Then reply with one line per item you added, each starting with "+ " and naming
the section, or reply with exactly "none" if nothing in this recording concerns
her. No preamble, no explanation."""

# Long class recordings produce huge transcripts; bound what goes in the prompt.
TODO_TRANSCRIPT_CHARS = 40000


def todo_note_path() -> Path:
    return Path(str(CFG["todo_note"])).expanduser()


def build_todo_prompt(row, summary: Path) -> str:
    recorded = row["recorded_date"] or "unknown"
    try:
        weekday = datetime.strptime(recorded, "%Y-%m-%d").strftime("%A")
    except ValueError:
        weekday = "unknown"
    try:
        text = strip_frontmatter(summary.read_text()).strip()
    except OSError:
        text = ""
    transcript = (row["transcript"] or "").strip()
    if len(transcript) > TODO_TRANSCRIPT_CHARS:
        transcript = (transcript[:TODO_TRANSCRIPT_CHARS]
                      + "\n[... transcript truncated for length ...]")
    return TODO_PROMPT.format(
        summary=text or "(the summary note could not be read)",
        transcript=transcript or "(no transcript text was captured)",
        recorded=recorded,
        weekday=weekday,
        today=datetime.now().strftime("%Y-%m-%d %A"),
        todo_note=todo_note_path(),
        subject=CFG["subject"],
    )


def run_todo_agent(row, summary: Path) -> tuple[int, str]:
    """One headless Hermes run: read the summary, edit the todo note."""
    STAGING.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".todo-prompt.txt",
                                     dir=STAGING, delete=False) as fh:
        fh.write(build_todo_prompt(row, summary))
        prompt_file = Path(fh.name)
    # The prompt goes in by file: nothing shell-interpreted, so quotes and
    # backticks in a transcript cannot break the call.
    cmd = [
        str(CFG["todo_agent"]), "chat", "-Q", "--oneshot",
        "--source", "recorder-todo",
        "-t", str(CFG["todo_toolsets"]),
        "-s", str(CFG["todo_skills"]),
        "--reasoning", str(CFG["todo_reasoning"]),
        "--max-turns", str(CFG["todo_max_turns"]),
        "--run-budget", str(CFG["todo_run_budget"]),
        "--in", str(Path(CFG["vault"])),
        "--query-file", str(prompt_file),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=CFG["todo_timeout"])
    except subprocess.TimeoutExpired:
        log(f"todo: agent timed out after {CFG['todo_timeout']}s")
        prompt_file.unlink(missing_ok=True)
        return 1, f"timed out after {CFG['todo_timeout']}s"
    finally:
        prompt_file.unlink(missing_ok=True)
    out = (res.stdout or "").strip()
    if res.returncode != 0:
        return res.returncode, ((res.stderr or "").strip() or out)[-300:]
    return 0, out


def parse_todo_additions(output: str) -> list[str]:
    items = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("+ "):
            items.append(line[2:].strip())
    return items


def run_todo_extraction(row, summary: Path) -> tuple[bool, list[str], str]:
    """Extract the items for one recording. Returns (ok, item texts, error).

    One API call by default, with the headless agent behind it as a fallback.
    todo_mode = "agent" goes straight to the agent, for the case where the
    extraction needs to go and look something up, which a single call cannot do.
    """
    if str(CFG["todo_mode"]).lower() == "agent":
        rc, out = run_todo_agent(row, summary)
        if rc != 0:
            return False, [], out
        return True, parse_todo_additions(out), ""

    api_error = ""
    try:
        added, _payload = call_todo_api(row, summary)
        return True, added, ""
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {exc}"[:300]
        log(f"todo: api path failed: {api_error}")

    if not CFG["todo_fallback_agent"]:
        return False, [], api_error

    log(f"todo: falling back to the agent for {row['name']}")
    rc, out = run_todo_agent(row, summary)
    if rc != 0:
        return False, [], f"api: {api_error} | agent: {out}"[:300]
    return True, parse_todo_additions(out), ""


def todo_phase(redo_all: bool = False, only: str | None = None,
               limit: int | None = None, confirmed: bool = False) -> list[str]:
    if not CFG["todo_extract"]:
        log("todo: disabled")
        return []
    conn = db()
    rows = conn.execute(
        "SELECT * FROM recordings WHERE summary_path IS NOT NULL ORDER BY name"
    ).fetchall()
    pending = []
    for row in rows:
        if only and Path(row["name"]).stem != only:
            continue
        if row["todo_at"] and not redo_all:
            continue
        summary = Path(row["summary_path"])
        if summary.exists():
            pending.append((row, summary))
    if limit:
        pending = pending[:limit]
    if not pending:
        log("todo: nothing pending")
        return []
    if redo_all and not confirmed:
        refuse_all("todo", len(pending), sum(r["duration"] or 0 for r, _ in pending))
        return []

    added: list[str] = []
    for row, summary in pending:
        log(f"todo: {row['name']} -> {summary.name}")
        # The agent run is the slow step and can take a minute. Without this
        # the desktop goes quiet after "Summarized" and the pass looks hung.
        desktop_notify("Working on your todo list", summary.stem[:90], progress=True)
        ok, items, err = run_todo_extraction(row, summary)
        if not ok:
            log(f"todo: FAILED {row['name']}: {err}")
            conn.execute("UPDATE recordings SET error=? WHERE sha256=?",
                         (f"todo: {err}"[:500], row["sha256"]))
            conn.commit()
            continue
        conn.execute("UPDATE recordings SET todo_at=?, todo_added=?, error=NULL WHERE sha256=?",
                     (datetime.now().strftime("%Y-%m-%d %H:%M"),
                      "\n".join(items), row["sha256"]))
        conn.commit()
        if items:
            for item in items:
                log(f"todo: + {item}")
            added.extend(items)
        else:
            log(f"todo: {row['name']} had nothing that concerns her")
    if added:
        desktop_notify(f"Added {plural(len(added), 'action item')} to your todo list",
                       todo_note_path().name, progress=True)
    else:
        desktop_notify("Nothing for your todo list",
                       "no action items for you in this recording", progress=True)
    return added


# --------------------------------------------------------------------------
# phase 5: vault maintenance
# --------------------------------------------------------------------------

def vault_maintenance() -> None:
    vault = Path(CFG["vault"])
    for script in ("MANIFEST_GENERATOR.sh", "LINT.sh"):
        path = vault / "SCRIPTS" / script
        if not path.exists():
            continue
        res = subprocess.run([str(path)], cwd=str(vault), capture_output=True, text=True)
        tail = (res.stdout or res.stderr).strip().splitlines()
        log(f"vault: {script} exit={res.returncode} {tail[-1] if tail else ''}")
        if script == "LINT.sh" and res.returncode != 0:
            for line in tail:
                if line.startswith("[HARD]"):
                    log(f"vault: {line}")


# --------------------------------------------------------------------------
# phase 6: delete from device
# --------------------------------------------------------------------------

def delete_phase() -> int:
    if not CFG["delete_from_device"]:
        log("delete: disabled")
        return 0
    conn = db()
    # A no-speech recording has no note by design: the row itself carries the
    # decision, and the audio copy plus its hash is the thing that must exist
    # before the card is touched.
    rows = conn.execute(
        """SELECT * FROM recordings
           WHERE deleted_at IS NULL AND audio_path IS NOT NULL
             AND (note_path IS NOT NULL OR no_speech_at IS NOT NULL)
           ORDER BY name"""
    ).fetchall()
    if not rows:
        log("delete: nothing to clear")
        return 0
    dev = device_node()
    if not dev:
        log(f"delete: {len(rows)} file(s) waiting, recorder not attached")
        return 0
    mp = mount_device(dev)
    if not mp:
        log("delete: could not mount the recorder")
        return 0
    deleted = 0
    for row in rows:
        copy = Path(row["audio_path"])
        if not copy.exists() or sha256_of(copy) != row["sha256"]:
            log(f"delete: SKIP {row['name']}, local copy missing or changed")
            continue
        if row["note_path"] and not Path(row["note_path"]).exists():
            log(f"delete: SKIP {row['name']}, transcript note missing")
            continue
        if not row["note_path"] and not row["no_speech_at"]:
            log(f"delete: SKIP {row['name']}, neither a note nor a no-speech marker")
            continue
        target = mp / row["rel_path"]
        if not target.exists():
            conn.execute("UPDATE recordings SET deleted_at=? WHERE sha256=?",
                         (datetime.now().strftime("%Y-%m-%d %H:%M"), row["sha256"]))
            conn.commit()
            continue
        try:
            target.unlink()
            deleted += 1
            conn.execute("UPDATE recordings SET deleted_at=? WHERE sha256=?",
                         (datetime.now().strftime("%Y-%m-%d %H:%M"), row["sha256"]))
            conn.commit()
            log(f"delete: removed {row['rel_path']} from the recorder")
        except OSError as exc:
            log(f"delete: FAILED {row['rel_path']}: {exc}")
    subprocess.run(["/usr/bin/sync"], capture_output=True)
    if not unmount_device(dev):
        log("delete: recorder left mounted")
    else:
        log("delete: recorder unmounted")
    return deleted


# --------------------------------------------------------------------------
# notify
# --------------------------------------------------------------------------

def wrap(text: str, width: int = 30) -> str:
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
        else:
            out.extend(textwrap.wrap(para, width=width, break_long_words=False) or [""])
    return "\n".join(out)


def notify(message: str) -> None:
    if not CFG["notify"]:
        return
    try:
        res = subprocess.run(
            [str(CFG["hermes_bin"]), "send", "-t", CFG["notify_target"], wrap(message)],
            capture_output=True, text=True, timeout=90,
        )
        log(f"notify: exit={res.returncode} {(res.stderr or res.stdout).strip()[:200]}")
    except Exception as exc:
        log(f"notify: failed {exc}")


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

# Section names a reply may prefix an item with. Kept in step with the todo
# note: a name missing here means its prefix survives into the phone message.
TODO_SECTIONS = ("other task files", "medical and disability", "money",
                 "school and work", "personal and errands", "security",
                 "general", "done")


def bullet_lines(text: str, emoji: str = "🔹", width: int = 30) -> list[str]:
    """One bullet per line, every wrapped line carrying the emoji.

    The phone column is 30 wide, and a bullet whose text wraps loses its shape
    and stops reading as a list. `wrap()` cannot fix that: it would break the
    continuation onto an unmarked line, so the wrap happens here with the emoji
    repeated on each line instead.
    """
    avail = max(10, width - len(emoji) - 2)
    parts = textwrap.wrap(text, width=avail, break_long_words=False) or [""]
    return [f"{emoji} {p}" for p in parts]


def digest_item(item: str) -> str:
    """Drop the "section: " prefix the agent reports, since the message already
    groups the items under one header and the phone column is 30 wide. The
    separator varies run to run, so accept the forms seen in practice."""
    for sep in (": ", " — ", " - ", " – "):
        head, found, rest = item.partition(sep)
        if found and head.strip().lower() in TODO_SECTIONS:
            return rest.strip()
    return item


def build_digest(new_rows, todo_added: list[str], deleted: int) -> str:
    """The SimpleX message for one ingest pass. Kept separate so it can be
    rendered without sending, and because the phone column is only 30 wide."""
    count = len(new_rows)
    if count:
        lines = [
            "🎙️ RECORDER",
            f"{count} new recording{'s' if count != 1 else ''} pulled off and transcribed.",
        ]
        for row in new_rows[:6]:
            lines.append(f"🔹 {Path(row['name']).stem} {fmt_duration(row['duration'])}")
        if count > 6:
            lines.append(f"🔹 +{count - 6} more")
        if deleted:
            lines.append("✅ cleared from the device")
        elif not CFG["delete_from_device"]:
            lines.append("📄 files left on the device")
        else:
            lines.append("⚠️ device not cleared, will retry")
    else:
        lines = ["🎙️ RECORDER", "follow-up pass, no new recordings."]
    if todo_added:
        lines.append("📌 TODO LIST")
        for item in todo_added[:6]:
            lines.extend(bullet_lines(digest_item(item)))
        if len(todo_added) > 6:
            lines.append(f"🔹 +{len(todo_added) - 6} more")
    return "\n".join(lines)


def ingest_pass() -> None:
    lock = LOCK_FILE.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("ingest: another run holds the lock, skipping")
        return

    started = time.time()
    log("=== ingest pass start ===")
    new_rows = copy_phase()
    transcribed = transcribe_phase()
    summarized = summarize_phase()
    todo_added = todo_phase()
    if transcribed or summarized:
        vault_maintenance()
    deleted = delete_phase()
    log(f"=== ingest pass done in {fmt_duration(time.time() - started)}: "
        f"{len(new_rows)} copied, {transcribed} transcribed, {summarized} summarized, "
        f"{len(todo_added)} todo items, {deleted} cleared ===")

    if new_rows or todo_added:
        notify(build_digest(new_rows, todo_added, deleted))

    # The unplug signal. Read the pending count from the database rather than
    # trusting this pass: a failed mount or a delete that skipped a file leaves
    # recordings on the card, and "safe to unplug" would then be a lie.
    pending_clear = db().execute(
        "SELECT COUNT(*) FROM recordings WHERE deleted_at IS NULL AND note_path IS NOT NULL"
    ).fetchone()[0]
    if device_node():
        if pending_clear:
            desktop_notify("Recorder not clear",
                           f"{plural(pending_clear, 'recording')} still on the card, will retry",
                           urgency="critical", icon="media-eject")
        else:
            desktop_notify("Safe to unplug", "card is clear", urgency="critical",
                           icon="media-eject")
    elif new_rows or todo_added or deleted:
        desktop_notify("Done",
                       f"{plural(pending_clear, 'recording')} will clear next time you plug it in"
                       if pending_clear else "recorder not attached")


def watch_loop() -> None:
    log(f"watch: starting, polling every {CFG['poll_seconds']}s for {CFG['device_label']}")
    was_present = False
    while True:
        try:
            dev = device_node()
            present = dev is not None
            if present and not was_present:
                log(f"watch: {CFG['device_label']} attached at {dev}")
                ingest_pass()
            elif was_present and not present:
                log(f"watch: {CFG['device_label']} detached")
            was_present = present
            if present:
                time.sleep(5)
            else:
                time.sleep(CFG["poll_seconds"])
        except Exception as exc:
            log(f"watch: error {exc!r}, continuing")
            time.sleep(10)


def cmd_status() -> None:
    conn = db()
    rows = conn.execute("SELECT * FROM recordings").fetchall()
    print(f"{len(rows)} recordings known")
    stages = {
        "copied": sum(1 for r in rows if r["copied_at"]),
        "transcribed": sum(1 for r in rows if r["transcribed_at"]),
        "summarized": sum(1 for r in rows if r["summarized_at"]),
        "action items pulled": sum(1 for r in rows if r["todo_at"]),
        "no speech, no note": sum(1 for r in rows if r["no_speech_at"]),
        "cleared from device": sum(1 for r in rows if r["deleted_at"]),
        "errors": sum(1 for r in rows if r["error"]),
    }
    for key, value in stages.items():
        print(f"  {key}: {value}")
    total = sum(r["duration"] or 0 for r in rows)
    print(f"  audio on disk: {fmt_duration(total)}")
    print(f"  todo items added: {sum(len((r['todo_added'] or '').splitlines()) for r in rows)}")
    print(f"device: {device_node() or 'not attached'}")
    print(f"audio dir: {CFG['audio_dir']}")
    print(f"vault: {Path(CFG['vault']) / CFG['transcripts_subdir']}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="recorder", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("watch")
    sub.add_parser("ingest")
    sub.add_parser("copy")
    sub.add_parser("delete")
    sub.add_parser("status")
    def add_redo_flags(parser):
        parser.add_argument("--all", action="store_true",
                            help="reprocess everything, including work already done")
        parser.add_argument("--limit", type=int, metavar="N",
                            help="stop after N recordings")
        parser.add_argument("--yes", action="store_true",
                            help="required with --all: confirms the model spend")

    p_t = sub.add_parser("transcribe")
    add_redo_flags(p_t)
    p_s = sub.add_parser("summarize")
    add_redo_flags(p_s)
    p_todo = sub.add_parser("todo")
    add_redo_flags(p_todo)
    p_todo.add_argument("--only", metavar="STEM",
                        help="one recording stem, e.g. 261016_0054")
    args = parser.parse_args()

    ensure_cuda_env()
    BASE.mkdir(parents=True, exist_ok=True)

    if args.cmd == "watch":
        watch_loop()
    elif args.cmd == "ingest":
        ingest_pass()
    elif args.cmd == "copy":
        copy_phase()
    elif args.cmd == "transcribe":
        transcribe_phase(redo_all=args.all, limit=args.limit, confirmed=args.yes)
    elif args.cmd == "summarize":
        summarize_phase(redo_all=args.all, limit=args.limit, confirmed=args.yes)
    elif args.cmd == "todo":
        todo_phase(redo_all=args.all, only=args.only, limit=args.limit,
                   confirmed=args.yes)
    elif args.cmd == "delete":
        delete_phase()
    elif args.cmd == "status":
        cmd_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
