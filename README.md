# recorder

Plug a digital voice recorder into a Linux box and get back transcripts, summaries and a todo list without touching anything.

I record voice notes on a cheap Olympus WS-852. The card fills up, I plug it in, and this does the rest: copies the audio off, unmounts the card so the recorder can go back in my pocket, transcribes on the GPU, writes a verbatim note and a content-named summary into an Obsidian vault, pulls the action items into my task list, clears the audio off the recorder, and tells me when it is safe to unplug.

It was built for one machine and one person. It is published because the design notes were worth writing down, and because most of the interesting problems here are not specific to voice recorders: unlabeled speakers, a reasoning model that will spend your whole token budget thinking, and an unattended pipeline that has to be safe to kill at any moment.

## The flow

```
plug in
  │
  ├─ copy ──────── hash, stage, atomic move into ~/Recordings, then unmount
  │
  ├─ transcribe ── faster-whisper large-v3-turbo on CUDA
  │
  ├─ summarize ─── one note per recording, named from its content
  │
  ├─ action items  one structured API call into a markdown task list
  │
  ├─ maintain ──── vault lint + manifest
  │
  ├─ delete ────── gated on local hash + note existing, then clears the card
  │
  └─ notify ────── desktop popups per phase, a message, and "safe to unplug"
```

Every phase stamps its own column in a SQLite database keyed by the sha256 of each recording, so re-plugging only redoes unfinished work, and a crash or an OOM kill costs at most the file in flight. Nothing is ever deleted from the recorder until its transcript is on disk and the local copy still hashes to what was recorded.

## Requirements

- **Linux with systemd user units.** The watcher is a user service, and mounting is done with `udisks2` so it never needs root.
- **An NVIDIA GPU with the CUDA runtime available as pip wheels.** No CUDA toolkit is needed: `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` from PyPI provide the libraries `ctranslate2` loads at runtime. This was tested on a 4GB laptop GPU.
- **Python 3.13.** `faster-whisper` and `ctranslate2` are the only hard dependencies.
- **A Fireworks API key** for the summarize and action-item phases, in an env file or the environment as `FIREWORKS_API_KEY`. Both phases can be turned off.
- **An Obsidian vault**, or at least a directory tree to write markdown into. The vault layout is configurable.
- Optional: **Hermes**, used for the fallback path in the action-item phase and for sending the summary message to a chat platform. Everything else works without it.

## Install

```sh
# 1. Put the code somewhere permanent
mkdir -p ~/.local/share/recorder
cp recorder.py ~/.local/share/recorder/

# 2. A venv with the transcriber and the CUDA runtime
python3.13 -m venv ~/.local/share/recorder/venv
~/.local/share/recorder/venv/bin/pip install \
    faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12

# 3. Pre-download the model, about 1.6 GB
~/.local/share/recorder/venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Systran/faster-whisper-large-v3-turbo',
    local_dir='$HOME/.local/share/recorder/models/whisper-large-v3-turbo')"

# 4. The wrapper, which sets LD_LIBRARY_PATH before python starts
cp examples/recorder ~/.local/bin/recorder
chmod +x ~/.local/bin/recorder

# 5. The watcher
mkdir -p ~/.config/systemd/user
cp examples/recorder-watch.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now recorder-watch.service
```

Step 4 matters more than it looks. `LD_LIBRARY_PATH` has to point at the pip-installed CUDA libraries **before the process starts**, so setting it from inside python does nothing. The wrapper sets it and execs, and `ensure_cuda_env()` re-execs if it ever finds the variable missing.

## Configure

All settings live in `DEFAULTS` at the top of `recorder.py`. `~/.config/recorder/config.toml` overrides any of them, and `config.example.toml` in this repo lists the lot commented out. The ones worth knowing:

| Setting | Default | What it does |
|---|---|---|
| `device_label` | `WS-852` | The filesystem label the watcher polls for |
| `audio_dir` | `~/Recordings` | Flat archive of the pulled audio |
| `vault` | `~/Documents/OBSIDIAN` | Where the markdown goes |
| `model` | `.../models/whisper-large-v3-turbo` | Path to the CTranslate2 model |
| `initial_prompt` | generic | Names and jargon the transcriber should expect |
| `subject` | `the user` | Whose action items the last phase is looking for |
| `min_speech_seconds` | `1.0` | Below this a recording is marked as having no speech |
| `delete_from_device` | `true` | Clear recordings off the card once transcribed |
| `todo_mode` | `api` | `api` for one API call, `agent` for a headless agent |
| `todo_model` | `gpt-oss-120b` | Must be a model that does not reason past the budget |
| `notify` | `true` | Send a message after each pass |
| `desktop_notify` | `true` | Desktop popups per phase |

`initial_prompt` and `subject` exist so the code itself carries no personal details. Set both to your own.

## Usage

```sh
recorder watch                  # long-running watcher, what the service runs
recorder ingest                 # one full pass, now
recorder copy                   # one phase at a time
recorder transcribe [--all]
recorder summarize [--all]
recorder todo [--all] [--only STEM]
recorder delete
recorder status
```

`--all` redoes everything rather than only the pending work, which is what you want after changing a prompt. `recorder todo --only 261016_0054` reprocesses a single recording by its filename stem.

The single-phase commands deliberately skip the lock that `ingest` takes, so do not run `summarize` by hand while the watcher is mid-pass: they write the same notes.

## How it works

**Copy.** Mounts the card with `udisksctl`, walks the recording folders, and hashes every file. A hash already in the database means the file was ingested before, so it is skipped, which is what makes plugging the same card in twice harmless. New files are copied into a staging directory with a `.part` suffix, re-hashed there, and only then moved into place with an atomic rename. A hash that disagrees on re-read means the file is discarded rather than trusted. Recording dates come from the `YYMMDD_NNNN` filename because vfat mtimes are unreliable.

The card is unmounted here, before any GPU work, so the recorder is free to unplug while transcription runs.

**Transcribe.** Loads `faster-whisper` on CUDA, float16, beam 5, VAD on. The transcript is written both to a verbatim note in the vault and into the database. The database copy is what the next phase reads, not the note, because an Obsidian autocorrect plugin rewrites notes underneath the pipeline and would race it.

Each file is wrapped in its own error handler. One corrupt file records an error and the batch continues.

**Summarize.** Sends the transcript to a Fireworks model that returns a title, a description and a body. The note is named from that title with the recording date prefixed, so the folder still sorts chronologically and a summary is findable by what it says rather than by a recorder stem. Any previous name for the same recording is pruned first, so a re-run does not strand orphans, and the raw note gets a `related:` back-link to the summary.

**Action items.** The default path is a single API call that returns JSON, and the pipeline does the inserting: filing each item into the right section, skipping anything already on the list, appending dates and details to existing lines rather than adding near-duplicates, and bumping the file's `updated` field. A headless agent path exists behind `todo_mode = "agent"` for when the extraction needs to look something up.

**Maintenance.** Runs the vault's lint and manifest generator, but only when something was actually written.

**Delete.** Runs last, and only for recordings that have a transcript note. Two gates per file: the local copy must still exist and still hash to the recorded sha256, and the note must be on disk. Only then is the file removed from the card.

**Report.** One message to a chat platform, plus desktop notifications per phase. The "safe to unplug" notification reads the pending count from the database rather than from the pass that just ran, because a failed mount or a skipped delete leaves files on the card and asking the pass would make the all-clear a lie.

## Design notes

The parts that took more than one attempt.

**`max_tokens` covers reasoning, and that is a trap.** The model I first used for the action-item phase reasons before it answers, and the token budget covers the thinking. On a real recording it spent all 8000 tokens reasoning and returned empty content, three times in a row, 150 seconds, nothing. Two lessons, both measured rather than assumed:

- Cap the reasoning effort. At `medium`, and at the default, it blew the budget and returned nothing. At `low` it answered in 19 seconds using 3000 reasoning tokens.
- **Do not escalate the budget on a retry.** It makes things worse: 16k produced 16k of reasoning and then needed 32k. Retry the same request.
- The failure is bursty, not a coin flip. Three immediate retries all failed inside one minute while the identical prompt succeeded two minutes later. Retries alone were never going to fix it.

So I benchmarked six models on the same prompt and the same recording. `gpt-oss-120b` with low reasoning effort answered in **3 seconds** with 8 items. The reasoning router took 50 seconds for 2 items, or returned nothing. `nemotron-lightning`, `glm-5p3`, `glm-5p3-fast` and `deepseek-v4p1-flash` all returned empty content. The headless agent took 58 to 246 seconds for 1 to 4 items.

**Whisper does not label speakers, and a wrong label is worse than no label.** A recording of a conversation arrives as one undifferentiated stream with `you` and `me` unattached. The summarizer initially read "you need to follow up on your appointment" and, having no labels to work with, asserted as fact that the speaker was talking to someone else. The action-item phase filters on who owns each item, so it trusted that label and silently dropped nine of ten items from one recording.

The fix was to stop the summarizer asserting anything it cannot see. It now names the speaker when the recording says so, writes `unclear` when it cannot tell, and quotes the words that establish each owner, so a wrong call is visible instead of invisible. The extraction phase reads the verbatim transcript rather than the summary, and treats a second person saying "you need to" as talking to the subject. None of this recovers information the audio never had, which is why the real answer is a preamble naming who is speaking, and why speaker diarization is the honest upgrade path.

**Delete last, and gate it twice.** The ordering is load-bearing. Unmount before transcribing so the recorder is free. Delete only after the note exists and the local copy re-hashes correctly, so a transcription failure can never cost the original recording. A pipeline that clears a card on a successful copy is one bug away from losing data permanently.

**State in SQLite, keyed by content.** Not by filename, not by mtime. The same recording copied to a different folder is still the same recording, and vfat timestamps are not trustworthy. Every phase stamps its own column, which gives resumability, a natural work queue, and an audit trail of what happened to each file.

**Text meant for a phone needs its own formatting.** The summary message goes to a chat client about 33 characters wide with no markdown rendering. A bullet whose text wraps loses its shape and stops reading as a list, so list items are wrapped in code with the bullet repeated on every line rather than left to a general-purpose text wrapper. A prose sentence wrapping is fine; a list item wrapping is not.

## Cost and performance

Measured on a 4GB laptop GPU and 7.5GB of RAM:

| | |
|---|---|
| Transcription | RTF 0.015 to 0.05, so about 20 to 60x faster than realtime |
| 3h08m of backlog | 4m46s |
| Model peak memory | about 1.9GB |
| Summarize, per recording | a few seconds |
| Action items, per recording | 3 to 6 seconds |

The action-item phase is the only part that costs meaningfully per recording. Transcription is free once you own the GPU.

## Limits

- **No speaker diarization.** Two people on one recording are one stream of text. See the design note above.
- **One recorder, one person.** The watcher polls for a single filesystem label, and the action-item prompts assume one subject.
- **The action-item phase assumes a markdown task list** with `## ` sections and `- [ ]` checkboxes. The section names in the prompt and in `TODO_CATCHALL` are examples from my own list and are worth changing to match yours.
- **Filenames drive the recording date.** `YYMMDD_NNNN` is the Olympus convention. A different recorder will need a different parser, and until then dates fall back to unknown.
- **Silence is handled bluntly.** A recording with under a second of detected speech gets a note saying so and is skipped by the summarizer.
- **`udisksctl` needs a session bus.** This runs from a systemd user unit, which has one. From a bare cron job it would not.

## License

No license file yet, which means all rights reserved by default. Open an issue if you want to use this and I will add one.
