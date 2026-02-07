# AI-Driven Automated Interviewer for Project Presentations

An AI system that **listens to a student presenting a project** (screen share + speech) and conducts an **adaptive interview** based on content and responses, then produces a **score + feedback report**.

## Features

- **Presentation understanding**: OCR on screen content, speech-to-text (STT), analysis of UI, code, slides
- **Dynamic interviewing**: Context-aware questions from extracted content; follow-ups based on answers and screen
- **Evaluation & feedback**: Scores on **technical depth**, **clarity**, **originality**, **understanding**; HTML report

All core components use **free, local** tools: faster-whisper (STT), EasyOCR, Ollama (LLM), edge-tts (TTS).

## Requirements

- **Python 3.10+**
- **Ollama** installed and running (e.g. `ollama run llama3.2`)
- Microphone and screen share (presenter’s machine)

## Setup

1. **Create a virtualenv and install dependencies**

   ```bash
   cd AutomatedInterviewer
   conda create -n name python=3.12
   conda activate name
   pip install -r requirements.txt
   ```

2. **Install and run Ollama** (if not already)

   - Install from [ollama.com](https://ollama.com)
   - In a terminal: `ollama run llama3.2` (or another model; set `INTERVIEW_OLLAMA_MODEL` if different)

3. **Optional**: Copy `.env.example` to `.env` and adjust (e.g. `INTERVIEW_OLLAMA_MODEL`, `INTERVIEW_MAX_QUESTIONS`).

## Run the live demo

1. Start the server from the project root:

   ```bash
   python run.py
   ```

   Or: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`

2. Open **http://localhost:8000** in a browser.

3. Click **Start Interview**:
   - Share your screen (or leave as-is for the machine’s screen).
   - Present your project out loud; the system will capture screen (OCR) and speech (STT).
   - After a short presentation phase, the AI will ask context-aware questions (TTS).
   - Answer verbally; after each answer it evaluates and may ask a follow-up.
   - When the interview ends, the **Report** section and the **Open full report (HTML)** link show scores and feedback.

4. Use **End Interview** to stop early.

## How to test this project (step-by-step)

### Step 1: One-time setup (if not done)

```bash
cd AutomatedInterviewer
conda activate name_of_the_env
pip install -r requirements.txt
```

### Step 2: Optional – test microphone only

```bash
python scripts/test_microphone.py
```

Speak when prompted. You should see a **peak level** (e.g. 3000+). If you see an error or peak 0, fix mic permissions in **System Settings → Privacy & Security → Microphone**.

### Step 3: Start the app

```bash
python run.py
```

Leave this running. You should see: `Uvicorn running on http://0.0.0.0:8000`.

### Step 4: Open the UI

In your browser go to: **http://localhost:8000**

You should see the **AI Automated Interviewer** page with buttons: **Start Interview**, **End Interview**, **Test voice (record 6s + transcribe)**.

### Step 5: Optional – test voice (mic + speech-to-text)

Before running a full interview:

1. Click **"Test voice (record 6s + transcribe)"**.
2. When it says *Recording 6 seconds… speak now*, **speak clearly** for a few seconds.
3. After ~6 seconds it will show either a **transcript** of what you said or an **error**.

If you see a transcript, the mic and Whisper pipeline work. If you see an error, fix it (e.g. mic permission, or the float16 fix already applied) before the full interview.

### Step 6: Optional – start Ollama (for AI questions)

For **real AI-generated questions** (not generic fallbacks), in a **second terminal**:

```bash
ollama run llama3.2
```

Wait until the model is loaded. If you don’t have Ollama, skip this; the app will still run with fallback questions.

### Step 7: Run a full interview

1. On the demo page, click **"Start Interview"**.
2. **Speak** and **show something on screen** (e.g. code, slides). The page will show:
   - **Live transcript** (updates every ~1.5 s; first text may take ~5–10 seconds).
   - **Screen content (OCR)** (updates every ~8 s).
3. After about **30 seconds** the AI will ask the first question (you’ll hear it via TTS). **Answer out loud** for at least 5–10 seconds.
4. It will ask more questions (default 5 total), then show the **Report** with scores and **"Open full report (HTML)"**.
5. Use **"End Interview"** anytime to stop early.

### Step 8: Shorter test (optional)

To run a quick 2-question test:

```bash
export INTERVIEW_MAX_QUESTIONS=2
export INTERVIEW_MIN_PRESENTATION_SECONDS=10
python run.py
```

Then open http://localhost:8000 and click **Start Interview**. You’ll get 2 questions after ~10 seconds of presentation.

### Step 9: View the report

When the interview ends:

- On the page: see **Report** (scores + summary) and click **"Open full report (HTML)"**.
- On disk: open a file in the **`reports/`** folder (e.g. `report_<timestamp>.html`).

---

**Quick checklist:** Setup → (optional) mic script → `python run.py` → open http://localhost:8000 → (optional) Test voice → (optional) Ollama → Start Interview → speak + show screen → get report.

## Configuration (env)

| Variable | Default | Description |
|----------|---------|-------------|
| `INTERVIEW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `INTERVIEW_OLLAMA_MODEL` | `llama3.2` | Model name |
| `INTERVIEW_WHISPER_MODEL_SIZE` | `base` | faster-whisper size: tiny, base, small, medium, large-v3 |
| `INTERVIEW_MAX_QUESTIONS` | `5` | Max questions per session |
| `INTERVIEW_MIN_PRESENTATION_SECONDS` | `30` | Seconds of presentation before first question |
| `INTERVIEW_SCREEN_CAPTURE_INTERVAL_SEC` | `8` | Seconds between screen OCR captures |

## Project layout

```
AutomatedInterviewer/
├── config.py              # Settings (env)
├── requirements.txt
├── run.py                 # Run uvicorn
├── README.md
├── static/
│   └── index.html         # Demo UI
├── reports/               # Generated HTML reports (created at runtime)
└── src/
    ├── api/
    │   └── app.py         # FastAPI app, WebSocket, routes
    └── services/
        ├── screen_ocr.py  # Screen capture + EasyOCR
        ├── stt.py         # faster-whisper STT
        ├── tts.py         # edge-tts TTS
        ├── llm.py         # Ollama question generation & evaluation
        ├── evaluator.py   # Score aggregation & HTML report
        └── orchestrator.py # Interview flow
```

## API (summary)

- `GET /` – Demo UI
- `POST /interview/start` – Start interview (background)
- `POST /interview/stop` – Stop interview
- `GET /interview/status` – Current state + last report (for polling)
- `GET /interview/report` – Last report JSON
- `WebSocket /ws` – Live state updates (phase, question, transcript, scores)
- `GET /reports/{filename}` – Download a generated report HTML file

## Troubleshooting

- **Ollama not found**: Start Ollama and pull a model, e.g. `ollama run llama3.2`.
- **"No speech yet" / voice not detected**:
  1. **Test the microphone**: From project root run `python scripts/test_microphone.py`. It records 3 seconds and prints a peak level. If you see "Error opening stream" or peak is 0, the app cannot access the mic. With the server running, you can also use **"Test voice (record 6s + transcribe)"** on the demo page to verify the full mic + Whisper pipeline.
  2. **Grant permission**: On macOS, allow Terminal (or Cursor/VS Code) to use the microphone in **System Settings → Privacy & Security → Microphone**.
  3. **Pick the right device**: The app uses the system default input. If that’s wrong (e.g. a display or external device), list devices with the app running: open **http://localhost:8000/api/audio/devices**. Note the `index` of your mic, then set it in `.env`: `INTERVIEW_AUDIO_INPUT_DEVICE=1` (use the index you see).
  4. **Wait for first transcript**: Transcription runs every ~5 seconds. Speak for at least 5–10 seconds; the live transcript updates every 1.5 s once there is text.
  5. If the UI shows **"[Microphone error: ...]"**, the message describes the failure (e.g. permission or invalid device).
- **OCR empty**: Ensure the captured screen contains visible text (window/slide/code in focus).
- **TTS no sound**: edge-tts plays via system audio; check volume and that no other process is blocking audio.

## License

MIT.
