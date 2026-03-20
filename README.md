# IntelliInterview

Resume-driven interview app built with FastAPI, Ollama, and faster-whisper.

## Flow

1. Upload a resume in PDF, DOCX, TXT, or MD format.
2. The backend extracts candidate context from the resume.
3. Click `Start Interview` to get the first question.
4. Record an answer from the browser or type/edit it manually.
5. Click `Send Answer` to evaluate the answer and fetch the next question.
6. After 3 questions, the final report is generated with:
   - Communication
   - Technical relevance
   - Confidence
   - Clarity
   - Overall quality

## Run
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open [http://localhost:8000](http://localhost:8000).

## Windows Run
Use PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Open `http://localhost:8000` in Chrome or Edge.

# Requirements
- Ollama installed locally
- `llama3.2` available in Ollama
- Browser microphone permission enabled for recording answers
- Open the app on `localhost` or HTTPS for direct browser microphone recording

## Windows Notes
- Keep Ollama running before starting the app.
- If browser recording is blocked, the app falls back to audio capture/file upload for transcription.
- If Whisper transcription fails on Windows, retry from `localhost` and make sure the browser recording was saved successfully.

# APIs
- `POST /resume/upload`
- `POST /interview/start`
- `POST /interview/transcribe`
- `POST /interview/answer`
- `POST /interview/stop`
- `GET /interview/status`
- `GET /interview/report`
- `GET /reports/{filename}`
- `WebSocket /ws`
