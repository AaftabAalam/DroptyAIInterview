# IntelliInterview

Resume-driven interview app built with FastAPI, Ollama, and faster-whisper.

## Flow

1. Upload a resume in PDF, DOCX, TXT, or MD format.
2. The backend extracts candidate context from the resume.
3. Click `Start Interview` to get the first question.
4. Record an answer from the browser or type/edit it manually.
5. Click `Send Answer` to evaluate the answer and fetch the next question.
6. After 5 questions, the final report is generated with:
   - Communication
   - Technical relevance
   - Confidence
   - Clarity
   - Overall quality

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open [http://localhost:8000](http://localhost:8000).

## Requirements

- Ollama installed locally
- `llama3.2` available in Ollama
- Browser microphone permission enabled for recording answers

## Main routes

- `POST /resume/upload`
- `POST /interview/start`
- `POST /interview/transcribe`
- `POST /interview/answer`
- `POST /interview/stop`
- `GET /interview/status`
- `GET /interview/report`
- `GET /reports/{filename}`
- `WebSocket /ws`
