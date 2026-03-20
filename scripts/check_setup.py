import sys
from pathlib import Path


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

def main():
    print("1. Config...")
    try:
        from config import get_settings
        s = get_settings()
        print(f"   OK (model={s.ollama_model}, max_questions={s.max_questions})")
    except Exception as e:
        print(f"   FAIL:", e)
        return 1

    print("2. Ollama reachable...")
    try:
        import httpx
        r = httpx.get(f"{s.ollama_base_url}/api/tags", timeout=5.0)
        if r.status_code == 200:
            print("   OK")
        else:
            print(f"   HTTP {r.status_code} (start Ollama if you need the LLM)")
    except Exception as e:
        print(f"   Not reachable: {e}")
        print("   Start Ollama (e.g. ollama run llama3.2) for question generation.")

    print("3. App import...")
    try:
        from src.api.app import app
        print("   OK")
    except Exception as e:
        print(f"   FAIL:", e)
        return 1

    print("\nAll checks done. Run: python run.py  then open http://localhost:8000")
    return 0

if __name__ == "__main__":
    sys.exit(main())
