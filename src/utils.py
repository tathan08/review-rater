import json, re, subprocess

def run_ollama(model: str, prompt: str) -> str:
    # Call Ollama; ensure 'ollama pull {model}' was done beforehand
    res = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    txt = res.stdout.decode("utf-8", errors="ignore").strip()
    return txt

def extract_json(txt: str):
    # Be robust: find the first {...} block
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    block = m.group(0)
    return json.loads(block)
