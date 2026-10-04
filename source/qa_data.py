from __future__ import annotations

import json
import urllib.request
from pathlib import Path


SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
HOTPOT_URL = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOTPOT_MIRROR = "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json?download=true"


def load_squad(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(SQUAD_URL, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    contexts = []
    questions = []
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            context_id = len(contexts)
            contexts.append(context)
            for qa in paragraph["qas"]:
                answers = list(dict.fromkeys(item["text"].strip() for item in qa["answers"]))
                answers = [answer for answer in answers if answer]
                if answers:
                    questions.append((context_id, context, qa["question"].strip(), answers))
    return contexts, questions


def load_hotpot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        error = None
        for url in (HOTPOT_MIRROR, HOTPOT_URL):
            try:
                with urllib.request.urlopen(url, timeout=30) as response, part.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                part.replace(path)
                break
            except Exception as current:
                error = current
                if part.exists():
                    part.unlink()
        else:
            raise RuntimeError("could not download HotpotQA") from error
    return json.loads(path.read_text(encoding="utf-8"))
