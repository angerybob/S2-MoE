import json
import re
from pathlib import Path

# =========================
# Configuration parameters
# =========================
DATASET_PATH = "datastes/gsm8K.jsonl"
PREDICTIONS_PATH = "answers/gsm8K_answers.jsonl"
OUTPUT_PATH = "answers/gsm8K_eval.json"

QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
PREDICTION_FIELD = "prediction"
ID_FIELD = "id"


def extract_reference(answer_text):
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer_text)
    if not match:
        return None
    return match.group(1)


def extract_prediction(pred_text):
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", pred_text.replace(",", ""))
    if not numbers:
        return None
    return numbers[-1]


def normalize_number(value):
    try:
        if value is None:
            return None
        if "." in value:
            return str(float(value))
        return str(int(value))
    except ValueError:
        return None


def load_dataset(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_predictions(path):
    preds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("summary"):
                continue
            preds.append(record)
    return preds


def main():
    dataset = load_dataset(DATASET_PATH)
    predictions = load_predictions(PREDICTIONS_PATH)

    pred_by_id = {p.get(ID_FIELD, idx): p for idx, p in enumerate(predictions)}

    results = []
    correct = 0
    total = 0

    for idx, item in enumerate(dataset):
        pred = pred_by_id.get(idx)
        if pred is None:
            continue
        ref_raw = item.get(ANSWER_FIELD, "")
        pred_raw = pred.get(PREDICTION_FIELD, "")

        ref = normalize_number(extract_reference(ref_raw))
        guess = normalize_number(extract_prediction(pred_raw))

        is_correct = ref is not None and guess is not None and ref == guess
        total += 1
        correct += 1 if is_correct else 0

        results.append(
            {
                "id": idx,
                "question": item.get(QUESTION_FIELD, ""),
                "reference": ref,
                "prediction": guess,
                "correct": is_correct,
            }
        )

    accuracy = correct / total if total else 0.0
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }

    output = {"summary": summary, "details": results}
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
