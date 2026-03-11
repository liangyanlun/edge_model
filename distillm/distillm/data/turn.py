import json, random

src = "mini_data.jsonl"
train_out = r"dolly/train.jsonl"
valid_out = r"dolly/valid.jsonl"

random.seed(42)
valid_ratio = 0.01

train_f = open(train_out, "w", encoding="utf-8")
valid_f = open(valid_out, "w", encoding="utf-8")

with open(src, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)

        obj = {
            "instruction": ex.get("instruction", "").strip(),
            "input": (ex.get("input", "") or "").strip(),
            "output": ex.get("output", "").strip(),
        }
        if not obj["instruction"] or not obj["output"]:
            continue

        out = valid_f if random.random() < valid_ratio else train_f
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")

train_f.close()
valid_f.close()

print("done:", train_out, valid_out)
