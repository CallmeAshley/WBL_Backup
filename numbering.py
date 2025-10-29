import json

input_path = "생명과학_신경과학.jsonl"
output_path = "생명과학_신경과학_numbering.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

for i, item in enumerate(items, start=1):
    item["id"] = i

with open(output_path, "w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
