import json, os

input_folder = "/mnt/c/Users/Flitto/Documents/WBL/ori"
output_folder = "/mnt/c/Users/Flitto/Documents/WBL/processed"
os.makedirs(output_folder, exist_ok=True)

for file in [f for f in os.listdir(input_folder) if f.endswith(".jsonl")]:
    in_path = os.path.join(input_folder, file)
    out_path = os.path.join(output_folder, file)

    with open(in_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    unique, seen = [], set()
    for item in items:
        q = item["question"]
        if q not in seen:
            seen.add(q)
            unique.append(item)

    for i, item in enumerate(unique, start=1):
        item["id"] = i

    with open(out_path, "w", encoding="utf-8") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{file}: {len(items)} → {len(unique)}개 (중복 제거 후 저장 완료)")
