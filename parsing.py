import os
import json
import re

# === 사용자 설정 ===
input_path = "/mnt/c/Users/Flitto/Documents/WBL/화학_계산화학.json"          # 원본 JSON 파일 경로
output_dir = "/mnt/c/Users/Flitto/Documents/WBL"      # 출력 폴더 경로
os.makedirs(output_dir, exist_ok=True)

# === JSON 파일 불러오기 ===
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === question 필드 안의 여러 문항 분리 ===
questions_raw = data["question"].strip().split("\n")
questions = []

# 각 문항마다 id 증가시키며 저장
for i, q in enumerate(questions_raw, start=1):
    q_clean = q.strip()
    if not q_clean:
        continue
    questions.append({
        "id": i,
        "question": q_clean,
        "subject": data.get("subject", "")
    })

# === 50개씩 분할 ===
chunks = [questions[i:i + 50] for i in range(0, len(questions), 50)]

# === JSONL로 저장 ===
for idx, chunk in enumerate(chunks):
    output_path = os.path.join(output_dir, "화학_계산화학_parsed.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in chunk:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved: {output_path} ({len(chunk)} items)")
