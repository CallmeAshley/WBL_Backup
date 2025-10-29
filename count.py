import os, json

folder = "/mnt/c/Users/Flitto/Documents/WBL/processed"
total = 0

for f in os.listdir(folder):
    if f.endswith(".jsonl"):
        with open(os.path.join(folder, f), "r", encoding="utf-8") as file:
            total += sum(1 for line in file if line.strip())

print("총 객체 수:", total)
