import json
from collections import Counter

file_path = r"C:\Local D\Galeri Belajar\Project\SR_02\data_prep_for_sft_v3\standardized\gold\final_all_tasks_merged.jsonl"

level_counter = Counter()

total_objects = 0
total_assistant_lists = 0
total_questions = 0

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read().strip()

decoder = json.JSONDecoder()

idx = 0

while idx < len(content):

    obj, end = decoder.raw_decode(content, idx)

    total_objects += 1

    assistant_data = obj.get("assistant", [])

    # assistant stored as stringified JSON
    if isinstance(assistant_data, str):
        try:
            assistant_data = json.loads(assistant_data)
        except json.JSONDecodeError:
            assistant_data = []

    if isinstance(assistant_data, list):

        total_assistant_lists += 1

        for q in assistant_data:

            # nested stringified JSON
            if isinstance(q, str):
                try:
                    q = json.loads(q)
                except json.JSONDecodeError:
                    continue

            if isinstance(q, dict):

                total_questions += 1

                level = q.get("level", "UNKNOWN")
                level_counter[level] += 1

    idx = end

    while idx < len(content) and content[idx].isspace():
        idx += 1

print("\n========== DATASET SUMMARY ==========")
print(f"Total JSON objects     : {total_objects}")
print(f"Total assistant lists  : {total_assistant_lists}")
print(f"Total questions        : {total_questions}")

print("\n========== LEVEL COUNTS ==========")

for level, count in level_counter.items():
    percentage = (count / total_questions) * 100
    print(f"{level:<10} : {count:>6} ({percentage:.2f}%)")