import json
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(r'C:\Local D\Galeri Belajar\Project\SR_02\data_prep_for_sft_v3')
SOURCE_DIR = BASE_DIR / 'output' / 'current_experiments'
SYSTEM_PROMPT_PATH = BASE_DIR / 'instruction' / 'system_prompt.md'
OUTPUT_BASE = BASE_DIR / 'standardized'

def load_system_prompt():
    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read().strip()

def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_jsonl(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def process_pipeline():
    new_system_prompt = load_system_prompt()
    tasks_data = {} # Format: {task_name: {subject_name: [list_of_data]}}

    print("🚀 Crawling files...")
    for json_file in SOURCE_DIR.rglob('*.json'):
        rel_parts = json_file.relative_to(SOURCE_DIR).parts
        if len(rel_parts) < 6: continue 
        
        subject = rel_parts[3] 
        task = rel_parts[5]    

        if task not in tasks_data:
            tasks_data[task] = {}
        if subject not in tasks_data[task]:
            tasks_data[task][subject] = []

        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                if isinstance(content, list):
                    tasks_data[task][subject].extend(content)
                else:
                    tasks_data[task][subject].append(content)
            except Exception as e:
                print(f"❌ Error reading {json_file}: {e}")

    all_merged_clean = []

    print("🧹 Cleaning and organizing into sub-folders...")
    for task, subjects in tasks_data.items():
        task_all_subjects_clean = []
        
        # Sub-folder per task
        task_raw_dir = OUTPUT_BASE / 'raw' / task
        task_clean_dir = OUTPUT_BASE / 'clean' / task
        task_gold_dir = OUTPUT_BASE / 'gold' / task

        for subject, items in subjects.items():
            subj_name = subject.replace(' ', '_')
            
            # 1. RAW
            save_json(items, task_raw_dir / f"{task}_{subj_name}.json")

            # 2. CLEANING
            cleaned_items = [
                {
                    "system": new_system_prompt,
                    "user": item.get("user", item.get("user_prompt", "")),
                    "assistant": item.get("assistant", item.get("assistant_response", ""))
                } for item in items
            ]
            
            # Save Clean (JSON) & Gold (JSONL)
            save_json(cleaned_items, task_clean_dir / f"{task}_{subj_name}.json")
            save_jsonl(cleaned_items, task_gold_dir / f"{task}_{subj_name}.jsonl")

            task_all_subjects_clean.extend(cleaned_items)
            all_merged_clean.extend(cleaned_items)

        # Save Merged per Task (Outside subject sub-folders but inside task folder)
        save_json(task_all_subjects_clean, task_clean_dir / f"{task}_all_merged.json")
        save_jsonl(task_all_subjects_clean, task_gold_dir / f"{task}_all_merged.jsonl")

    # Final Overall Merged
    save_json(all_merged_clean, OUTPUT_BASE / 'clean' / "final_all_tasks_merged.json")
    save_jsonl(all_merged_clean, OUTPUT_BASE / 'gold' / "final_all_tasks_merged.jsonl")

    print(f"\n✅ Selesai! Cek folder: {OUTPUT_BASE}")

if __name__ == "__main__":
    process_pipeline()