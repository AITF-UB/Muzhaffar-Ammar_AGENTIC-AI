import os
import json
import re

# =====================================================
# FOLDER JSON
# =====================================================

folder_path = r'C:\Local D\Galeri Belajar\Project\SR_02\data_prep_for_sft_v3\data\chunk\Kurikulum Merdeka\SMA\Kelas 10'

# =====================================================
# REGEX
# =====================================================

# Tangkap:
# [1]
# [1,2]
# [1, 2]
# [1-3]
# [1; 2]
# [ 12 ]
citation_pattern = re.compile(
    r'\[\s*\d+(?:\s*[,;\-]\s*\d+)*\s*\]'
)

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    if not isinstance(text, str):
        return text

    original = text

    # Hapus citation
    text = re.sub(citation_pattern, '', text)

    # Rapikan spasi berlebih
    text = re.sub(r'[ \t]+', ' ', text)

    # Rapikan spasi sebelum tanda baca
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Rapikan multiple newline
    text = re.sub(r'\n+', '\n', text)

    text = text.strip()

    return text

# =====================================================
# RECURSIVE CLEAN
# =====================================================

def recursive_clean(data):

    if isinstance(data, dict):
        return {
            key: recursive_clean(value)
            for key, value in data.items()
        }

    elif isinstance(data, list):
        return [
            recursive_clean(item)
            for item in data
        ]

    elif isinstance(data, str):
        return clean_text(data)

    return data

# =====================================================
# PROCESS FILES
# =====================================================

def process_json_files(folder):

    total = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if not file.endswith('.json'):
                continue

            file_path = os.path.join(root, file)

            print(f'Processing: {file}')

            # =========================================
            # LOAD JSON
            # =========================================

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            except Exception as e:
                print(f'❌ Gagal baca {file}')
                print(e)
                continue

            # =========================================
            # CLEAN
            # =========================================

            cleaned_data = recursive_clean(data)

            # =========================================
            # BACKUP
            # =========================================

            backup_path = file_path.replace('.json', '_backup.json')

            if not os.path.exists(backup_path):

                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        data,
                        f,
                        indent=2,
                        ensure_ascii=False
                    )

            # =========================================
            # SAVE
            # =========================================

            try:

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        cleaned_data,
                        f,
                        indent=2,
                        ensure_ascii=False
                    )

                print(f'✅ Cleaned: {file}')
                total += 1

            except Exception as e:
                print(f'❌ Gagal save {file}')
                print(e)

    print('\n================================')
    print(f'Selesai cleaning {total} file')
    print('================================')

# =====================================================
# MAIN
# =====================================================

if __name__ == '__main__':
    process_json_files(folder_path)