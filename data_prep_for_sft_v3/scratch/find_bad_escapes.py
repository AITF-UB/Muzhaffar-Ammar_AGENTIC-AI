"""Find all unescaped backslash sequences in the raw response that would break json.loads."""
import json
import re

path = r"logs\raw_responses\essay_definisi_eksponen_1778514591.txt"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Remove \r for clean analysis
text = text.replace("\r\n", "\n")

print(f"Total chars: {len(text)}")

# Try parsing and show exact error
try:
    json.loads(text)
    print("JSON parsed OK!")
except json.JSONDecodeError as e:
    print(f"JSON error: {e}")
    print(f"Context around pos {e.pos}: {repr(text[max(0,e.pos-30):e.pos+30])}")

# Find all single backslashes followed by a letter (not part of valid JSON escapes)
# Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
# In the raw text, properly escaped LaTeX looks like \\frac 
# Unescaped LaTeX looks like \frac (single backslash)

# Let's find all \X patterns where X is a letter, excluding valid JSON escapes
for m in re.finditer(r'(?<!\\)\\([a-zA-Z])', text):
    char_after = m.group(1)
    pos = m.start()
    # Check if this is a valid JSON escape
    if char_after in ('n', 'r', 't', 'b', 'f'):
        # Could be valid JSON escape - skip
        continue
    if char_after == 'u':
        # Check if it's \uXXXX (valid unicode escape)
        remaining = text[pos+2:pos+6]
        if re.match(r'[0-9a-fA-F]{4}', remaining):
            continue
    # This is likely an unescaped LaTeX command
    context = text[max(0, pos-10):pos+20]
    print(f"  BAD ESCAPE at pos {pos}: {repr(context)}")
