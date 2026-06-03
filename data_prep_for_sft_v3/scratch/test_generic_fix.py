"""Test a smarter JSON escape fixer that handles \\\\sqrt correctly."""
import json
import re


def fix_invalid_json_escapes(s):
    r"""
    Fix invalid backslash escapes in a JSON string.
    
    JSON only allows these escape sequences:
      \" \\ \/ \b \f \n \r \t \uXXXX
    
    The model sometimes outputs \cdot, \approx, \times etc. with a single
    backslash, which are invalid JSON escapes. We need to double these.
    
    But we must NOT touch already-valid sequences like:
      \\frac  (JSON: escaped backslash + "frac" = literal \frac) 
      \n      (JSON: newline)
      \"      (JSON: escaped quote)
    
    Strategy: Process character by character, tracking escape state.
    """
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\':
            # We found a backslash. Check what comes next.
            if i + 1 < len(s):
                next_char = s[i + 1]
                if next_char in ('"', '/', 'b', 'f', 'n', 'r', 't'):
                    # Valid JSON escape — keep as-is
                    result.append(s[i])
                    result.append(s[i + 1])
                    i += 2
                elif next_char == '\\':
                    # This is \\ — an escaped backslash (valid JSON).
                    # Output both and skip ahead.
                    result.append('\\')
                    result.append('\\')
                    i += 2
                elif next_char == 'u':
                    # Could be \uXXXX unicode escape. Check for 4 hex digits.
                    if i + 5 < len(s) and re.match(r'[0-9a-fA-F]{4}', s[i+2:i+6]):
                        # Valid unicode escape
                        result.append(s[i:i+6])
                        i += 6
                    else:
                        # Invalid \u without proper hex — double the backslash
                        result.append('\\\\')
                        i += 1
                else:
                    # Invalid escape like \c, \a, \x, \l, \s, etc.
                    # Double the backslash to make it a literal backslash in JSON
                    result.append('\\\\')
                    i += 1
            else:
                # Trailing backslash at end of string
                result.append('\\\\')
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


# Test 1: basic LaTeX escapes
json_str = r'{"val": "x \cdot y and \\frac{1}{2} and $E(3) \approx 48.64$"}'
print(f"Test 1 input:  {repr(json_str)}")
fixed = fix_invalid_json_escapes(json_str)
print(f"Test 1 fixed:  {repr(fixed)}")
try:
    data = json.loads(fixed)
    print(f"Test 1 parsed: val = {repr(data['val'])}")
except json.JSONDecodeError as e:
    print(f"Test 1 error: {e}")

# Test 2: preserve valid JSON escapes
json_str2 = '{"val": "line1\\nline2\\ttab and \\"quote\\""}'
print(f"\nTest 2 input:  {repr(json_str2)}")
fixed2 = fix_invalid_json_escapes(json_str2)
print(f"Test 2 fixed:  {repr(fixed2)}")
try:
    data2 = json.loads(fixed2)
    print(f"Test 2 parsed: val = {repr(data2['val'])}")
except json.JSONDecodeError as e:
    print(f"Test 2 error: {e}")

# Test 3: double-escaped LaTeX (\\frac should stay as-is)
json_str3 = '{"val": "properly escaped \\\\frac{1}{2} and \\\\sqrt{4}"}'
print(f"\nTest 3 input:  {repr(json_str3)}")
fixed3 = fix_invalid_json_escapes(json_str3)
print(f"Test 3 fixed:  {repr(fixed3)}")
try:
    data3 = json.loads(fixed3)
    print(f"Test 3 parsed: val = {repr(data3['val'])}")
except json.JSONDecodeError as e:
    print(f"Test 3 error: {e}")

# Test 4: real raw response file
print("\n=== Test 4: Real raw response file ===")
with open(r"logs\raw_responses\essay_definisi_eksponen_1778514591.txt", "r", encoding="utf-8") as f:
    raw = f.read()

fixed_raw = fix_invalid_json_escapes(raw)
try:
    data4 = json.loads(fixed_raw)
    print(f"SUCCESS! Parsed {len(data4)} items")
    for item in data4:
        print(f"  {item['id']}: {item['level']}")
except json.JSONDecodeError as e:
    print(f"FAILED: {e}")
    print(f"Context: {repr(fixed_raw[max(0,e.pos-30):e.pos+30])}")

# Test 5: all previous raw responses
print("\n=== Test 5: All raw response files ===")
import os
raw_dir = r"logs\raw_responses"
for fname in sorted(os.listdir(raw_dir)):
    fpath = os.path.join(raw_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    fixed_content = fix_invalid_json_escapes(content)
    try:
        json.loads(fixed_content)
        print(f"  OK: {fname}")
    except json.JSONDecodeError as e:
        print(f"  FAIL: {fname} — {e.msg} at pos {e.pos}")
        print(f"        Context: {repr(fixed_content[max(0,e.pos-20):e.pos+20])}")
