import json
import re

text = r'''{
    "pembahasan": "Line 1\nLine 2\nHere is \frac{1}{2} and \times \u201C \nu"
}'''

def _sanitize_json(text: str) -> str:
    def replacer(match):
        word = match.group(1)
        if word in ("n", "r", "t", "b", "f", "u", "v"):
            return "\\" + word
        return "\\\\" + word
    
    return re.sub(r'(?<!\\)\\([a-zA-Z]+)', replacer, text)

print("ORIGINAL TEXT:")
print(repr(text))

fixed_text = _sanitize_json(text)

print("\nFIXED TEXT:")
print(repr(fixed_text))

try:
    data = json.loads(fixed_text)
    print("\nPARSED JSON:")
    print(repr(data['pembahasan']))
except Exception as e:
    print("JSON Error:", e)
