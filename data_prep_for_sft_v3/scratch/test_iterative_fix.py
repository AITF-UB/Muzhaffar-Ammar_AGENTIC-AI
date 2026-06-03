import json

text = r'''{
    "pembahasan": "Model matematis.\n\nUntuk memprediksi $$J_n = 2^n$$\n\n**Keterbatasan:**\n1. Algoritma\n2. Intervensi\n\frac{1}{2} \times \left( x \right) \u201C",
    "real_newline": "Line 1
Line 2"
}'''

def robust_json_parse(json_str):
    while True:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if "Invalid \\escape" in str(e):
                pos = e.pos
                # e.pos is the index of the character AFTER the backslash (e.g., 'f' in '\frac')
                # Wait! Let's check: in previous run, e.pos for \left was 23.
                # text[23] was '\'. This means e.pos is the index of the backslash itself!
                char_at_pos = json_str[pos]
                if char_at_pos == '\\':
                    # The backslash is at pos. So we want to replace it with \\
                    json_str = json_str[:pos] + "\\\\" + json_str[pos+1:]
                else:
                    # Just in case, if e.pos is after the backslash
                    # e.g., pos is at 'f' and pos-1 is '\'
                    if json_str[pos-1] == '\\':
                        json_str = json_str[:pos-1] + "\\\\" + json_str[pos:]
                    else:
                        raise e
            elif "Invalid control character" in str(e):
                pos = e.pos
                char = json_str[pos]
                if char == '\n':
                    json_str = json_str[:pos] + "\\n" + json_str[pos+1:]
                elif char == '\t':
                    json_str = json_str[:pos] + "\\t" + json_str[pos+1:]
                else:
                    raise e
            else:
                raise e

print("ORIGINAL TEXT:")
print(repr(text))

try:
    data = robust_json_parse(text)
    print("\nPARSED JSON:")
    print(repr(data['pembahasan']))
    print(repr(data['real_newline']))
except Exception as e:
    print("JSON Error:", e)
