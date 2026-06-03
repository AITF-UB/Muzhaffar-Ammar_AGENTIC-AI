import json
import re

text = r'''{
    "test": "Here is \left and \frac and \times"
}'''

while True:
    try:
        data = json.loads(text)
        print("Success:", data)
        break
    except json.JSONDecodeError as e:
        print(f"Error: {e.msg} at pos {e.pos}")
        if e.msg == "Invalid \\escape":
            # The backslash is at e.pos - 1 ? Let's check.
            # text[e.pos] is the invalid char.
            # wait, let's print text around e.pos
            print("char at pos:", repr(text[e.pos]))
            print("char at pos-1:", repr(text[e.pos-1]))
            print("char at pos-2:", repr(text[e.pos-2]))
            
            # fix it by doubling the backslash
            # wait, if text[e.pos] is the invalid char and text[e.pos-1] is the backslash:
            # We want to replace the backslash at e.pos-1 with double backslash
            text = text[:e.pos-1] + "\\\\" + text[e.pos:]
            print("Fixed text:")
        else:
            break
