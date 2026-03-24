import re

with open('src/app/dash_callbacks.py') as f:
    callbacks = f.read()

outputs = re.findall(r"Output\(['\"]([^\"']+)['\"]", callbacks)

with open('src/app/dash_layout.py') as f:
    layout = f.read()

layout_ids = re.findall(r"id=['\"]([^\"']+)['\"]", layout)

missing = set(outputs) - set(layout_ids)
print("Missing IDs:", missing)
