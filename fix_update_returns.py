with open("src/services/metrics_service.py", "r") as f:
    text = f.read()

# Replace current_pos back to target_pos in _update_returns
# It's at lines 340-350 roughly.
import re

def repl(m):
    # m.group(1) is the _update_returns body up to the gross_return line
    return m.group(1) + "gross_return = target_pos * price_return"

text = re.sub(r'(def _update_returns.*?target_pos = Decimal\(\'0.0\'\)\n\s+)gross_return = current_pos \* price_return', repl, text, flags=re.DOTALL)

with open("src/services/metrics_service.py", "w") as f:
    f.write(text)

