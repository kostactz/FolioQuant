import re

with open("src/services/metrics_service.py", "r") as f:
    text = f.read()

# Replace direct assignment in __init__
text = text.replace("self.signal_threshold = Decimal(str(signal_threshold))", "self._signal_threshold = Decimal(str(signal_threshold))")
text = text.replace("self.hysteresis_band = self.signal_threshold * Decimal('0.5')", "self.hysteresis_band = self._signal_threshold * Decimal('0.5')")

# Add property
prop_code = """
    @property
    def signal_threshold(self) -> Decimal:
        return self._signal_threshold

    @signal_threshold.setter
    def signal_threshold(self, value: Decimal):
        self._signal_threshold = Decimal(str(value))
        self.hysteresis_band = self._signal_threshold * Decimal('0.5')
"""
# Insert property after __init__
text = re.sub(r'(def __init__.*?\n(?: {4}.*?\n)*)', r'\1' + prop_code, text, count=1)

with open("src/services/metrics_service.py", "w") as f:
    f.write(text)
