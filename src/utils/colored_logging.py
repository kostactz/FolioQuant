"""Colored console logging utilities.

Provides a lightweight ANSI color formatter and a helper to configure
the root logger's console handler so that `folio.trades`,
`folio.indicators` and warnings/errors are visually distinct.
"""
import logging
import os

RESET = "\x1b[0m"
COLORS = {
    'grey': '\x1b[38;21m',
    'green': '\x1b[32m',
    'yellow': '\x1b[33m',
    'red': '\x1b[31m',
    'blue': '\x1b[34m',
    'magenta': '\x1b[35m',
    'cyan': '\x1b[36m',
}


class ColorFormatter(logging.Formatter):
    """Formatter that injects ANSI colors into log output.

    Rules:
      - `folio.trades` messages are green
      - `folio.indicators` messages are cyan
      - WARNING messages are yellow
      - ERROR/CRITICAL messages are red
      - other INFO messages are blue
    """

    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        # allow disabling colors via NO_COLOR env var
        self.use_colors = use_colors and (os.environ.get('NO_COLOR') is None)

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)

        if not self.use_colors:
            return message

        # Priority: level-based warnings/errors override logger-based colors
        if record.levelno >= logging.ERROR:
            color = COLORS['red']
        elif record.levelno >= logging.WARNING:
            color = COLORS['yellow']
        elif record.name.startswith('folio.trades'):
            color = COLORS['green']
        elif record.name.startswith('folio.indicators'):
            color = COLORS['cyan']
        elif record.levelno == logging.INFO:
            color = COLORS['blue']
        else:
            color = COLORS['grey']

        return f"{color}{message}{RESET}"


def configure_colored_logging(level=logging.WARNING, fmt=None, datefmt=None):
    """Configure root logger with a colored console handler.

    This removes existing StreamHandlers on the root logger and installs
    a single StreamHandler using `ColorFormatter`.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # remove existing stream handlers to avoid double-printing
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler):
            root.removeHandler(h)

    if fmt is None:
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(ColorFormatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(handler)
