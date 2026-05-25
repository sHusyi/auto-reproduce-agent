"""UI rendering layer — all terminal I/O and formatting lives here.

The rest of the codebase calls into this module for display, never using
Rich or ANSI codes directly. This makes it possible to swap the terminal UI
for a web UI or GUI without touching business logic.
"""

from src.ui.terminal import Terminal
from src.ui.orchestrator_display import OrchestratorDisplay
