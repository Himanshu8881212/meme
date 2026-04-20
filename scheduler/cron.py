from __future__ import annotations

import sys
from pathlib import Path

CRONTAB_TEMPLATE = """# memory-system scheduled jobs
# Install with: crontab -e  (paste these lines)
# Adjust the python path and project root as needed.

# Daily decay pass — 03:30 local time
30 3 * * * cd {root} && {python} -m main decay >> {root}/vault/_meta/cron.log 2>&1

# Weekly vault monitor — Sundays at 04:00
0 4 * * 0 cd {root} && {python} -m main monitor >> {root}/vault/_meta/cron.log 2>&1

# Weekly deep reflection — Sundays at 04:30
30 4 * * 0 cd {root} && {python} -m main meta >> {root}/vault/_meta/cron.log 2>&1
"""


def render(project_root: Path) -> str:
    return CRONTAB_TEMPLATE.format(
        root=str(project_root.resolve()),
        python=sys.executable,
    )
