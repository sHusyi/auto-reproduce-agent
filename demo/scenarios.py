"""Demo scenario definitions and setup.

Each scenario modifies a clean pytorch-cifar clone with a specific issue
that the agent must diagnose and fix.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_URL = "https://github.com/kuangliu/pytorch-cifar"


@dataclass
class Scenario:
    name: str
    difficulty: str
    description: str
    issue_description: str  # What the agent needs to figure out
    target_metrics: dict[str, float] = field(default_factory=lambda: {"accuracy": 93.0})

    # The patch function modifies the cloned repo to introduce the issue
    def apply(self, repo_path: Path) -> None:
        raise NotImplementedError


class Scenario1MissingDep(Scenario):
    """Easy: requirements.txt is missing torchvision."""

    def __init__(self):
        super().__init__(
            name="missing_dependency",
            difficulty="Easy",
            description="requirements.txt missing torchvision — agent must detect import error and install",
            issue_description="requirements.txt lists torch but not torchvision. "
            "Running the training script causes 'ModuleNotFoundError: No module named torchvision'.",
            target_metrics={"accuracy": 90.0},  # Lower target since we just want it to run
        )

    def apply(self, repo_path: Path) -> None:
        req = repo_path / "requirements.txt"
        if req.exists():
            content = req.read_text()
            # Remove torchvision line if present
            lines = [l for l in content.split("\n") if "torchvision" not in l]
            req.write_text("\n".join(lines))
        else:
            req.write_text("torch\nnumpy\n")


class Scenario2HardcodedPath(Scenario):
    """Medium: data path is hardcoded to a non-existent location."""

    def __init__(self):
        super().__init__(
            name="hardcoded_path",
            difficulty="Medium",
            description="Hardcoded data path points to non-existent directory — agent must find and fix",
            issue_description="The CIFAR-10 data directory is hardcoded to '/nonexistent/data/' "
            "instead of './data'. Training fails with FileNotFoundError.",
        )

    def apply(self, repo_path: Path) -> None:
        main_py = repo_path / "main.py"
        if not main_py.exists():
            return
        content = main_py.read_text()
        # Replace './data' or '../data' with a clearly wrong path
        content = content.replace("'./data'", "'/tmp/nonexistent-data'")
        content = content.replace('"./data"', '"/tmp/nonexistent-data"')
        main_py.write_text(content)


class Scenario3WrongHyperparam(Scenario):
    """Hard: Learning rate is 10x too high, causing loss divergence."""

    def __init__(self):
        super().__init__(
            name="wrong_hyperparam",
            difficulty="Hard",
            description="Learning rate set 10x too high — agent must diagnose from loss divergence",
            issue_description="The learning rate in main.py is set to 1.0 instead of 0.1. "
            "Training runs but loss diverges (NaN after a few batches). "
            "Agent must identify that LR is the cause, not a code bug.",
            target_metrics={"accuracy": 85.0},
        )

    def apply(self, repo_path: Path) -> None:
        main_py = repo_path / "main.py"
        if not main_py.exists():
            return
        content = main_py.read_text()
        # Change lr from 0.1 to 1.0
        content = content.replace("lr=0.1", "lr=1.0")
        main_py.write_text(content)


# Registry of all scenarios
SCENARIOS: dict[str, Scenario] = {
    "missing_dependency": Scenario1MissingDep(),
    "hardcoded_path": Scenario2HardcodedPath(),
    "wrong_hyperparam": Scenario3WrongHyperparam(),
}


def prepare_scenario(name: str, base_dir: str | Path | None = None) -> Path:
    """Clone pytorch-cifar and apply a scenario's modifications.

    Args:
        name: Scenario name (key in SCENARIOS dict)
        base_dir: Base directory for the clone (uses temp dir if None)

    Returns:
        Path to the prepared repository
    """
    scenario = SCENARIOS.get(name)
    if scenario is None:
        available = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario: {name}. Available: {available}")

    base = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="demo-"))
    base.mkdir(parents=True, exist_ok=True)
    repo_path = base / f"pytorch-cifar-{name}"

    if repo_path.exists():
        shutil.rmtree(repo_path)

    # Clone
    print(f"Cloning {REPO_URL}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(repo_path)],
        check=True, capture_output=True,
        timeout=60,
    )

    # Apply scenario
    print(f"Applying scenario: {scenario.name} ({scenario.difficulty})")
    scenario.apply(repo_path)

    print(f"Ready: {repo_path}")
    return repo_path


def list_scenarios() -> str:
    """Return a formatted list of available scenarios."""
    lines = ["Available Demo Scenarios", "=" * 30]
    for name, s in SCENARIOS.items():
        lines.append(f"\n[{s.difficulty}] {s.name}")
        lines.append(f"  {s.description}")
    return "\n".join(lines)
