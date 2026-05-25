"""Environment detection tool — lets the LLM inspect the runtime environment.

No hardcoded rules. The LLM decides what to do with the information.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool


def _run(cmd: str, timeout: int = 10) -> str:
    """Run a detection command, return stripped output or empty string."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout.strip() or r.stderr.strip() or "")
    except Exception:
        return ""


def create_env_tools(workspace_root: str | Path) -> list:
    """Create environment detection tools."""
    ws = Path(workspace_root)

    @tool
    def detect_environment() -> str:
        """Detect the current runtime environment.

        Returns info about: Python version, available package managers,
        GPU/CUDA, OS, and disk space. Use this before installing packages
        to choose the right tool, and after installing to verify.
        """
        lines = []

        # OS
        lines.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")

        # Python
        lines.append(f"Python: {sys.version.split()[0]} ({sys.executable})")

        # Package managers — just detect availability, no opinion
        pms = []
        for name, check_cmd in [
            ("uv", "uv --version 2>/dev/null"),
            ("pip", f"{sys.executable} -m pip --version 2>/dev/null"),
            ("conda", "conda --version 2>/dev/null"),
            ("poetry", "poetry --version 2>/dev/null"),
        ]:
            output = _run(check_cmd)
            if output:
                version = output.split("\n")[0][:60]
                pms.append(f"{name}: {version}")
            else:
                pms.append(f"{name}: not found")
        lines.append("Package managers:")
        for pm in pms:
            lines.append(f"  {pm}")

        # GPU
        lines.append("GPU:")
        nvidia_smi = _run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")
        if nvidia_smi:
            lines.append(f"  NVIDIA: {nvidia_smi}")
        cuda_ver = _run("nvcc --version 2>/dev/null | grep 'release' | awk '{print $5}' | tr -d ','")
        if cuda_ver:
            lines.append(f"  CUDA version: {cuda_ver}")
        if sys.platform == "darwin" and platform.machine() == "arm64":
            lines.append("  Apple Silicon (MPS available)")
        elif sys.platform == "darwin":
            lines.append("  Apple Intel (no GPU acceleration)")
        rocm = _run("rocm-smi --showproductname 2>/dev/null")
        if rocm:
            lines.append(f"  ROCm: {rocm[:120]}")
        if not nvidia_smi and not cuda_ver and not rocm:
            if "Apple" not in " ".join(lines):
                lines.append("  No GPU detected")

        # Installed Python packages (top-level, via pip list)
        pip_list = _run(f"{sys.executable} -m pip list --format=columns 2>/dev/null | head -40", timeout=15)
        if pip_list:
            lines.append(f"\nInstalled packages (pip list):\n{pip_list}")

        # Disk space in workspace
        try:
            usage = shutil.disk_usage(ws)
            free_gb = usage.free / (1024**3)
            lines.append(f"\nWorkspace: {ws}")
            lines.append(f"Disk free: {free_gb:.1f} GB")
        except Exception:
            pass

        return "\n".join(lines)

    return [detect_environment]
