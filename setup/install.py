# -*- coding: utf-8 -*-
"""
One-time environment setup for Google Colab.

Run `full_setup()` once after every runtime restart.
All heavy packages are cached to Drive so pip skips re-downloading
on subsequent sessions (when the target directory already exists).

When run locally (not on Colab), Vulkan setup is skipped and packages
are installed into the normal Python environment instead of Drive.
"""

import os
import subprocess
import sys


# -- environment detection ---------------------------------------------------
def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# -- paths -------------------------------------------------------------------
DRIVE_ROOT = "/content/drive/MyDrive"
PACKAGES_DIR = os.path.join(DRIVE_ROOT, "continual_rl_packages")


# -- helpers -----------------------------------------------------------------
def _run(cmd: str, check: bool = True) -> int:
    # security-reviewed: all commands are hardcoded strings, no user input
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")
    return result.returncode


# -- vulkan ------------------------------------------------------------------
def setup_vulkan() -> None:
    """Install Vulkan ICD files required for GPU-accelerated rendering on Colab.

    Only relevant on Colab's Linux VM. Skipped automatically when running
    locally on macOS or a regular Linux machine.
    Must be re-run every Colab session (the files are ephemeral).
    """
    if not _is_colab():
        print("[setup] Not on Colab — skipping Vulkan setup.")
        return

    print("[setup] Configuring Vulkan ... ", end="", flush=True)
    _run("mkdir -p /usr/share/vulkan/icd.d")
    _run(
        "wget -q https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json"
        " -O /usr/share/vulkan/icd.d/nvidia_icd.json"
    )
    _run(
        "wget -q https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json"
        " -O /usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )
    _run("apt-get install -y --no-install-recommends libvulkan-dev -q")
    print("done.")


# -- local install advice ----------------------------------------------------
def _local_install_advice(packages: list) -> None:
    """Print install instructions for local development instead of auto-installing.

    Splits packages into two groups:
      - cross-platform: safe to install on macOS / Windows.
      - linux-only: require Linux + NVIDIA GPU (mani_skill, gym-pusht/pygame).
    """
    # These packages require Linux + NVIDIA GPU or SDL headers to compile.
    linux_only = {"mani_skill", "gym-pusht"}  # require Linux + NVIDIA GPU / SDL
    cross_platform = [p for p in packages if p not in linux_only]

    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    env_label = sys.prefix if in_venv else "system Python (no active venv detected)"

    print(f"[setup] Running locally in: {env_label}")
    print("[setup] Skipping auto-install — manage packages in your own environment.")
    print()
    print("[setup] Install cross-platform packages (safe on macOS/Windows):")
    print(f"    pip install {' '.join(cross_platform)}")
    print()
    print("[setup] Linux/GPU-only packages (install only on Colab / Linux + NVIDIA):")
    for p in linux_only:
        print(f"    # pip install {p}")
    print()
    print("[setup] Tip: mani_skill and gym-pusht both require an NVIDIA GPU and")
    print("[setup]      Linux. On macOS you can edit code but must run the sim on Colab.")


# -- packages ----------------------------------------------------------------
def install_packages(target_dir: str = PACKAGES_DIR) -> None:
    """Install Python dependencies.

    On Colab:  installs into ``target_dir`` on Drive so subsequent restarts
               skip the download entirely (~10 s vs ~3 min).
    Locally:   installs normally into the active Python environment;
               ``target_dir`` is ignored.

    Args:
        target_dir: Drive directory used for caching on Colab.
    """
    packages = [
        "torch",
        "torchvision",
        "mani_skill",
        "tyro",
        "diffusers",   # DDPM scheduler
        "einops",      # tensor rearrangement used in UNet
        "zarr",        # efficient demo storage
        "gym-pusht",   # Push-T environment (Remi Cadene)
    ]
    pkg_str = " ".join(packages)

    # Use the pip that belongs to the current Python interpreter, not whatever
    # 'pip' resolves to in $PATH (avoids Python 2 pip on macOS).
    pip = f'"{sys.executable}" -m pip'

    if _is_colab():
        os.makedirs(target_dir, exist_ok=True)
        if target_dir not in sys.path:
            sys.path.insert(0, target_dir)
        print(f"[setup] Installing packages to Drive cache ({target_dir}) ...", flush=True)
        _run(f'{pip} install --target="{target_dir}" --upgrade {pkg_str} -q')
        import site
        site.main()
    else:
        _local_install_advice(packages)

    print("[setup] Packages ready.")


# -- combined entry point ----------------------------------------------------
def full_setup(packages_dir: str = PACKAGES_DIR) -> None:
    """One-liner called at the top of every session (Colab or local).

    On Colab:
        1. Configures Vulkan for GPU rendering (ephemeral, must redo each session).
        2. Installs pip packages into Drive cache (fast after first run).

    Locally (macOS / Linux):
        1. Skips Vulkan entirely.
        2. Installs pip packages into the active Python environment.
    """
    setup_vulkan()
    install_packages(packages_dir)
    print("[setup] Environment ready.")
