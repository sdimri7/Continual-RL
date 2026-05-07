"""
One-time environment setup for Google Colab.

Run `full_setup()` once after every runtime restart.
All heavy packages are cached to Drive so pip skips re-downloading
on subsequent sessions (when the target directory already exists).
"""

import os
import subprocess
import sys


# ── paths ──────────────────────────────────────────────────────────────────
DRIVE_ROOT = "/content/drive/MyDrive"
PACKAGES_DIR = os.path.join(DRIVE_ROOT, "continual_rl_packages")


# ── helpers ────────────────────────────────────────────────────────────────
def _run(cmd: str, check: bool = True) -> int:
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")
    return result.returncode


# ── vulkan ─────────────────────────────────────────────────────────────────
def setup_vulkan() -> None:
    """Install Vulkan ICD files required for GPU-accelerated rendering.

    Must be re-run every Colab session (files live on the ephemeral VM).
    """
    print("[setup] Configuring Vulkan … ", end="", flush=True)
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


# ── packages ───────────────────────────────────────────────────────────────
def install_packages(target_dir: str = PACKAGES_DIR) -> None:
    """Install Python dependencies, caching them to Drive.

    On subsequent sessions pip detects the packages are already installed
    and skips the download, making restarts fast (~10 s vs ~3 min).

    Args:
        target_dir: Directory on Drive where packages are cached.
    """
    os.makedirs(target_dir, exist_ok=True)

    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    packages = [
        "mani_skill",
        "tyro",
        "diffusers",          # DDPM scheduler
        "einops",             # tensor rearrangement used in UNet
        "zarr",               # efficient demo storage
        "gym-pusht",          # Push-T environment (Remi Cadene)
    ]
    pkg_str = " ".join(packages)
    print(f"[setup] Installing packages to {target_dir} … ", flush=True)
    _run(f'pip install --target="{target_dir}" --upgrade {pkg_str} -q')

    # Force Python to recognise newly installed packages in the target dir.
    import site
    site.main()
    print("[setup] Packages ready.")


# ── combined entry point ───────────────────────────────────────────────────
def full_setup(packages_dir: str = PACKAGES_DIR) -> None:
    """Mount-agnostic one-liner called at the top of every Colab session.

    Steps
    -----
    1. Vulkan (always required, files are ephemeral).
    2. Pip install with Drive cache (fast on second+ runs).
    3. Appends package dir to sys.path if not already there.
    """
    setup_vulkan()
    install_packages(packages_dir)
    print("[setup] Environment ready. ✓")
