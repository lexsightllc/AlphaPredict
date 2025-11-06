"""Run the full AlphaPredict pipeline inside a Kaggle environment."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


KAGGLE_DATA_ROOT = Path("/kaggle/input/hull-tactical-market-prediction")
KAGGLE_CODE_ROOT = Path("/kaggle/input/alphapredict")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TARGET_REPO_NAME = "AlphaPredict"
ENV_FLAG = "ALPHAPREDICT_MONOLITHIC"


class PipelineError(RuntimeError):
    """Raised when the Kaggle pipeline cannot be executed."""


def _copy_repo_to_working(current_repo: Path) -> Path:
    """Ensure a writable copy of the repository exists in /kaggle/working."""

    working_repo = KAGGLE_WORKING_ROOT / TARGET_REPO_NAME

    if working_repo.exists():
        return working_repo

    if KAGGLE_CODE_ROOT.exists():
        candidate = KAGGLE_CODE_ROOT
        if (candidate / TARGET_REPO_NAME).is_dir():
            candidate = candidate / TARGET_REPO_NAME
    else:
        candidate = current_repo

    if not candidate.exists():
        raise PipelineError(
            "Unable to locate the AlphaPredict sources to copy into /kaggle/working."
        )

    shutil.copytree(candidate, working_repo)
    return working_repo


def _ensure_data_files(repo_root: Path) -> None:
    """Copy Kaggle competition CSVs into data/External."""

    destination = repo_root / "data" / "External"
    destination.mkdir(parents=True, exist_ok=True)

    for filename in ("train.csv", "test.csv"):
        source = KAGGLE_DATA_ROOT / filename
        target = destination / filename
        if not source.exists():
            raise PipelineError(f"Required dataset {source} is missing.")
        if not target.exists():
            shutil.copy2(source, target)


def _run_command(command: list[str], cwd: Path | None = None) -> None:
    """Execute a subprocess command, streaming output to the notebook."""

    print(f"\n[AlphaPredict] Running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=cwd)


def _install_dependencies(repo_root: Path) -> None:
    """Install Python dependencies unless explicitly skipped."""

    if os.environ.get("ALPHAPREDICT_SKIP_PIP") == "1":
        print("[AlphaPredict] Skipping dependency installation (ALPHAPREDICT_SKIP_PIP=1).")
        return

    requirements_min = repo_root / "requirements-minimal.txt"
    requirements_full = repo_root / "requirements.txt"

    if requirements_min.exists():
        _run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_min)])
    elif requirements_full.exists():
        _run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_full)])
    else:
        print("[AlphaPredict] No requirements files found; skipping pip install.")


def _run_pipeline(repo_root: Path) -> None:
    """Execute training and backtesting scripts sequentially."""

    scripts_dir = repo_root / "scripts"
    _run_command([sys.executable, str(scripts_dir / "train.py")], cwd=repo_root)
    _run_command([sys.executable, str(scripts_dir / "backtest.py")], cwd=repo_root)


def _is_repo_root(path: Path) -> bool:
    """Return ``True`` if *path* appears to be the AlphaPredict repository root."""

    return (path / "scripts" / "train.py").is_file() and (
        path / "scripts" / "backtest.py"
    ).is_file()


def _determine_repo_root() -> Path:
    """Best-effort detection of the repository root across Kaggle environments."""

    candidates: list[Path] = []

    env_hint = os.environ.get("ALPHAPREDICT_REPO_ROOT")
    if env_hint:
        candidates.append(Path(env_hint))

    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parents[1])

    if sys.argv and sys.argv[0]:
        argv_path = Path(sys.argv[0])
        if argv_path.exists():
            candidates.append(argv_path.resolve().parent)

    cwd = Path.cwd()
    search_roots = [cwd, *cwd.parents]
    kaggle_roots = [KAGGLE_WORKING_ROOT, KAGGLE_CODE_ROOT]
    search_roots.extend(kaggle_roots)

    for root in search_roots:
        candidates.append(root)
        candidates.append(root / TARGET_REPO_NAME)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue

        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)

        if resolved.is_file():
            resolved = resolved.parent

        if _is_repo_root(resolved):
            return resolved

    raise PipelineError(
        "Unable to locate the AlphaPredict repository root. Please run the script "
        "from within or alongside the project directory."
    )


def main() -> None:
    repo_root = _determine_repo_root()

    if os.environ.get(ENV_FLAG) != "1":
        working_repo = _copy_repo_to_working(repo_root)
        if working_repo.resolve() != repo_root.resolve():
            env = os.environ.copy()
            env[ENV_FLAG] = "1"
            command = [sys.executable, str(working_repo / "scripts" / "run_kaggle_pipeline.py")]
            print("[AlphaPredict] Re-launching pipeline from /kaggle/working copy.")
            subprocess.run(command, check=True, env=env)
            return
        os.environ[ENV_FLAG] = "1"

    _ensure_data_files(repo_root)
    _install_dependencies(repo_root)
    _run_pipeline(repo_root)


if __name__ == "__main__":  # pragma: no cover - entry point
    try:
        main()
    except PipelineError as exc:
        print(f"[AlphaPredict] ERROR: {exc}")
        sys.exit(1)
