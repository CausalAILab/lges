"""Locate optional Python reference sources independently of the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def default_lges_repo(project_root=PROJECT_ROOT):
    root = Path(project_root).resolve()
    candidates = (root.parent, root.parent / "lges", root.parent.parent / "lges")
    return next(
        (
            p
            for p in candidates
            if (p / "ges/ges/utils.py").is_file() and (p / "src/experiments").is_dir()
        ),
        candidates[1],
    )


def resolve_lges_repo(path=None):
    root = (
        Path(path).expanduser().resolve() if path is not None else default_lges_repo()
    )
    if not (root / "ges/ges/utils.py").is_file():
        raise FileNotFoundError(
            f"LGES graph utilities not found under {root}; pass --lges-repo"
        )
    return root
