from pathlib import Path

from imessage_mlx.utils import ensure_private_dir


def test_ensure_private_dir_secures_intermediate_directories(tmp_path: Path) -> None:
    nested = ensure_private_dir(tmp_path / "private/intermediate/final")
    assert nested.stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "private").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "private/intermediate").stat().st_mode & 0o777 == 0o700
