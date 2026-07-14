from io import StringIO

from imessage_mlx.progress import ProgressBar


def test_progress_bar_emits_bounded_log_updates() -> None:
    stream = StringIO()
    progress = ProgressBar("Stage", 20, stream=stream, log_step_percent=25)

    for _ in range(20):
        progress.advance()
    progress.close()

    output = stream.getvalue()
    assert "Stage [------------------------] 0/20 (  0%)" in output
    assert "Stage [########################] 20/20 (100%)" in output
    assert len(output.splitlines()) == 6


def test_disabled_progress_bar_is_silent() -> None:
    stream = StringIO()
    progress = ProgressBar("Stage", 10, enabled=False, stream=stream)

    progress.advance()
    progress.close()

    assert stream.getvalue() == ""
