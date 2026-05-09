from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import pytest

from gia_evidence_finder.cli import main


def test_extract_command_supports_readme_quick_example(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = tmp_path / "sample.md"
    document.write_text(
        """# README

No evaluation, no search, no opening book.
UltraChess is a browser-based chess variant playground.
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gia-evidence-finder",
            "extract",
            str(document),
            "--claim",
            "UltraChess is browser-based",
        ],
    )

    main()

    payload = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert payload["abstained"] is False
    matches = cast(list[dict[str, object]], payload["matches"])
    assert matches[0]["text"] == "UltraChess is a browser-based chess variant playground."
