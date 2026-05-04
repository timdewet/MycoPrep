"""GitHub Releases update check.

The frozen MycoPrep bundle pings the GitHub Releases API on startup; if
the latest tag is newer than the running ``mycoprep.__version__`` the
GUI shows a banner pointing to the release page. Running from source
(``python -m mycoprep`` in a venv) skips the check entirely so
developers don't get spurious "update available" prompts when they're
sitting on a tag-less working copy.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

import mycoprep

GITHUB_API = "https://api.github.com/repos/timdewet/MycoPrep/releases/latest"
WINDOWS_ASSET_NAME = "MycoPrep-windows.zip"


def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def _parse_version(v: str) -> tuple[int, ...]:
    """SemVer-ish tuple compare without pulling in ``packaging``.

    Strips a leading ``v``, splits on ``.``, and stops at the first
    non-digit run within a part (so ``1.0.0rc1`` → ``(1, 0, 0)``). The
    release workflow guarantees clean SemVer tags, so this is always
    enough — we only need ordering, not pre-release semantics.
    """
    v = v.strip().lstrip("vV")
    parts: list[int] = []
    for p in v.split("."):
        digits = ""
        for c in p:
            if c.isdigit():
                digits += c
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


@dataclass(frozen=True)
class ReleaseInfo:
    version: str         # "0.2.0" — no leading "v"
    tag: str             # "v0.2.0" — as published on GitHub
    release_url: str     # human-facing release page (always populated)
    asset_url: str       # direct download for MycoPrep-windows.zip ("" if missing)
    notes: str           # markdown release body


class UpdateChecker(QObject):
    """Async GitHub Releases poller.

    Issue ``check()`` once on startup. If a newer release is published
    and the current build is frozen, ``updateAvailable`` fires with a
    :class:`ReleaseInfo`. Otherwise ``upToDate`` or ``checkFailed``
    fires (the latter is silently ignored by the UI — failures here
    must never block app launch).
    """

    updateAvailable = pyqtSignal(object)  # ReleaseInfo
    upToDate = pyqtSignal()
    checkFailed = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._nam = QNetworkAccessManager(self)
        self._reply: Optional[QNetworkReply] = None

    def check(self, *, force_in_dev: bool = False) -> None:
        """Issue the GitHub API request. No-op when running from source.

        Pass ``force_in_dev=True`` to override the frozen check (useful
        for testing the UI path from a venv).
        """
        if not is_frozen() and not force_in_dev:
            return
        if self._reply is not None:
            return  # check already in flight
        req = QNetworkRequest(QUrl(GITHUB_API))
        req.setRawHeader(b"Accept", b"application/vnd.github+json")
        req.setRawHeader(
            b"User-Agent", f"MycoPrep/{mycoprep.__version__}".encode("ascii", "replace")
        )
        self._reply = self._nam.get(req)
        self._reply.finished.connect(self._on_finished)

    def _on_finished(self) -> None:
        reply = self._reply
        self._reply = None
        if reply is None:
            return
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self.checkFailed.emit(reply.errorString())
                return
            try:
                payload = json.loads(bytes(reply.readAll()).decode("utf-8", "replace"))
            except Exception as e:  # noqa: BLE001
                self.checkFailed.emit(f"bad JSON: {e}")
                return

            tag = (payload.get("tag_name") or "").strip()
            if not tag:
                self.checkFailed.emit("no tag_name in release payload")
                return
            version = tag.lstrip("vV")
            if _parse_version(version) <= _parse_version(mycoprep.__version__):
                self.upToDate.emit()
                return

            asset_url = ""
            for asset in payload.get("assets") or []:
                if asset.get("name") == WINDOWS_ASSET_NAME:
                    asset_url = asset.get("browser_download_url") or ""
                    break

            info = ReleaseInfo(
                version=version,
                tag=tag,
                release_url=payload.get("html_url") or "",
                asset_url=asset_url,
                notes=payload.get("body") or "",
            )
            self.updateAvailable.emit(info)
        finally:
            reply.deleteLater()
