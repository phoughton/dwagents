"""FilesystemBackend variants tuned for the dwagents CLI."""

from __future__ import annotations

import os

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import WriteResult


class OverwritingFilesystemBackend(FilesystemBackend):
    """`FilesystemBackend` whose `write` overwrites an existing file.

    Upstream `FilesystemBackend.write` refuses to clobber an existing path and
    instructs the agent to use `edit_file` instead. For agent-owned output
    directories that fallback is brittle on whitespace-sensitive content (e.g.
    tab-indented XML): a partial write becomes uneditable and the agent has no
    other recovery path because the dwagents runtime does not expose `execute`.
    Overwriting unconditionally lets the agent recompose and re-emit the file
    in a single `write_file` call.
    """

    def write(self, file_path: str, content: str) -> WriteResult:
        resolved_path = self._resolve_path(file_path)

        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)

            return WriteResult(path=file_path)
        except (OSError, UnicodeEncodeError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")
