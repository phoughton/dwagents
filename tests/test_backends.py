"""Tests for `dwagents.backends.filesystem.OverwritingFilesystemBackend`."""

from __future__ import annotations

from pathlib import Path

import pytest

from dwagents.backends.filesystem import OverwritingFilesystemBackend


@pytest.fixture
def backend(tmp_path: Path) -> OverwritingFilesystemBackend:
    return OverwritingFilesystemBackend(root_dir=tmp_path, virtual_mode=False)


def test_write_creates_new_file(backend: OverwritingFilesystemBackend, tmp_path: Path):
    target = tmp_path / "new.txt"

    result = backend.write(str(target), "hello")

    assert result.error is None
    assert result.path == str(target)
    assert target.read_text() == "hello"


def test_write_overwrites_existing_file(
    backend: OverwritingFilesystemBackend, tmp_path: Path
):
    target = tmp_path / "x.txt"
    target.write_text("first")

    result = backend.write(str(target), "second")

    assert result.error is None, "expected no error when overwriting"
    assert result.path == str(target)
    assert target.read_text() == "second"


def test_write_creates_parent_directories(
    backend: OverwritingFilesystemBackend, tmp_path: Path
):
    target = tmp_path / "sub" / "dir" / "file.txt"

    result = backend.write(str(target), "nested")

    assert result.error is None
    assert target.read_text() == "nested"


def test_write_then_read_roundtrip(
    backend: OverwritingFilesystemBackend, tmp_path: Path
):
    target = tmp_path / "rt.txt"

    backend.write(str(target), "alpha")
    backend.write(str(target), "beta")

    read_result = backend.read(str(target))

    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "beta"
