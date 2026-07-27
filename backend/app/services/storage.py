"""Dataset file storage.

Uploads are streamed to disk in chunks and hashed on the way through. We
never call `await upload.read()` without a size argument: that materialises
the entire file in memory, so a single large upload can take the process
down regardless of what MAX_UPLOAD_MB says.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB
ALLOWED_SUFFIXES = {".csv", ".tsv", ".txt"}


class UploadTooLarge(ValueError):
    pass


class UnsupportedFileType(ValueError):
    pass


@dataclass
class StoredFile:
    path: Path
    original_filename: str
    size_bytes: int
    sha256: str

    @property
    def relative_path(self) -> str:
        """Path relative to STORAGE_DIR, which is what we persist.

        Storing an absolute path breaks the moment the container's mount
        point changes or the app moves to object storage.
        """
        return str(self.path.relative_to(settings.STORAGE_DIR)).replace("\\", "/")


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UnsupportedFileType(
            f"'{suffix or 'no extension'}' is not supported. "
            f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )
    return suffix


async def save_upload(upload: UploadFile, project_id: uuid.UUID) -> StoredFile:
    """Stream an upload to disk, enforcing the size limit as we go."""
    suffix = _safe_suffix(upload.filename or "")

    dest_dir = settings.STORAGE_DIR / "datasets" / str(project_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}{suffix}"

    digest = hashlib.sha256()
    total = 0
    try:
        with dest.open("wb") as fh:
            while chunk := await upload.read(CHUNK_SIZE):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise UploadTooLarge(
                        f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit."
                    )
                digest.update(chunk)
                fh.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)  # never leave a partial file behind
        raise

    if total == 0:
        dest.unlink(missing_ok=True)
        raise ValueError("Uploaded file is empty.")

    logger.info("Stored upload", extra={"bytes": total, "project_id": str(project_id)})
    return StoredFile(
        path=dest,
        original_filename=upload.filename or dest.name,
        size_bytes=total,
        sha256=digest.hexdigest(),
    )


def resolve(relative_path: str) -> Path:
    """Turn a stored relative path back into an absolute one, safely.

    The containment check blocks path traversal: without it, a crafted
    value like '../../etc/passwd' would escape the storage root.
    """
    candidate = (settings.STORAGE_DIR / relative_path).resolve()
    root = settings.STORAGE_DIR.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Resolved path escapes the storage directory.")
    return candidate


def delete(relative_path: str) -> None:
    try:
        resolve(relative_path).unlink(missing_ok=True)
    except ValueError:
        logger.warning("Refused to delete path outside storage root")
