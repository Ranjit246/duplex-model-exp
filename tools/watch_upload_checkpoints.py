"""Watch a training output dir, upload each finished step_<N> checkpoint to HF, then delete it.

A step_<N> folder is considered finished when its total size on disk has not changed
for STABLE_SECONDS. Each step_<N> is uploaded to the same-named subfolder on the hub.
After an upload the step folder is kept on disk until a higher-numbered step has also
been uploaded — i.e. the most recent uploaded checkpoint always remains locally so a
training resume can read it without re-downloading from the hub. config.json at the
parent is uploaded once when first observed.

Files larger than CHUNK_BYTES (default 45 GiB) are streamed into <CHUNK_TMP_ROOT>
as `<name>.part_00`, `.part_01`, ... and each chunk is uploaded individually. The
original file is NOT uploaded as-is, so the hub copy needs `reassemble_checkpoint.sh`
(also uploaded to the repo root) to be glued back together before use.

Env vars:
    HF_TOKEN         HuggingFace write token (required)
    CKPT_ROOT        Local dir to watch (default: output/moshiko-finetuned)
    REPO_ID          Target HF repo (default: Ranjit/moshiko-kame-hinglish-ft-exp)
    POLL_SECONDS     How often to scan (default: 60)
    STABLE_SECONDS   How long size must be unchanged (default: 300)
    CHUNK_BYTES      Max bytes per chunk for >50GB files (default: 45 GiB)
    CHUNK_TMP_ROOT   Where to stage chunks (default: /tmp/ckpt_chunks)
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ckpt-watcher")


@dataclass
class StepState:
    size: int
    last_changed: float  # monotonic timestamp


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def stream_split_file(src: Path, work_dir: Path, max_bytes: int) -> list[Path]:
    """Stream src into work_dir as <src.name>.part_00, .part_01, ...

    Reads with a moderate IO buffer; never holds more than one chunk on disk at a
    time IF callers upload+delete each chunk before requesting the next. This
    function writes all chunks up front — for incremental upload, use
    `iter_split_file` instead.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    buf_size = 64 * 1024 * 1024  # 64MiB IO
    with src.open("rb") as fin:
        idx = 0
        while True:
            chunk_path = work_dir / f"{src.name}.part_{idx:02d}"
            written = 0
            with chunk_path.open("wb") as fout:
                while written < max_bytes:
                    to_read = min(buf_size, max_bytes - written)
                    buf = fin.read(to_read)
                    if not buf:
                        break
                    fout.write(buf)
                    written += len(buf)
            if written == 0:
                chunk_path.unlink(missing_ok=True)
                break
            chunks.append(chunk_path)
            log.info("  wrote %s (%.2f GiB)", chunk_path.name, written / 2**30)
            idx += 1
    return chunks


def upload_step(
    api: HfApi,
    repo_id: str,
    step_dir: Path,
    chunk_bytes: int,
    chunk_tmp_root: Path,
) -> None:
    # Identify files exceeding HF's per-file LFS cap.
    large_files: list[Path] = []
    for p in step_dir.rglob("*"):
        try:
            if p.is_file() and p.stat().st_size > chunk_bytes:
                large_files.append(p)
        except OSError:
            pass

    ignore_patterns = [str(p.relative_to(step_dir).as_posix()) for p in large_files]

    log.info(
        "uploading %s -> %s:%s/ (chunked: %d file(s))",
        step_dir, repo_id, step_dir.name, len(large_files),
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(step_dir),
        path_in_repo=step_dir.name,
        commit_message=f"upload {step_dir.name} (skinny files)",
        ignore_patterns=ignore_patterns if ignore_patterns else None,
    )

    for src in large_files:
        rel = src.relative_to(step_dir)
        size_gib = src.stat().st_size / 2**30
        log.info("splitting %s (%.2f GiB) into <= %.2f GiB chunks via %s",
                 rel, size_gib, chunk_bytes / 2**30, chunk_tmp_root)
        work_dir = chunk_tmp_root / step_dir.name / rel.parent
        if work_dir.exists():
            shutil.rmtree(work_dir)
        try:
            chunks = stream_split_file(src, work_dir, chunk_bytes)
            for chunk in chunks:
                in_repo = f"{step_dir.name}/{rel.parent.as_posix()}/{chunk.name}".replace("//", "/")
                log.info("uploading chunk %s", in_repo)
                api.upload_file(
                    repo_id=repo_id,
                    repo_type="model",
                    path_or_fileobj=str(chunk),
                    path_in_repo=in_repo,
                    commit_message=f"upload {in_repo}",
                )
                # free disk as we go
                chunk.unlink(missing_ok=True)
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    log.info("upload of %s finished", step_dir.name)


def upload_config_if_needed(api: HfApi, repo_id: str, ckpt_root: Path, already: set[str]) -> None:
    cfg = ckpt_root / "config.json"
    if not cfg.exists() or "config.json" in already:
        return
    log.info("uploading config.json")
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(cfg),
        path_in_repo="config.json",
        commit_message="upload config.json",
    )
    already.add("config.json")


def upload_reassemble_script_if_needed(
    api: HfApi, repo_id: str, already: set[str]
) -> None:
    if "reassemble_checkpoint.sh" in already:
        return
    script = Path(__file__).parent / "reassemble_checkpoint.sh"
    if not script.exists():
        return
    log.info("uploading reassemble_checkpoint.sh")
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(script),
        path_in_repo="reassemble_checkpoint.sh",
        commit_message="upload reassemble_checkpoint.sh",
    )
    already.add("reassemble_checkpoint.sh")


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN env var is required")
        return 2

    ckpt_root = Path(os.environ.get("CKPT_ROOT", "output/moshiko-finetuned")).resolve()
    repo_id = os.environ.get("REPO_ID", "Ranjit/moshiko-kame-hinglish-ft-exp")
    poll_seconds = int(os.environ.get("POLL_SECONDS", "60"))
    stable_seconds = int(os.environ.get("STABLE_SECONDS", "300"))
    chunk_bytes = int(os.environ.get("CHUNK_BYTES", str(45 * 2**30)))
    chunk_tmp_root = Path(os.environ.get("CHUNK_TMP_ROOT", "/tmp/ckpt_chunks")).resolve()

    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except HfHubHTTPError as e:
        log.error("cannot access repo %s: %s", repo_id, e)
        return 2

    log.info(
        "watching %s -> %s (poll=%ds, stable=%ds, chunk=%.2f GiB, tmp=%s)",
        ckpt_root, repo_id, poll_seconds, stable_seconds,
        chunk_bytes / 2**30, chunk_tmp_root,
    )

    states: dict[str, StepState] = {}
    uploaded_once: set[str] = set()
    # step_<N> dirs that have been uploaded but are kept on disk until a newer one uploads.
    pending_delete: dict[str, Path] = {}
    max_uploaded_step: int | None = None

    while True:
        try:
            if not ckpt_root.exists():
                log.info("ckpt root %s missing, sleeping", ckpt_root)
            else:
                upload_config_if_needed(api, repo_id, ckpt_root, uploaded_once)
                upload_reassemble_script_if_needed(api, repo_id, uploaded_once)

                step_dirs = sorted(
                    [p for p in ckpt_root.glob("step_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else -1,
                )

                now = time.monotonic()
                for step_dir in step_dirs:
                    name = step_dir.name
                    cur_size = dir_size_bytes(step_dir)
                    prev = states.get(name)
                    if prev is None:
                        states[name] = StepState(size=cur_size, last_changed=now)
                        log.info("tracking %s (size=%.2f GiB)", name, cur_size / 2**30)
                        continue
                    if cur_size != prev.size:
                        log.info(
                            "%s still writing: %.2f -> %.2f GiB",
                            name, prev.size / 2**30, cur_size / 2**30,
                        )
                        states[name] = StepState(size=cur_size, last_changed=now)
                        continue
                    stable_for = now - prev.last_changed
                    if stable_for < stable_seconds:
                        log.info(
                            "%s stable for %ds/%ds (%.2f GiB) — waiting",
                            name, int(stable_for), stable_seconds, cur_size / 2**30,
                        )
                        continue

                    try:
                        upload_step(api, repo_id, step_dir, chunk_bytes, chunk_tmp_root)
                    except Exception as e:
                        log.error("upload failed for %s: %s — will retry", name, e)
                        # reset stability window so we don't hammer on a transient failure
                        states[name] = StepState(size=cur_size, last_changed=now)
                        continue

                    states.pop(name, None)
                    pending_delete[name] = step_dir
                    try:
                        step_num = int(name.split("_")[1])
                    except (IndexError, ValueError):
                        step_num = -1
                    if max_uploaded_step is None or step_num > max_uploaded_step:
                        max_uploaded_step = step_num
                    log.info("retaining %s on disk until a newer step uploads", name)

                # Delete any uploaded step that's older than the latest uploaded one.
                if max_uploaded_step is not None:
                    for pname, pdir in list(pending_delete.items()):
                        try:
                            pnum = int(pname.split("_")[1])
                        except (IndexError, ValueError):
                            continue
                        if pnum < max_uploaded_step:
                            try:
                                if pdir.exists():
                                    shutil.rmtree(pdir)
                                    log.info(
                                        "deleted local %s (newer step_%d uploaded)",
                                        pdir, max_uploaded_step,
                                    )
                            except OSError as e:
                                log.error("failed to delete %s: %s", pdir, e)
                                continue
                            pending_delete.pop(pname, None)

                # forget size-tracking state for dirs that no longer exist
                live = {p.name for p in step_dirs}
                for gone in [k for k in states if k not in live]:
                    states.pop(gone, None)
        except Exception as e:
            log.exception("watcher loop error: %s", e)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
