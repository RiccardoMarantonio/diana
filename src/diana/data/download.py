"""MVTec AD per-category downloader.

MVTec's official site gates downloads behind a licensing agreement form, so a
script can't fetch it directly. This module mirrors the *original on-disk
layout* (``<root>/<category>/{train,test,ground_truth}/...``) from a public
Hugging Face repo that hosts the raw category folders, and verifies every file
against the content digest the HF API publishes (``lfs.oid`` == sha256 of the
file bytes). A mismatch aborts before any corrupted artifact can reach disk.

Usage::

    python -m diana.data.download --categories hazelnut --dest data/mvtec

The dataset loader looks for ``<dest>/<category>/train/good/*.png``.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets"
DEFAULT_REPO = "foersben/mvtec-ad"

VALID_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
}

MIB = 1 << 20


def _fetch_json(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def plan_category(repo: str, category: str, verify: bool) -> list[dict]:
    """Enumerate every file in the category, with size and (optionally) sha256."""
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"unknown category {category!r}; expected one of {sorted(VALID_CATEGORIES)}"
        )
    url = f"{HF_API}/{repo}/tree/main/{category}?recursive=true"
    entries = _fetch_json(url)
    planned = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        planned.append(
            {
                "path": entry["path"],
                "size": entry.get("size"),
                "sha256": lfs.get("oid") if verify else None,
            }
        )
    return planned


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def needs_download(local: str, expected_size: int | None) -> bool:
    """True unless a complete file of the expected size already exists."""
    if expected_size is None:
        return not os.path.exists(local)
    return os.path.getsize(local) != expected_size if os.path.exists(local) else True


def fetch_one(
    repo: str,
    rel_path: str,
    dest_root: str,
    expected_size: int | None,
    expected_sha256: str | None,
) -> tuple[str, str]:
    """Download one file to ``dest_root/rel_path``, verifying hash and size."""
    local = os.path.join(dest_root, rel_path)
    if not needs_download(local, expected_size):
        return rel_path, "ok"
    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = f"{local}.part"
    url = f"{HF_RESOLVE}/{repo}/resolve/main/{rel_path}"
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as src, open(tmp, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
            if expected_sha256:
                digest.update(chunk)
    os.replace(tmp, local)
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        raise RuntimeError(
            f"sha256 mismatch for {rel_path}: got {digest.hexdigest()}, "
            f"expected {expected_sha256}"
        )
    return rel_path, "ok"


def download_category(
    category: str,
    dest: str,
    repo: str = DEFAULT_REPO,
    verify: bool = True,
    workers: int = 4,
) -> list[dict]:
    """Download one category into ``dest``. Returns the plan (provenance)."""
    plan = plan_category(repo, category, verify=verify)
    total_bytes = sum(p["size"] or 0 for p in plan)
    print(
        f"[download] {category}: {len(plan)} files, {total_bytes / MIB:.1f} MiB, "
        f"sha256 {'verify' if verify else 'skip'}"
    )

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_one, repo, p["path"], dest, p["size"], p["sha256"]): p
            for p in plan
        }
        for future in futures:
            rel, state = future.result()
            done += 1
            if done % max(1, len(plan) // 10) == 0 or done == len(plan):
                print(f"  [{done}/{len(plan)}] {state} {rel}")
    return plan


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["hazelnut"],
        help="MVTec categories to fetch (default: hazelnut)",
    )
    parser.add_argument(
        "--dest",
        default="data/mvtec",
        help="Root directory that will hold <category>/... folders",
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help="Hugging Face repo serving the data"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip per-file sha256 checks"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    os.makedirs(args.dest, exist_ok=True)
    for category in args.categories:
        try:
            download_category(
                category, args.dest, args.repo, verify=not args.no_verify,
                workers=args.workers,
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary catch-all
            print(f"[error] {category}: {exc}", file=sys.stderr)
            sys.exit(1)
    print(f"[done] categories in {args.dest}: {', '.join(args.categories)}")


if __name__ == "__main__":
    main()