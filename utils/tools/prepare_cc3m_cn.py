#!/usr/bin/env python3
"""
一键脚本：国内环境下载/清洗 CC3M，并生成 cc3m_natural_10K_WObanana.csv

功能：
1) 使用 HuggingFace `conceptual_captions`（支持 HF 镜像）流式读取；
2) 过滤 caption 中含 banana 的样本；
3) 下载图片（失败重试 + 超时）；
4) 坏图过滤（PIL verify + 最小分辨率）；
5) 生成训练 CSV（image, caption）。
"""

import argparse
import csv
import hashlib
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import requests
from PIL import Image, UnidentifiedImageError
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CC3M subset for BadCLIP in CN environment")
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HuggingFace endpoint mirror for CN environment",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--target-count", type=int, default=10000, help="Number of valid samples to keep")
    parser.add_argument("--max-tries", type=int, default=200000, help="Max source samples to iterate before stopping")
    parser.add_argument("--num-workers", type=int, default=32, help="Download threads")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retries per image")
    parser.add_argument("--timeout", type=float, default=8.0, help="Request timeout seconds")
    parser.add_argument("--min-size", type=int, default=64, help="Min image width/height")

    parser.add_argument(
        "--image-dir",
        type=str,
        default="/mnt/zfs/tang/0_MMCL/BadCLIP-master/data/GCC_Training500K/cc3m",
        help="Directory to store downloaded images",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/mnt/zfs/tang/0_MMCL/BadCLIP-master/data/GCC_Training500K/cc3m_natural_10K_WObanana.csv",
        help="Output CSV path with columns image,caption",
    )
    parser.add_argument(
        "--relative-prefix",
        type=str,
        default="cc3m",
        help="Prefix used in CSV image column (relative to csv directory)",
    )
    return parser.parse_args()


def caption_has_banana(text: str) -> bool:
    if not text:
        return True
    return bool(re.search(r"banana", text, flags=re.IGNORECASE))


def safe_filename(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h}.jpg"


def validate_image_bytes(content: bytes, min_size: int) -> Optional[bytes]:
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
        with Image.open(io.BytesIO(content)) as img:
            img = img.convert("RGB")
            if min(img.size) < min_size:
                return None
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=95)
            return out.getvalue()
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def download_one(
    url: str,
    caption: str,
    image_dir: str,
    relative_prefix: str,
    retries: int,
    timeout: float,
    min_size: int,
) -> Optional[Tuple[str, str]]:
    if not url:
        return None

    filename = safe_filename(url)
    abs_path = os.path.join(image_dir, filename)
    rel_path = f"{relative_prefix}/{filename}" if relative_prefix else filename

    if os.path.exists(abs_path):
        try:
            with Image.open(abs_path) as img:
                if min(img.size) >= min_size:
                    return rel_path, caption
        except (UnidentifiedImageError, OSError, ValueError):
            try:
                os.remove(abs_path)
            except OSError:
                pass

    for i in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, stream=False)
            if resp.status_code != 200:
                continue
            valid_bytes = validate_image_bytes(resp.content, min_size=min_size)
            if valid_bytes is None:
                return None
            with open(abs_path, "wb") as f:
                f.write(valid_bytes)
            return rel_path, caption
        except requests.RequestException:
            if i < retries - 1:
                time.sleep(0.3 * (i + 1))
            continue
    return None


def main() -> int:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    print(f"[INFO] HF_ENDPOINT={args.hf_endpoint}")
    print("[INFO] Loading conceptual_captions (streaming=True)...")

    try:
        ds = load_dataset("conceptual_captions", split=args.split, streaming=True)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset metadata: {e}")
        return 2

    selected = []
    attempted = 0
    skipped_banana = 0

    def generate_candidates():
        nonlocal attempted, skipped_banana
        for item in ds:
            attempted += 1
            if attempted > args.max_tries:
                break
            caption = (item.get("caption") or "").strip()
            if caption_has_banana(caption):
                skipped_banana += 1
                continue
            url = item.get("image_url")
            if not url:
                continue
            yield url, caption

    print("[INFO] Downloading images with retries and filtering bad images...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = []
        for url, caption in generate_candidates():
            futures.append(
                ex.submit(
                    download_one,
                    url,
                    caption,
                    args.image_dir,
                    args.relative_prefix,
                    args.retries,
                    args.timeout,
                    args.min_size,
                )
            )
            if len(futures) >= args.target_count * 3:
                break

        for i, fut in enumerate(futures, start=1):
            res = fut.result()
            if res is not None:
                selected.append(res)
            if i % 500 == 0:
                print(f"[INFO] Processed futures={i}, valid={len(selected)}")
            if len(selected) >= args.target_count:
                break

    if len(selected) < args.target_count:
        print(
            f"[WARN] Only got {len(selected)} valid samples (< {args.target_count}). "
            "You can increase --max-tries or rerun."
        )

    selected = selected[: args.target_count]

    with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption"])
        writer.writerows(selected)

    print("[DONE] Finished preparing dataset")
    print(f"[DONE] CSV: {args.csv_path}")
    print(f"[DONE] Images dir: {args.image_dir}")
    print(f"[DONE] valid={len(selected)}, attempted={attempted}, skipped_banana={skipped_banana}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
