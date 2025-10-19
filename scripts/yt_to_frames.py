import argparse
import math
import os
import sys
import subprocess
from pathlib import Path

import cv2
import pandas as pd

def download_youtube(url: str, out_dir: Path) -> Path:
    """
    Downloads a YouTube video as MP4 using yt-dlp and returns the local file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use yt-dlp CLI to avoid import issues across environments
    # Output template ensures a stable filename
    template = str(out_dir / "%(title)s.%(ext)s")
    cmd = ["yt-dlp", "-f", "mp4", "-o", template, url]
    print("[yt-dlp] downloading…")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError("yt-dlp failed (see output above).")

    # Find the most recent mp4 in out_dir
    mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise FileNotFoundError("No MP4 found after yt-dlp run.")
    return mp4s[0]

def extract_frames(video_path: Path, out_frames_dir: Path, every_sec: float, start_sec: float, end_sec: float | None) -> pd.DataFrame:
    """
    Extracts frames every `every_sec` seconds from `video_path` into `out_frames_dir`.
    Returns a DataFrame manifest with filename, width, height, frame_idx, timestamp_sec.
    """
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if total_frames > 0 else None

    if end_sec is None and duration_sec is not None:
        end_sec = duration_sec

    if start_sec < 0:
        start_sec = 0
    if end_sec is not None and end_sec < start_sec:
        raise ValueError("end_sec must be >= start_sec")

    # Compute frame indices to grab
    t = start_sec
    target_times = []
    while end_sec is None or t <= end_sec:
        target_times.append(t)
        t += every_sec
        if duration_sec is not None and t > duration_sec:
            break

    records = []
    saved = 0

    for t in target_times:
        # Convert time to nearest frame index
        frame_idx = int(round(t * fps))
        # Clamp
        if frame_idx < 0:
            frame_idx = 0
        if total_frames and frame_idx >= total_frames:
            frame_idx = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            # try advancing one frame (helps at some GOP boundaries)
            ok2, frame2 = cap.read()
            if not ok2 or frame2 is None:
                print(f"[warn] could not read frame at ~{t:.2f}s (idx {frame_idx})")
                continue
            frame = frame2

        h, w = frame.shape[:2]
        ts_sec = frame_idx / fps

        fname = f"frame_{frame_idx:08d}.jpg"
        fpath = out_frames_dir / fname

        # Write JPEG (quality ~95)
        cv2.imwrite(str(fpath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        records.append({
            "filename": fname,
            "width": int(w),
            "height": int(h),
            "frame_idx": int(frame_idx),
            "timestamp_sec": round(ts_sec, 3)
        })
        saved += 1

    cap.release()
    df = pd.DataFrame.from_records(records, columns=["filename", "width", "height", "frame_idx", "timestamp_sec"])
    return df

def main():
    ap = argparse.ArgumentParser(description="Download a YouTube video, extract frames every N seconds, and write a manifest CSV with image dimensions.")
    ap.add_argument("--url", type=str, required=True, help="YouTube video URL")
    ap.add_argument("--workdir", type=Path, default=Path("dataset_source"), help="Working directory (video + frames go here)")
    ap.add_argument("--every-sec", type=float, default=1.0, help="Extract one frame every N seconds (default: 1.0)")
    ap.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds (default: 0)")
    ap.add_argument("--end-sec", type=float, default=None, help="End time in seconds (default: full video)")
    ap.add_argument("--frames-subdir", type=str, default="frames", help="Subfolder name for extracted frames (default: frames)")
    ap.add_argument("--manifest-name", type=str, default="frames_manifest.csv", help="CSV manifest filename (default: frames_manifest.csv)")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    video_path = download_youtube(args.url, args.workdir)

    frames_dir = args.workdir / args.frames_subdir
    df = extract_frames(
        video_path=video_path,
        out_frames_dir=frames_dir,
        every_sec=args.every_sec,
        start_sec=args.start_sec,
        end_sec=args.end_sec
    )

    manifest_path = args.workdir / args.manifest_name
    df.to_csv(manifest_path, index=False)
    print(f"\nSaved {len(df)} frames to: {frames_dir}")
    print(f"Manifest CSV: {manifest_path}")
    if not df.empty:
        print("\nSample rows:")
        print(df.head().to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
