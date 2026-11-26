
from pathlib import Path
import csv
import cv2

# ==========================
# CONFIGURATION
# ==========================

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "videos"
DATASET_DIR = ROOT / "dataset"

SAMPLE_INTERVAL = 1.0  # Extract 1 image per second
MAX_WIDTH = 1920
EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Session 5 is reserved exclusively for testing.
# For the other sessions, reserve the last video
# for validation.
SESSION_SPLITS = {
    "session1_daylight": {"valid": 1},
    "session2_warm_light": {"valid": 1},
    "session3_dim_light": {"valid": 1},
    "session4_background": {"valid": 1},
    "session5_test": {"test": "all"},
}


def get_split(session, index, total):
    if session == "session5_test":
        return "test"

    if index == total - 1:
        return "valid"

    return "train"


def resize_frame(frame):
    height, width = frame.shape[:2]

    if width <= MAX_WIDTH:
        return frame

    scale = MAX_WIDTH / width
    new_height = round(height * scale)

    return cv2.resize(
        frame,
        (MAX_WIDTH, new_height),
        interpolation=cv2.INTER_AREA
    )


def extract_video(video_path, session, split):
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        print(f"ERROR: Cannot open {video_path.name}")
        return []

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0 or total_frames <= 0:
        print(f"ERROR: Invalid metadata in {video_path.name}")
        video.release()
        return []

    duration = total_frames / fps
    output_dir = DATASET_DIR / split / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    second = 0

    while second < duration:
        # Seek to the requested timestamp
        video.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        success, frame = video.read()

        if not success:
            print(f"Warning: Could not read {video_path.name} at {second}s")
            second += SAMPLE_INTERVAL
            continue

        frame = resize_frame(frame)

        filename = (
            f"{session}_{video_path.stem}_"
            f"{int(second * 1000):06d}ms.jpg"
        )

        output_path = output_dir / filename

        saved = cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if saved:
            records.append({
                "session": session,
                "video": video_path.name,
                "split": split,
                "timestamp_seconds": second,
                "image": filename
            })

        second += SAMPLE_INTERVAL

    video.release()
    return records


def main():
    all_records = []

    print("\n=== SMART TABLE: FRAME EXTRACTION ===\n")

    for session in SESSION_SPLITS:
        session_dir = VIDEOS_DIR / session

        if not session_dir.exists():
            print(f"Missing folder: {session}")
            continue

        videos = sorted(
            path for path in session_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in EXTENSIONS
        )

        print(f"\n{session}: {len(videos)} videos")

        for index, video_path in enumerate(videos):
            split = get_split(session, index, len(videos))

            print(f"  Processing {video_path.name} -> {split}")

            records = extract_video(video_path, session, split)
            all_records.extend(records)

            print(f"    Extracted {len(records)} images")

    # Save the image provenance log
    log_path = ROOT / "docs" / "frame_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "session",
                "video",
                "split",
                "timestamp_seconds",
                "image"
            ]
        )
        writer.writeheader()
        writer.writerows(all_records)

    print("\n========== SUMMARY ==========")

    for split in ("train", "valid", "test"):
        count = sum(r["split"] == split for r in all_records)
        print(f"{split}: {count} images")

    print(f"Total: {len(all_records)} images")
    print(f"Log saved to: {log_path}")
    print("=============================\n")


if __name__ == "__main__":
    main()
