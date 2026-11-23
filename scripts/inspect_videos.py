
from pathlib import Path
import cv2
import math

# Project root (one level above the scripts folder)
PROJECT_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_DIR / "videos"

EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

total_videos = 0
total_duration = 0
total_images = 0

print("\n=== SMART TABLE DETECTION: VIDEO INSPECTION ===\n")

for video_path in sorted(VIDEOS_DIR.rglob("*")):

    if not video_path.is_file():
        continue

    if video_path.suffix.lower() not in EXTENSIONS:
        continue

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        print(f"ERROR: Cannot open {video_path.name}")
        continue

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or frame_count <= 0:
        print(f"ERROR: Invalid video metadata: {video_path.name}")
        video.release()
        continue

    duration = frame_count / fps
    images = math.ceil(duration)

    total_videos += 1
    total_duration += duration
    total_images += images

    print(f"Video: {video_path.relative_to(VIDEOS_DIR)}")
    print(f"  Resolution: {width} x {height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Estimated images (1 FPS): {images}")
    print()

    video.release()

print("========== SUMMARY ==========")
print(f"Total videos: {total_videos}")
print(f"Total duration: {total_duration / 60:.2f} minutes")
print(f"Estimated images at 1 FPS: {total_images}")
print("=============================\n")
