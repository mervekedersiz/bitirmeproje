import argparse
import csv
from pathlib import Path
import tempfile

import cv2
import numpy as np


DROP_HEIGHT_CM = 150.0
POSITION_FILTER_FRAMES = 5


def detect_ball_centers(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError("Video does not contain a valid frame rate.")

    centers = []
    kernel = np.ones((3, 3), np.uint8)
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array([20, 100, 90], np.uint8), np.array([45, 255, 255], np.uint8)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

        candidates = []
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            aspect = width / max(height, 1)
            if 80 <= area <= 2500 and 0.4 <= aspect <= 2.5:
                candidates.append((area, centroids[label]))

        if not candidates:
            centers.append((np.nan, np.nan))
        else:
            _, center = max(candidates, key=lambda candidate: candidate[0])
            centers.append(center)

    capture.release()
    if np.count_nonzero(np.isfinite(centers).all(axis=1)) < 5:
        raise RuntimeError("Could not detect enough ball positions to analyze the video.")
    return fps, np.asarray(centers, dtype=float)


def smooth_pixel_positions(values, window_size=POSITION_FILTER_FRAMES):
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("Position filter window must be an odd number of at least 3.")
    if len(values) < window_size:
        raise ValueError(
            f"Need at least {window_size} detected frames to smooth positions."
        )

    radius = window_size // 2
    smoothed = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = min(max(index - radius, 0), len(values) - window_size)
        sample_indices = np.arange(start, start + window_size, dtype=float)
        relative_indices = sample_indices - index
        coefficients = np.polyfit(
            relative_indices, values[start:start + window_size], deg=2
        )
        smoothed[index] = np.polyval(coefficients, 0.0)
    return smoothed


def find_release_frame(pixel_y):
    frame_deltas = np.diff(pixel_y)
    for index in range(len(frame_deltas) - 1):
        if frame_deltas[index] >= 8 and frame_deltas[index + 1] >= 8:
            return index
    return min(int(np.argmin(pixel_y)) + 1, len(pixel_y) - 2)


def draw_graph(
    path, title, x_label, y_label, times, values, color,
    comparison_values=None, comparison_label="Raw data", value_label="Filtered data"
):
    width, height = 1200, 760
    left, right, top, bottom = 110, 35, 85, 100
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom

    x_min, x_max = float(np.min(times)), float(np.max(times))
    plotted_values = (
        np.concatenate((values, comparison_values))
        if comparison_values is not None
        else values
    )
    y_min, y_max = float(np.min(plotted_values)), float(np.max(plotted_values))
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding

    def px_x(value):
        return int(x0 + (value - x_min) / max(x_max - x_min, 1e-12) * (x1 - x0))

    def px_y(value):
        return int(y1 - (value - y_min) / (y_max - y_min) * (y1 - y0))

    cv2.putText(image, title, (left, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2)
    for fraction in np.linspace(0, 1, 6):
        y_value = y_min + fraction * (y_max - y_min)
        y = px_y(y_value)
        cv2.line(image, (x0, y), (x1, y), (225, 225, 225), 1)
        cv2.putText(
            image, f"{y_value:.1f}", (12, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (65, 65, 65), 1
        )
    for fraction in np.linspace(0, 1, 7):
        x_value = x_min + fraction * (x_max - x_min)
        x = px_x(x_value)
        cv2.line(image, (x, y0), (x, y1), (235, 235, 235), 1)
        cv2.putText(
            image, f"{x_value:.2f}", (x - 22, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (65, 65, 65), 1
        )

    cv2.line(image, (x0, y0), (x0, y1), (40, 40, 40), 2)
    cv2.line(image, (x0, y1), (x1, y1), (40, 40, 40), 2)
    if comparison_values is not None:
        comparison_points = np.array(
            [
                (px_x(float(t)), px_y(float(value)))
                for t, value in zip(times, comparison_values)
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(image, [comparison_points], False, (140, 140, 140), 1, cv2.LINE_AA)
        for point in comparison_points[:, 0, :]:
            cv2.circle(image, tuple(point), 2, (140, 140, 140), -1, cv2.LINE_AA)
        cv2.line(image, (left, 68), (left + 24, 68), (140, 140, 140), 2)
        cv2.putText(
            image, comparison_label, (left + 32, 74),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1
        )
        legend_x = left + 32 + cv2.getTextSize(
            comparison_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )[0][0] + 24
        cv2.line(image, (legend_x, 68), (legend_x + 24, 68), color, 3)
        cv2.putText(
            image, value_label, (legend_x + 32, 74),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1
        )

    points = np.array(
        [(px_x(float(t)), px_y(float(value))) for t, value in zip(times, values)],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(image, [points], False, color, 3, cv2.LINE_AA)
    for point in points[:, 0, :]:
        cv2.circle(image, tuple(point), 4, color, -1, cv2.LINE_AA)

    cv2.putText(
        image, x_label, ((x0 + x1) // 2 - 45, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2
    )
    cv2.putText(
        image, y_label, (12, top - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1
    )
    encoded, buffer = cv2.imencode(path.suffix, image)
    if not encoded:
        raise RuntimeError(f"Could not encode graph: {path}")
    path.write_bytes(buffer.tobytes())


def create_detection_video(video_path, output_path, centers, fps, release_frame, impact_frame):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video for annotated output: {video_path}")

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("Could not read the first frame for annotated output.")
    height, width = first_frame.shape[:2]

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Could not initialize MP4 writer for annotated video.")

    frame_index = 0
    try:
        while ok:
            center_is_detected = np.isfinite(centers[frame_index]).all()
            if center_is_detected:
                center = tuple(np.rint(centers[frame_index]).astype(int))
                if frame_index > 0:
                    visible_trail = centers[:frame_index + 1]
                    visible_trail = visible_trail[np.isfinite(visible_trail).all(axis=1)]
                    trail = np.rint(visible_trail).astype(np.int32).reshape((-1, 1, 2))
                    if len(trail) > 1:
                        cv2.polylines(
                            first_frame, [trail], False, (255, 100, 0), 3, cv2.LINE_AA
                        )
                cv2.circle(first_frame, center, 23, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.drawMarker(
                    first_frame, center, (0, 0, 255), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA
                )
                cv2.putText(
                    first_frame, f"DETECTED BALL CENTER: ({center[0]}, {center[1]}) px",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
                )
            else:
                cv2.putText(
                    first_frame, "BALL NOT DETECTED IN THIS FRAME",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
                )
            cv2.putText(
                first_frame, f"Frame {frame_index} | {frame_index / fps:.3f} s",
                (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                first_frame, "HSV yellow/green detection | green circle: ball",
                (16, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )
            if frame_index == release_frame:
                cv2.putText(
                    first_frame, "RELEASE / START", (16, 124),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2, cv2.LINE_AA
                )
            if frame_index == impact_frame:
                cv2.putText(
                    first_frame, "FIRST IMPACT", (16, 154),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
                )
            writer.write(first_frame)
            frame_index += 1
            ok, first_frame = capture.read()

        if frame_index != len(centers):
            raise RuntimeError(
                f"Annotated video frame count mismatch: expected {len(centers)}, "
                f"processed {frame_index}."
            )
    finally:
        capture.release()
        writer.release()

    if not temp_path.exists() or temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Annotated video was not created.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(temp_path.read_bytes())
    temp_path.unlink()


def analyze(video_path, output_dir):
    fps, centers = detect_ball_centers(video_path)
    pixel_y = centers[:, 1]

    release_frame = find_release_frame(pixel_y)
    impact_frame = None
    for index in range(release_frame + 1, len(pixel_y) - 1):
        if pixel_y[index] >= pixel_y[index - 1] and pixel_y[index] > pixel_y[index + 1]:
            impact_frame = index
            break
    if impact_frame is None:
        raise RuntimeError("Could not identify the first impact/bounce in the video.")

    frame_numbers = np.arange(release_frame, impact_frame + 1)
    raw_y = pixel_y[release_frame:impact_frame + 1]
    pixel_drop = float(pixel_y[impact_frame] - pixel_y[release_frame])
    if pixel_drop <= 0:
        raise RuntimeError("Detected impact is not below the release point.")

    filtered_y = smooth_pixel_positions(raw_y)
    filtered_y[0] = raw_y[0]
    filtered_y[-1] = raw_y[-1]
    cm_per_pixel = DROP_HEIGHT_CM / pixel_drop
    raw_heights = DROP_HEIGHT_CM - (raw_y - raw_y[0]) * cm_per_pixel
    heights = DROP_HEIGHT_CM - (filtered_y - pixel_y[release_frame]) * cm_per_pixel
    raw_heights[0] = DROP_HEIGHT_CM
    raw_heights[-1] = 0.0
    heights[0] = DROP_HEIGHT_CM
    heights[-1] = 0.0
    dt = 1.0 / fps
    raw_all_velocities = np.diff(raw_heights) / dt
    all_velocities = np.diff(heights) / dt
    # The final frame interval contains the ground collision, not free-fall motion.
    raw_velocities = raw_all_velocities[:-1]
    velocities = all_velocities[:-1]
    velocity_times = (frame_numbers[:-2] + 0.5 - release_frame) / fps
    n = len(velocities)
    raw_accelerations = (
        raw_velocities[1:n - 2] - raw_velocities[0:n - 3]
    ) / dt
    accelerations = (velocities[1:n - 2] - velocities[0:n - 3]) / dt
    acceleration_times = (
        velocity_times[1:n - 2] + velocity_times[0:n - 3]
    ) / 2
    relative_times = (frame_numbers - release_frame) / fps
    video_times = frame_numbers / fps

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "video_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "video_time_s", "time_since_release_s",
            "ball_center_x_px", "ball_center_y_px", "filtered_center_y_px",
            "raw_height_cm", "height_cm", "raw_velocity_cm_s", "velocity_cm_s",
            "velocity_time_s", "raw_acceleration_cm_s2", "acceleration_cm_s2",
            "acceleration_time_s",
        ])
        for index, frame in enumerate(frame_numbers):
            writer.writerow([
                int(frame),
                f"{video_times[index]:.6f}",
                f"{relative_times[index]:.6f}",
                f"{centers[frame, 0]:.3f}",
                f"{pixel_y[frame]:.3f}",
                f"{filtered_y[index]:.3f}",
                f"{raw_heights[index]:.3f}",
                f"{heights[index]:.3f}",
                f"{raw_velocities[index]:.3f}" if index < len(raw_velocities) else "",
                f"{velocities[index]:.3f}" if index < len(velocities) else "",
                f"{velocity_times[index]:.6f}" if index < len(velocity_times) else "",
                f"{raw_accelerations[index]:.3f}" if index < len(raw_accelerations) else "",
                f"{accelerations[index]:.3f}" if index < len(accelerations) else "",
                f"{acceleration_times[index]:.6f}" if index < len(acceleration_times) else "",
            ])

    draw_graph(
        output_dir / "height_time.png", "Height vs time (raw and filtered)",
        "Time (s)", "Height (cm)", relative_times, heights, (45, 140, 30),
        comparison_values=raw_heights, comparison_label="Raw height",
        value_label="Filtered height (used)"
    )
    draw_graph(
        output_dir / "velocity_time.png", "Vertical velocity vs time", "Time (s)",
        "Velocity (cm/s)", velocity_times, velocities, (210, 95, 25)
    )
    draw_graph(
        output_dir / "acceleration_time.png", "Filtered vertical acceleration",
        "Time (s)", "Acceleration (m/s^2)", acceleration_times,
        accelerations / 100.0, (40, 70, 210)
    )
    draw_graph(
        output_dir / "acceleration_comparison.png",
        "Raw and filtered vertical acceleration", "Time (s)",
        "Acceleration (m/s^2)", acceleration_times, accelerations / 100.0,
        (40, 70, 210), comparison_values=raw_accelerations / 100.0,
        comparison_label="Raw acceleration", value_label="Filtered acceleration"
    )
    detection_video = output_dir / "video_detection.mp4"
    create_detection_video(
        video_path, detection_video, centers, fps, release_frame, impact_frame
    )

    print(f"Video: {video_path}")
    print(f"Frame rate: {fps:.3f} fps")
    print(f"Release frame: {release_frame} ({release_frame / fps:.3f} s)")
    print(f"Impact frame: {impact_frame} ({impact_frame / fps:.3f} s)")
    print(f"Calibration: {pixel_drop:.2f} px = {DROP_HEIGHT_CM:.1f} cm "
          f"({cm_per_pixel:.4f} cm/px)")
    print(
        f"Position smoothing: local quadratic fit over {POSITION_FILTER_FRAMES} "
        f"frames ({POSITION_FILTER_FRAMES / fps:.3f} s); velocity/acceleration "
        "still use the requested array differences."
    )
    print(f"Mean finite-difference acceleration: {np.mean(accelerations):.2f} cm/s^2")
    print(
        "Acceleration standard deviation (frame-to-frame measurement variation): "
        f"{np.std(accelerations):.2f} cm/s^2"
    )
    print(f"CSV and graphs saved in: {output_dir}")
    print(f"Annotated ball tracking video: {detection_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track a falling ball, calibrate a 150 cm drop, and plot its motion."
    )
    parser.add_argument(
        "video", nargs="?", type=Path, default=Path(__file__).with_name("video1.mp4")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent
    )
    args = parser.parse_args()
    analyze(args.video, args.output_dir)
