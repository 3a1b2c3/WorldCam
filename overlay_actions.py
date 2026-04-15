"""
Overlay recorded keyboard + mouse actions onto a gameplay video.

Usage:
    python overlay_actions.py --video <video.mp4> --log <input.txt> --output <overlay.mp4>

The input log format (one event per line):
    [timestamp] KeyDown: key_name
    [timestamp] KeyUp: key_name
    [timestamp] MouseMoveAbsolute: (dx, dy)

Keys tracked: W, A, S, D, Space, Shift
Mouse: relative dx/dy accumulated per frame, shown as a cursor on a circular pad.
"""

import argparse
import cv2
import numpy as np
import re
from collections import defaultdict


def parse_log(log_path):
    """Parse a raw input log file into a sorted list of (timestamp, event_type, detail)."""
    events = []
    with open(log_path, "r") as f:
        for line in f:
            match = re.match(
                r"\[(\d+\.\d+)\] (KeyDown|KeyUp|MouseMoveRelative|MouseMoveAbsolute): (.+)",
                line,
            )
            if match:
                timestamp = float(match.group(1))
                event_type = match.group(2)
                detail = match.group(3).strip()

                if event_type in ["KeyDown", "KeyUp"]:
                    detail = detail.strip("'").lower()
                    if detail == "key.space":
                        detail = "space"
                    elif detail == "key.shift":
                        detail = "shift"

                events.append((timestamp, event_type, detail))
    events.sort()
    return events


def build_timeline(events, fps):
    """Convert raw events into a per-frame timeline of key states and mouse deltas."""
    time_step = 1.0 / fps
    all_keys = ["w", "a", "s", "d", "space", "shift"]
    keyboard_state = defaultdict(bool)
    mouse_dx, mouse_dy = 0, 0
    event_index = 0
    current_time = 0.0
    max_time = events[-1][0] if events else 0.0
    timeline = []

    while current_time <= max_time + time_step:
        while (
            event_index < len(events)
            and current_time <= events[event_index][0] < current_time + time_step
        ):
            timestamp, event_type, detail = events[event_index]
            if event_type == "KeyDown":
                keyboard_state[detail] = True
            elif event_type == "KeyUp":
                keyboard_state[detail] = False
            elif event_type in ["MouseMoveAbsolute", "MouseMoveRelative"]:
                m = re.match(r"\((-?\d+),\s*(-?\d+)\)", detail)
                if m:
                    mouse_dx += int(m.group(1))
                    mouse_dy += int(m.group(2))
            event_index += 1

        timeline.append(
            {
                "time": round(current_time, 3),
                "mouse_dx": mouse_dx,
                "mouse_dy": mouse_dy,
                "keys": {k: keyboard_state.get(k, False) for k in all_keys},
            }
        )
        mouse_dx, mouse_dy = 0, 0
        current_time += time_step

    return timeline


def overlay_video(video_path, timeline, output_path, max_frames=None):
    """Draw keyboard + mouse overlay on each frame and write to output."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Keyboard layout
    all_keys = ["w", "a", "s", "d", "space", "shift"]
    box = 80
    margin_y = 100
    base_y = height - margin_y - box - 40
    left_x = 200
    key_positions = {
        "shift": (left_x - (3 * box + 10), base_y),
        "w": (left_x + box, base_y - box - 10),
        "a": (left_x, base_y),
        "s": (left_x + box, base_y),
        "d": (left_x + 2 * box, base_y),
        "space": (left_x, base_y + box + 10),
    }

    # Mouse pad
    pad_cx, pad_cy = width - 200, height - 200
    pad_radius = 165

    frame_index = 0
    while cap.isOpened() and frame_index < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        idx = min(frame_index, len(timeline) - 1)
        state = timeline[idx]

        # Draw keys
        for key in all_keys:
            x, y = key_positions[key]
            pressed = state["keys"].get(key, False)
            color = (0, 255, 0) if pressed else (100, 100, 100)

            if key in ["space", "shift"]:
                cv2.rectangle(frame, (x, y), (x + 3 * box, y + box), color, -1)
                cv2.putText(
                    frame, key.upper(), (x + box, y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3,
                )
            else:
                cv2.rectangle(frame, (x, y), (x + box, y + box), color, -1)
                cv2.putText(
                    frame, key.upper(), (x + 20, y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3,
                )

        # Draw mouse pad + cursor
        dx, dy = state["mouse_dx"], state["mouse_dy"]
        cursor_x = int(np.clip(pad_cx + dx * 5, pad_cx - pad_radius, pad_cx + pad_radius))
        cursor_y = int(np.clip(pad_cy + dy * 5, pad_cy - pad_radius, pad_cy + pad_radius))

        cv2.circle(frame, (pad_cx, pad_cy), pad_radius, (100, 100, 100), -1)
        cv2.line(frame, (pad_cx, pad_cy), (cursor_x, cursor_y), (0, 0, 0), 2)
        cv2.circle(frame, (pad_cx, pad_cy), 5, (0, 0, 0), -1)
        cv2.circle(frame, (cursor_x, cursor_y), 7, (0, 255, 0), -1)

        out.write(frame)
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"  Frame {frame_index}/{total_frames}", end="\r")

    cap.release()
    out.release()
    print(f"\nSaved overlay video -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Overlay keyboard/mouse actions on a gameplay video.")
    parser.add_argument("--video", required=True, help="Path to the input .mp4 video.")
    parser.add_argument("--log", required=True, help="Path to the input .txt action log.")
    parser.add_argument("--output", default="overlay_output.mp4", help="Path to the output .mp4.")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (default: all).")
    args = parser.parse_args()

    print(f"Parsing log: {args.log}")
    events = parse_log(args.log)
    print(f"  {len(events)} events parsed")

    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)
    print(f"Building timeline at {fps:.1f} fps...")
    timeline = build_timeline(events, fps)
    print(f"  {len(timeline)} frames in timeline")

    print(f"Overlaying on video: {args.video}")
    overlay_video(args.video, timeline, args.output, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
