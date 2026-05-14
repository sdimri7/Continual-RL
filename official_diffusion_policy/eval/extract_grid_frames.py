import cv2
import os
import numpy as np
import argparse

def extract_frames(video_path, output_dir, grid_rows=3, grid_cols=4, target_row=0, target_col=0, num_frames=5):
    """
    Extracts evenly spaced frames from a video that contains a grid of parallel runs,
    and crops the output to only show a specific grid cell.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: Video has 0 frames.")
        return

    # Calculate 5 evenly spaced frame indices from start to end
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate dimensions of a single cell in the grid
    cell_w = width // grid_cols
    cell_h = height // grid_rows

    # Verify the target is within bounds
    if target_row >= grid_rows or target_col >= grid_cols:
        print(f"Warning: Target ({target_row},{target_col}) is outside the {grid_rows}x{grid_cols} grid.")
        print("Note: Indices are 0-indexed (e.g., for 3x4, rows are 0-2, cols are 0-3).")

    # Calculate crop coordinates for the target cell
    x_start = target_col * cell_w
    x_end = (target_col + 1) * cell_w
    y_start = target_row * cell_h
    y_end = (target_row + 1) * cell_h

    # Ensure within bounds strictly
    x_start, x_end = max(0, x_start), min(width, x_end)
    y_start, y_end = max(0, y_start), min(height, y_end)

    print(f"Video size: {width}x{height} | Total Frames: {total_frames}")
    print(f"Targeting grid cell ({target_row}, {target_col}) -> Crop Coordinates: [{x_start}:{x_end}, {y_start}:{y_end}]")

    for i, f_idx in enumerate(frame_indices):
        # Set video to specific frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at index {f_idx}")
            continue

        # Crop the frame to the target cell
        cropped_frame = frame[y_start:y_end, x_start:x_end]

        # Save output
        out_filename = f"grid_{target_row}_{target_col}_frame_{i+1}_step_{f_idx}.jpg"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, cropped_frame)
        print(f"Saved {i+1}/{num_frames}: {out_path}")

    cap.release()
    print(f"\nDone! Extracted {num_frames} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specific frames from a grid video (e.g., 3x4 environments).")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out_dir", default="extracted_run", help="Directory to save extracted frames")
    parser.add_argument("--rows", type=int, default=3, help="Total rows in the video grid (default: 3)")
    parser.add_argument("--cols", type=int, default=4, help="Total columns in the video grid (default: 4)")
    parser.add_argument("--target_row", type=int, default=0, help="Target row index (0-indexed, default: 0)")
    parser.add_argument("--target_col", type=int, default=0, help="Target column index (0-indexed, default: 0)")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of evenly spaced frames to extract (default: 5)")

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.out_dir,
        grid_rows=args.rows,
        grid_cols=args.cols,
        target_row=args.target_row,
        target_col=args.target_col,
        num_frames=args.num_frames
    )
