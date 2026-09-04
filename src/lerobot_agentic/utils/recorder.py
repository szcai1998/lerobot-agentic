import os
import pathlib
from typing import List, Optional
import numpy as np
import cv2
import imageio

class EpisodeVideoRecorder:
    """
    Records rollout frames and exports high-definition MP4/GIF videos
    with cognitive bounding box and sub-goal telemetry overlays.
    """
    def __init__(self, output_dir: str = "outputs/videos", fps: int = 25):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frames: List[np.ndarray] = []

    def add_frame(
        self,
        rgb_image: np.ndarray,
        sub_goal: Optional[str] = None,
        target_box_2d: Optional[List[int]] = None,
        step_idx: Optional[int] = None
    ):
        frame = rgb_image.copy()
        h, w, _ = frame.shape

        # Draw 2D bounding box if provided [ymin, xmin, ymax, xmax] in [0, 1000]
        if target_box_2d and len(target_box_2d) == 4:
            ymin, xmin, ymax, xmax = target_box_2d
            pt1 = (int(xmin * w / 1000.0), int(ymin * h / 1000.0))
            pt2 = (int(xmax * w / 1000.0), int(ymax * h / 1000.0))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 120), 2)
            cv2.putText(frame, "TARGET AFFORDANCE", (pt1[0], max(pt1[1] - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

        # Draw telemetry HUD overlay banner
        cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 25), -1)
        hud_text = f"STEP: {step_idx:03d} | GOAL: {sub_goal or 'IDLE'}"
        cv2.putText(frame, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 245), 1, cv2.LINE_AA)

        self.frames.append(frame)

    def save(self, filename: str = "rollout.mp4") -> str:
        if not self.frames:
            print("[Recorder] No frames to save.")
            return ""
        
        out_path = self.output_dir / filename
        imageio.mimsave(str(out_path), self.frames, fps=self.fps)
        print(f"[Recorder] Video saved ({len(self.frames)} frames) -> {out_path}")
        self.frames.clear()
        return str(out_path)
