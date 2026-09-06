import os

import cv2
import numpy as np
from google import genai
from google.genai import types

from lerobot_agentic.cognitive.schemas import SpatialGroundingPlan


class CognitiveSupervisor:
    """
    High-level reasoning brain using Google Gemini Robotics API (gemini-robotics-er-2-preview).
    Extracts spatial affordances, generates bounding boxes, and plans sub-goals at target 0.5-2 Hz.
    """
    def __init__(self, api_key: str | None = None, model_name: str = "gemini-robotics-er-2-preview"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            # Check project root .env
            env_file = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
            if os.path.exists(env_file):
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("export "):
                            line = line[len("export "):].strip()
                        if line.startswith("GEMINI_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip().strip("\"").strip("\x27")
                            os.environ["GEMINI_API_KEY"] = self.api_key
                            break

        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.model_name = model_name

    def plan_and_ground(self, rgb_image: np.ndarray, natural_language_goal: str, max_retries: int = 2) -> SpatialGroundingPlan:
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY is not set. Cannot run remote CognitiveSupervisor.")

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        image_bytes = buffer.tobytes()

        prompt = f"""
        You are the cognitive perception brain for an embodied 6-DOF robotic manipulator arm.
        User Goal: "{natural_language_goal}"
        
        Examine the camera image. Identify:
        1. Target manipuland object to pick/interact with
        2. Receptacle or destination zone
        3. Immediate next sub-goal (strictly one of: reach, grasp, lift, transport, place, retreat, recover)
        4. Normalized bounding boxes [ymin, xmin, ymax, xmax] in range [0, 1000].
        5. Short operational decision note (e.g. 'target acquired', 'slip detected')
        
        Strictly adhere to the JSON schema.
        """

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SpatialGroundingPlan,
                        temperature=0.1
                    )
                )
                return SpatialGroundingPlan.model_validate_json(response.text)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < max_retries:
                    import time
                    time.sleep(0.5 * (2 ** attempt))

        raise RuntimeError(f"CognitiveSupervisor API call failed on {self.model_name} after {max_retries} retries: {last_error}")
