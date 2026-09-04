import os
import io
from typing import Optional
import numpy as np
import cv2
from google import genai
from google.genai import types
from lerobot_agentic.cognitive.schemas import SpatialGroundingPlan

class CognitiveSupervisor:
    """
    High-level reasoning brain using Google Gemini Robotics API (gemini-robotics-er-2-preview or gemini-2.0-flash).
    Extracts spatial affordances, generates bounding boxes, and plans sub-goals at 1-2 Hz.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-robotics-er-2-preview"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.model_name = model_name

    def plan_and_ground(self, rgb_image: np.ndarray, natural_language_goal: str) -> SpatialGroundingPlan:
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
        3. Immediate next sub-goal (reach, grasp, lift, place)
        4. Normalized bounding boxes [ymin, xmin, ymax, xmax] in range [0, 1000].
        
        Strictly adhere to the JSON schema.
        """

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
        except Exception as e:
            # Fallback to gemini-2.0-flash
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
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
