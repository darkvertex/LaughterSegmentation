import json
import os
import shutil
from pathlib import Path as LocalPath

from cog import BasePredictor, Input, Path

from inference import main as run_inference


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load lightweight configuration once when the model container starts."""
        self.default_model_path = os.getenv("MODEL_PATH", "./models/model.safetensors")
        self.default_output_dir = "./output"

    def predict(
        self,
        audio: Path = Input(description="Input audio file (wav/mp3/opus, etc.)"),
        input_sec: float = Input(
            description="Window length in seconds used for inference", default=7.0, ge=2.1, le=30.0
        ),
        batch_size: int = Input(
            description="Number of windows per forward pass", default=10, ge=1, le=64
        ),
    ) -> Path:
        """Run laughter segmentation and return a JSON file with time segments."""
        if not os.path.exists(self.default_model_path):
            raise FileNotFoundError(
                "Model file not found at "
                f"{self.default_model_path}. Ensure model.safetensors is available in the container."
            )

        os.makedirs(self.default_output_dir, exist_ok=True)

        # Copy input to a local tmp path with stable naming.
        input_path = LocalPath("/tmp/input_audio")
        suffix = LocalPath(str(audio)).suffix or ".wav"
        input_path = input_path.with_suffix(suffix)
        shutil.copy(str(audio), str(input_path))

        run_inference(
            audio_path=str(input_path),
            output_dir=self.default_output_dir,
            model_path=self.default_model_path,
            input_sec=float(input_sec),
            batch_size=int(batch_size),
        )

        output_json = LocalPath(self.default_output_dir) / f"{input_path.stem}.json"
        if not output_json.exists():
            raise RuntimeError("Inference completed but output JSON was not created.")

        # Validate JSON before returning so Replicate users get clear failures.
        with output_json.open("r", encoding="utf-8") as f:
            json.load(f)

        return Path(str(output_json))
