import json
import os
from pathlib import Path as FsPath

from cog import BasePredictor, Input, Path as CogPath

from inference import (
    DEFAULT_MODEL_PATH,
    ensure_model_weights,
    load_model,
    segment_laughter,
    write_segments,
)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Download weights if needed and load the model once per container."""
        self.model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        self.model_path = ensure_model_weights(self.model_path)
        self.model = load_model(self.model_path)

    def predict(
        self,
        audio: CogPath = Input(description="Input audio file (wav/mp3/opus, etc.)"),
        input_sec: float = Input(
            description="Window length in seconds used for inference",
            default=7.0,
            ge=2.1,
            le=30.0,
        ),
        batch_size: int = Input(
            description="Number of windows per forward pass",
            default=10,
            ge=1,
            le=64,
        ),
    ) -> CogPath:
        """Run laughter segmentation and return a JSON file with time segments."""
        audio_path = FsPath(str(audio))
        stem = audio_path.stem or "audio"
        output_json = FsPath("/tmp") / f"{stem}.json"

        laughter = segment_laughter(
            self.model,
            str(audio_path),
            input_sec=float(input_sec),
            batch_size=int(batch_size),
        )
        write_segments(laughter, str(output_json))

        # Validate JSON before returning so Replicate users get clear failures.
        with output_json.open("r", encoding="utf-8") as f:
            json.load(f)

        return CogPath(str(output_json))
