import shutil
from pathlib import Path as LocalPath

from cog import BasePredictor, Input, Path

from serving import load_model, segment_audio


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Download weights if needed and load the model once per worker."""
        self.model, self.default_model_path = load_model()
        self.default_output_dir = "/tmp/laughter-output"

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
        """Run laughter and applause segmentation; return JSON with both timed-range maps."""
        # Copy input to a local tmp path with stable naming.
        input_path = LocalPath("/tmp/input_audio")
        suffix = LocalPath(str(audio)).suffix or ".wav"
        input_path = input_path.with_suffix(suffix)
        shutil.copy(str(audio), str(input_path))

        segment_audio(
            audio_path=str(input_path),
            model=self.model,
            model_path=self.default_model_path,
            output_dir=self.default_output_dir,
            input_sec=float(input_sec),
            batch_size=int(batch_size),
        )

        output_json = LocalPath(self.default_output_dir) / f"{input_path.stem}.json"
        return Path(str(output_json))
