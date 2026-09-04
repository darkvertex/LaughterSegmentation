import json
import os
import shutil
from pathlib import Path as LocalPath

from cog import BasePredictor, Input, Path

from inference import main as run_inference

HF_REPO_ID = "omine-me/LaughterSegmentation"
HF_WEIGHTS_FILE = "model.safetensors"
AUDIO_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
SAMPLE_RATE = 16000


def _local_weights_path() -> str:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return env_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", HF_WEIGHTS_FILE)


def _ensure_weights(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path

    from huggingface_hub import hf_hub_download

    dest_dir = os.path.dirname(os.path.abspath(model_path))
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    cached = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_WEIGHTS_FILE)
    if os.path.abspath(cached) != os.path.abspath(model_path):
        shutil.copy2(cached, model_path)
    return model_path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Download weights if needed and load the model once per worker."""
        import safetensors
        import torch
        from train.model import Model

        self.default_model_path = _ensure_weights(_local_weights_path())
        self.default_output_dir = "/tmp/laughter-output"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # The fine-tuned checkpoint contains every parameter, so skip the base-weight download.
        self.model = Model(AUDIO_MODEL_NAME, device, SAMPLE_RATE, pretrained=False).to(device)
        state_dict = safetensors.torch.load_file(
            self.default_model_path,
            device.index if device.type == "cuda" else "cpu",
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

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
            model=self.model,
        )

        output_json = LocalPath(self.default_output_dir) / f"{input_path.stem}.json"
        if not output_json.exists():
            raise RuntimeError("Inference completed but output JSON was not created.")

        # Validate JSON before returning so Replicate users get clear failures.
        with output_json.open("r", encoding="utf-8") as f:
            json.load(f)

        return Path(str(output_json))
