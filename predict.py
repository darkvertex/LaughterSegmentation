import json
import os
import shutil
from pathlib import Path as LocalPath
from typing import Optional

from cog import BasePredictor, Input, Path

from inference import main as run_inference
from laughter_levels import LevelsConfig, process_laughter_levels, resolve_markers_source

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
        markers: Optional[Path] = Input(
            description=(
                "Optional analysis JSON file with laughter/applause maps. "
                "When set, skip segmentation and process audio levels instead."
            ),
            default=None,
        ),
        markers_json: str = Input(
            description=(
                "Optional analysis JSON string (same schema as the segmentation output). "
                "Provide this or markers, not both."
            ),
            default="",
        ),
        process_applause: bool = Input(
            description="Also level applause ranges (default: laughter only)",
            default=False,
        ),
        output_format: str = Input(
            description="Processed-audio format when markers are provided",
            default="wav",
            choices=["wav", "flac", "mp3"],
        ),
        pre_lift_db: float = Input(
            description="Gain (dB) applied just before each laugh onset",
            default=1.5,
            ge=0.0,
            le=12.0,
        ),
        pre_lift_ms: float = Input(
            description="How far before each onset the pre-lift starts (ms)",
            default=280.0,
            ge=0.0,
            le=2000.0,
        ),
        pre_lift_attack_ms: float = Input(
            description="Pre-lift cosine ramp duration (ms)",
            default=100.0,
            ge=1.0,
            le=1000.0,
        ),
        comp_threshold_db: float = Input(
            description="Laughter compressor threshold (dB)",
            default=-22.0,
            ge=-60.0,
            le=0.0,
        ),
        comp_ratio: float = Input(
            description="Laughter compressor ratio",
            default=2.5,
            ge=1.0,
            le=20.0,
        ),
        comp_attack_ms: float = Input(
            description="Laughter compressor attack (ms)",
            default=10.0,
            ge=0.1,
            le=1000.0,
        ),
        comp_release_ms: float = Input(
            description="Laughter compressor release (ms)",
            default=200.0,
            ge=1.0,
            le=5000.0,
        ),
        norm_target_rms_db: float = Input(
            description="Target RMS (dBFS) for each laugh window",
            default=-18.0,
            ge=-40.0,
            le=0.0,
        ),
        norm_max_gain_db: float = Input(
            description="Max makeup gain (dB) when matching laugh RMS",
            default=6.0,
            ge=0.0,
            le=24.0,
        ),
        norm_max_cut_db: float = Input(
            description="Max attenuation (dB) when matching laugh RMS",
            default=12.0,
            ge=0.0,
            le=24.0,
        ),
        norm_attack_ms: float = Input(
            description="Normalizer ramp-in (ms)",
            default=50.0,
            ge=1.0,
            le=1000.0,
        ),
        norm_release_ms: float = Input(
            description="Normalizer ramp-out (ms)",
            default=250.0,
            ge=1.0,
            le=5000.0,
        ),
        limiter_ceiling_db: float = Input(
            description="Full-file limiter ceiling (dBFS)",
            default=-1.0,
            ge=-12.0,
            le=-0.1,
        ),
        limiter_release_ms: float = Input(
            description="Limiter release (ms)",
            default=80.0,
            ge=1.0,
            le=1000.0,
        ),
    ) -> Path:
        """Segment laughter/applause, or level audio when markers are provided."""
        os.makedirs(self.default_output_dir, exist_ok=True)

        # Copy input to a local tmp path with stable naming.
        input_path = LocalPath("/tmp/input_audio")
        suffix = LocalPath(str(audio)).suffix or ".wav"
        input_path = input_path.with_suffix(suffix)
        shutil.copy(str(audio), str(input_path))

        markers_payload = resolve_markers_source(markers, markers_json)
        if markers_payload is not None:
            config = LevelsConfig(
                process_applause=bool(process_applause),
                output_format=str(output_format),
                pre_lift_db=float(pre_lift_db),
                pre_lift_ms=float(pre_lift_ms),
                pre_lift_attack_ms=float(pre_lift_attack_ms),
                comp_threshold_db=float(comp_threshold_db),
                comp_ratio=float(comp_ratio),
                comp_attack_ms=float(comp_attack_ms),
                comp_release_ms=float(comp_release_ms),
                norm_target_rms_db=float(norm_target_rms_db),
                norm_max_gain_db=float(norm_max_gain_db),
                norm_max_cut_db=float(norm_max_cut_db),
                norm_attack_ms=float(norm_attack_ms),
                norm_release_ms=float(norm_release_ms),
                limiter_ceiling_db=float(limiter_ceiling_db),
                limiter_release_ms=float(limiter_release_ms),
            )
            output_audio = LocalPath(self.default_output_dir) / f"{input_path.stem}_leveled.{config.output_format}"
            process_laughter_levels(
                audio_path=str(input_path),
                markers=markers_payload,
                output_path=str(output_audio),
                config=config,
            )
            if not output_audio.exists():
                raise RuntimeError("Levels processing completed but output audio was not created.")
            return Path(str(output_audio))

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
