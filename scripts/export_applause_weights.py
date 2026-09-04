"""Export AMP applause-binary SavedModel Dense weights to a NumPy archive.

Runtime inference uses the .npz file and does not import TensorFlow.
This script is a one-off converter; TensorFlow is not a project dependency.

Example:

    uv venv /tmp/tf-export --python 3.11
    uv pip install --python /tmp/tf-export/bin/python tensorflow-cpu==2.15.1
    /tmp/tf-export/bin/python scripts/export_applause_weights.py \\
        --saved-model /path/to/pretrained/applause-binary-20210203 \\
        --output models/applause_mlp.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf


EXPECTED_SHAPES = (
    ((40, 30), (30,)),
    ((30, 20), (20,)),
    ((20, 10), (10,)),
    ((10, 2), (2,)),
)
EXPECTED_ACTIVATIONS = ("sigmoid", "sigmoid", "sigmoid", "softmax")


def _activation_name(layer) -> str:
    activation = getattr(layer, "activation", None)
    name = getattr(activation, "__name__", None)
    if name is None:
        raise ValueError(f"Layer {layer.name!r} has no named activation")
    return str(name)


def export_weights(saved_model_dir: Path, output_path: Path) -> None:
    model = tf.keras.models.load_model(saved_model_dir)
    if len(model.layers) != len(EXPECTED_SHAPES):
        raise ValueError(
            f"Expected {len(EXPECTED_SHAPES)} Dense layers, got {len(model.layers)}"
        )

    arrays: dict[str, np.ndarray] = {}
    activations: list[str] = []
    for index, (layer, (kernel_shape, bias_shape), expected_act) in enumerate(
        zip(model.layers, EXPECTED_SHAPES, EXPECTED_ACTIVATIONS, strict=True)
    ):
        weights = layer.get_weights()
        if len(weights) != 2:
            raise ValueError(
                f"Layer {layer.name!r} has {len(weights)} weight arrays, expected 2"
            )
        kernel, bias = (np.asarray(weights[0]), np.asarray(weights[1]))
        if tuple(kernel.shape) != kernel_shape or tuple(bias.shape) != bias_shape:
            raise ValueError(
                f"Layer {layer.name!r} shapes {kernel.shape}/{bias.shape} "
                f"do not match expected {kernel_shape}/{bias_shape}"
            )
        activation = _activation_name(layer)
        if activation != expected_act:
            raise ValueError(
                f"Layer {layer.name!r} activation {activation!r} != {expected_act!r}"
            )
        arrays[f"kernel_{index}"] = kernel.astype(np.float32)
        arrays[f"bias_{index}"] = bias.astype(np.float32)
        activations.append(activation)

    arrays["activations"] = np.array(activations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saved-model",
        type=Path,
        required=True,
        help="Directory of the AMP applause-binary TensorFlow SavedModel",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/applause_mlp.npz"),
        help="Destination .npz path",
    )
    args = parser.parse_args()
    export_weights(args.saved_model, args.output)


if __name__ == "__main__":
    main()
