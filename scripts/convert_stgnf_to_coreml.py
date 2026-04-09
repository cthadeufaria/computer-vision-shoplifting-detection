#!/usr/bin/env python3
"""
Convert STG-NF .tar checkpoint → CoreML .mlpackage

Requirements:
    Python 3.11 (coremltools native extensions not yet compiled for 3.12+)
    pip install torch==2.4.0 coremltools==8.1 onnxscript

Usage (from repo root):
    python3.11 scripts/convert_stgnf_to_coreml.py
    # or with a dedicated venv:
    /tmp/coreml_env/bin/python3.11 scripts/convert_stgnf_to_coreml.py

Output:
    artifacts/stg_nf/coreml/STGNFModel.mlpackage

Then copy into the Xcode project:
    cp -r artifacts/stg_nf/coreml/STGNFModel.mlpackage ios/ShopliftDetect/Resources/

CoreML I/O spec:
    Input:  pose_window  [1, 2, 24, 18]  float32  (batch, xy_channels, frames, joints)
    Output: nll_score    [1]             float32  (Swift: anomaly_score = -nll_score[0])
"""

import argparse
import math
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Repo-root path setup so we can import stg_nf_official submodules directly.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STG_NF_DIR = os.path.join(REPO_ROOT, "stg_nf_official")
sys.path.insert(0, STG_NF_DIR)

from models.STG_NF.model_pose import STG_NF  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (from the latest Multi Apr01_1416 checkpoint requested for iOS)
# ---------------------------------------------------------------------------
CHECKPOINT = os.path.join(
    REPO_ROOT,
    "artifacts",
    "stg_nf",
    "multi_runs",
    "Multi",
    "Apr01_1416",
    "Apr01_1419__checkpoint.pth.tar",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "artifacts", "stg_nf", "coreml")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "STGNFModel.mlpackage")
ONNX_TMP_PATH = os.path.join(OUTPUT_DIR, "STGNFModel.onnx")

# Model hyperparameters matching the checkpoint
MODEL_ARGS = dict(
    pose_shape=(2, 24, 18),
    hidden_channels=0,
    K=8,
    L=1,
    actnorm_scale=1.0,
    flow_permutation="invcov",   # saved value — falls to Permute2d (not InvertibleConv1x1)
    flow_coupling="affine",       # hardcoded in train_utils.py
    LU_decomposed=True,
    learn_top=False,
    R=3.0,
    edge_importance=False,
    temporal_kernel_size=None,
    strategy="uniform",
    max_hops=8,
    device="cpu",
)


# ---------------------------------------------------------------------------
# Shape-extraction patches applied BEFORE tracing.
#
# coremltools cannot convert `aten::Int(aten::size(...))` chains that arise
# from Python unpacking of tensor shapes.  We replace every occurrence with
# ops that do not produce int-from-shape nodes in TorchScript.
# ---------------------------------------------------------------------------

def _patch_dynamic_shapes():
    """
    Monkey-patch modules to remove all aten::size → aten::Int chains and
    squeeze-on-unknown-dim ops that coremltools cannot convert.
    """
    from models.STG_NF.stgcn import ConvTemporalGraphical
    import models.STG_NF.utils as nf_utils

    # 1. ConvTemporalGraphical: replace shape-unpacking view with unflatten.
    def _ctg_forward(self, x, A):
        x = self.conv(x)
        # Original: n, kc, t, v = x.size(); x = x.view(n, k, kc//k, t, v)
        # unflatten channel dim (B, k*oc, T, V) → (B, k, oc, T, V)
        x = torch.unflatten(x, 1, (self.kernel_size, -1))
        x = torch.einsum("nkctv,kvw->nctw", x, A)
        return x.contiguous(), A

    ConvTemporalGraphical.forward = _ctg_forward

    # 2. split_feature: remove squeeze(dim=1) so tensors stay 4D throughout.
    #    All callers (FlowStep.normal_flow, etc.) already have
    #    "if len(z.shape) == 3: z = z.unsqueeze(1)" guards — those become no-ops.
    def _split_feature(tensor, type="split", imgs=False):
        C = tensor.size(1)
        if imgs:
            if type == "split":
                return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
            elif type == "cross":
                return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
        # Non-imgs: return 4D tensors (no squeeze — dim-1 slices may be size≠1
        # and coremltools cannot statically verify the squeeze axis)
        if type == "split":
            return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
        elif type == "cross":
            return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

    nf_utils.split_feature = _split_feature
    # Patch all modules that imported split_feature directly
    import models.STG_NF.model_pose as model_pose_mod
    import models.STG_NF.modules_pose as modules_pose_mod
    model_pose_mod.split_feature = _split_feature
    modules_pose_mod.split_feature = _split_feature


# ---------------------------------------------------------------------------
# Wrapper: single-tensor forward, fixed batch=1, no dynamic shape extraction.
# ---------------------------------------------------------------------------
class STGNFWrapper(nn.Module):
    """Forward pass only, label=None, batch=1. Returns nll as shape [1]."""

    def __init__(self, model: STG_NF):
        super().__init__()
        self.model = model

    def _flow_encode(self, pose_window: torch.Tensor):
        """Replaces FlowNet.encode() — uses torch.zeros(1) instead of zeros(z.shape[0])."""
        z = pose_window
        logdet = torch.zeros(1)   # batch=1 constant, no aten::size
        for layer, _ in zip(self.model.flow.layers, self.model.flow.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def _prior(self, pose_window: torch.Tensor):
        """Replaces STG_NF.prior() — avoids data.shape[0] dynamic extraction."""
        from models.STG_NF.utils import split_feature

        # prior_h has shape [1, C*2, T, V]; already batch=1
        h = self.model.prior_h
        if self.model.learn_top:
            h = self.model.learn_top_fn(h)
        return split_feature(h, "split")

    def forward(self, pose_window: torch.Tensor) -> torch.Tensor:
        import math
        from models.STG_NF.modules_pose import gaussian_likelihood

        z, objective = self._flow_encode(pose_window)
        mean, logs = self._prior(pose_window)
        objective = objective + gaussian_likelihood(mean, logs, z)

        # bits/dim — c, t, v are 2, 24, 18 for this checkpoint (constants)
        nll = (-objective) / (math.log(2.0) * 2.0 * 24.0 * 18.0)
        return nll  # shape [1]


def build_model() -> STGNFWrapper:
    _patch_dynamic_shapes()   # must be called before model instantiation
    print("→ Building model …")
    model = STG_NF(**MODEL_ARGS)

    print(f"→ Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    # Critical: mark ActNorm as already initialised so it doesn't reinit on first batch.
    model.set_actnorm_init()
    model.eval()

    wrapper = STGNFWrapper(model)
    wrapper.eval()
    return wrapper


def verify_numeric(wrapper: STGNFWrapper, example_input: torch.Tensor) -> float:
    """Run the PyTorch model and return the NLL value for later comparison."""
    with torch.no_grad():
        nll = wrapper(example_input)
    return nll.item()


def export_program(wrapper: STGNFWrapper, example_input: torch.Tensor):
    """Export via torch.export in ATEN dialect (required by coremltools 8.x)."""
    from torch.export import export as torch_export

    print("→ Exporting via torch.export …")
    # Disable gradients on all parameters to encourage functional ATEN export.
    for p in wrapper.parameters():
        p.requires_grad_(False)

    ep = torch_export(wrapper, (example_input,))
    print(f"   Dialect: {ep.dialect}")

    # coremltools requires ATEN or EDGE dialect, not TRAINING.
    if ep.dialect == "TRAINING":
        print("   Decomposing TRAINING → ATEN dialect …")
        ep = ep.run_decompositions({})
        print(f"   Dialect after decomposition: {ep.dialect}")

    # Smoke-test: verify exported output matches original.
    with torch.no_grad():
        ep_nll = ep.module()(example_input).item()
    return ep, ep_nll


def convert_via_onnx(wrapper: STGNFWrapper, example_input: torch.Tensor) -> str:
    """Export to ONNX via legacy exporter, return path to .onnx file."""
    import onnx

    print("→ Exporting to ONNX (legacy exporter) …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Use legacy exporter explicitly to avoid the new dynamo-based exporter
    # which has issues with squeeze/unsqueeze patterns in this model.
    torch.onnx.export(
        wrapper,
        example_input,
        ONNX_TMP_PATH,
        opset_version=16,
        input_names=["pose_window"],
        output_names=["nll_score"],
        dynamic_axes=None,   # fixed batch=1
        verbose=False,
        dynamo=False,        # force legacy exporter
    )

    print("→ Checking ONNX model …")
    onnx_model = onnx.load(ONNX_TMP_PATH)
    onnx.checker.check_model(onnx_model)
    print("   ONNX check passed.")
    return ONNX_TMP_PATH


def onnx_numeric_check(onnx_path: str, example_input: torch.Tensor, python_nll: float) -> None:
    """Verify ONNX output matches PyTorch within tolerance."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = {"pose_window": example_input.numpy()}
    onnx_nll = sess.run(None, inp)[0][0]
    diff = abs(float(onnx_nll) - python_nll)
    print(f"   PyTorch NLL: {python_nll:.6f}  |  ONNX NLL: {float(onnx_nll):.6f}  |  diff: {diff:.2e}")
    assert diff < 0.01, f"ONNX/PyTorch NLL mismatch too large: {diff}"
    print("   Numeric check PASSED (diff < 0.01)")


def convert_to_coreml(exported_program) -> None:
    """Convert torch.ExportedProgram → CoreML .mlpackage via coremltools 8.x."""
    import coremltools as ct

    print("→ Converting ExportedProgram → CoreML (coremltools 8.x) …")
    input_shape = ct.Shape(shape=[1, 2, 24, 18])
    ml_model = ct.convert(
        exported_program,
        inputs=[ct.TensorType(name="pose_window", shape=input_shape)],
        outputs=[ct.TensorType(name="nll_score")],
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ml_model.save(OUTPUT_PATH)
    print(f"   Saved: {OUTPUT_PATH}")


def coreml_numeric_check(python_nll: float, example_input: torch.Tensor) -> None:
    """Load saved .mlpackage and verify output matches PyTorch."""
    import coremltools as ct

    print("→ Verifying CoreML output …")
    ml_model = ct.models.MLModel(OUTPUT_PATH)
    inp = {"pose_window": example_input.numpy()}
    result = ml_model.predict(inp)
    coreml_nll = float(result["nll_score"].flatten()[0])
    diff = abs(coreml_nll - python_nll)
    print(f"   PyTorch NLL: {python_nll:.6f}  |  CoreML NLL: {coreml_nll:.6f}  |  diff: {diff:.2e}")
    assert diff < 0.01, f"CoreML/PyTorch NLL mismatch too large: {diff}"
    print("   CoreML numeric check PASSED (diff < 0.01)")


def main():
    parser = argparse.ArgumentParser(description="Convert STG-NF to CoreML")
    parser.add_argument("--skip-coreml-verify", action="store_true",
                        help="Skip CoreML numeric verification (faster, useful on CI)")
    args = parser.parse_args()

    torch.manual_seed(42)
    example_input = torch.randn(1, 2, 24, 18)

    # 1. Build and load model
    wrapper = build_model()

    # 2. PyTorch reference NLL
    print("→ Computing PyTorch reference NLL …")
    python_nll = verify_numeric(wrapper, example_input)
    print(f"   NLL = {python_nll:.6f}")

    # 3. torch.export
    ep, ep_nll = export_program(wrapper, example_input)
    diff = abs(ep_nll - python_nll)
    print(f"   Exported NLL: {ep_nll:.6f}  |  diff from PyTorch: {diff:.2e}")
    assert diff < 0.01, f"Exported NLL mismatch too large: {diff}"
    print("   torch.export numeric check PASSED")

    # 4. CoreML conversion from ExportedProgram
    convert_to_coreml(ep)

    # 5. CoreML numeric verification
    if not args.skip_coreml_verify:
        coreml_numeric_check(python_nll, example_input)

    print("\n✓ Done.")
    print(f"  Model saved to: {OUTPUT_PATH}")
    print(f"\nNext step - copy into Xcode project:")
    print(f"  cp -r {OUTPUT_PATH} ios/ShopliftDetect/Resources/")


if __name__ == "__main__":
    main()
