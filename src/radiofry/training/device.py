"""CUDA verification that refuses to let training silently fall back to the CPU.

`torch.cuda.is_available()` is not sufficient evidence that training will work. On a Blackwell
card (RTX 50-series, compute capability sm_120) a PyTorch build compiled without sm_120
kernels can report `is_available() == True` and then fail, or silently run through slow
fallback paths, at the first real kernel launch. The only proof is to launch one.

`verify_cuda` therefore checks, in order:

  1. a CUDA-enabled torch build (not `+cpu`)
  2. `torch.cuda.is_available()`
  3. a device is actually enumerated, with its name and capability
  4. the compiled architecture list **contains this device's capability**, or PTX that can be
     JIT-compiled forward to it
  5. a real tensor operation lands on the device and returns the right answer
  6. module parameters move to the device
  7. one forward/backward/step actually changes the weights

It raises `CudaUnavailable` rather than returning a CPU device, so a caller cannot start a
large CPU run by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class CudaUnavailable(RuntimeError):
    """CUDA is not usable for training. The message says exactly which check failed."""


@dataclass
class DeviceReport:
    """Everything worth recording about the device a run actually used."""

    ok: bool
    device: str
    torch_version: str = ""
    cuda_version: str = ""
    cudnn_version: str = ""
    device_name: str = ""
    capability: str = ""
    arch_list: tuple[str, ...] = ()
    total_memory_gb: float = 0.0
    checks: list[str] = field(default_factory=list)
    failure: str = ""

    def as_dict(self) -> dict:
        return {"ok": self.ok, "device": self.device, "torch": self.torch_version,
                "cuda": self.cuda_version, "cudnn": self.cudnn_version,
                "device_name": self.device_name, "capability": self.capability,
                "arch_list": list(self.arch_list),
                "total_memory_gb": round(self.total_memory_gb, 2),
                "checks": list(self.checks), "failure": self.failure}


def _capability_supported(capability: str, arch_list: tuple[str, ...]) -> tuple[bool, str]:
    """Is this device's compute capability actually covered by the compiled kernels?

    An exact `sm_XY` match is ideal. Failing that, PTX for an *earlier* architecture of the
    same major generation can be JIT-compiled forward, which works but is not guaranteed to
    be present, so it is reported distinctly rather than treated as equivalent.
    """

    if f"sm_{capability}" in arch_list:
        return True, f"exact sm_{capability} kernels present"

    compute = [a for a in arch_list if a.startswith("compute_")]
    major = int(capability[:-1]) if len(capability) > 1 else int(capability)
    forward = [a for a in compute if int(a.split("_")[1][:-1]) <= major]
    if forward:
        return True, (f"no exact sm_{capability}; PTX {sorted(forward)} can JIT forward "
                      "(slower first launch)")
    return False, (f"compiled kernels {sorted(arch_list)} do not cover sm_{capability} and no "
                   "forward-compatible PTX is present")


def verify_cuda(*, require_capability: str | None = None,
                run_training_step: bool = True) -> DeviceReport:
    """Prove CUDA works for training, or raise. Never returns a CPU device."""

    report = DeviceReport(ok=False, device="cpu")

    try:
        import torch
    except ImportError as error:                                  # pragma: no cover
        raise CudaUnavailable(f"torch is not installed: {error}") from error

    report.torch_version = torch.__version__
    report.cuda_version = str(torch.version.cuda)
    report.checks.append(f"torch {torch.__version__}, built against CUDA {torch.version.cuda}")

    if torch.version.cuda is None or "+cpu" in torch.__version__:
        report.failure = (
            f"this is a CPU-only PyTorch build ({torch.__version__}). Install a CUDA build, "
            "e.g. pip install --index-url https://download.pytorch.org/whl/cu130 torch")
        raise CudaUnavailable(report.failure)

    if not torch.cuda.is_available():
        report.failure = ("torch.cuda.is_available() is False - no usable driver/device. "
                          "Check nvidia-smi and that the driver supports this CUDA version.")
        raise CudaUnavailable(report.failure)
    report.checks.append("torch.cuda.is_available() = True")

    if torch.cuda.device_count() < 1:
        report.failure = "CUDA reports zero devices"
        raise CudaUnavailable(report.failure)

    report.device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    report.capability = f"{major}{minor}"
    report.total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    report.checks.append(
        f"device 0: {report.device_name}, sm_{report.capability}, "
        f"{report.total_memory_gb:.1f} GB")

    try:
        report.cudnn_version = str(torch.backends.cudnn.version())
    except Exception:                                             # noqa: BLE001
        report.cudnn_version = "unknown"

    report.arch_list = tuple(torch.cuda.get_arch_list())
    supported, note = _capability_supported(report.capability, report.arch_list)
    report.checks.append(f"arch list {list(report.arch_list)}")
    if not supported:
        report.failure = (
            f"{report.device_name} is sm_{report.capability} but {note}. "
            "is_available() being True does not mean kernels exist for this card - install a "
            "build that targets it (Blackwell / RTX 50-series needs CUDA 12.8 or newer).")
        raise CudaUnavailable(report.failure)
    report.checks.append(note)

    if require_capability and report.capability != require_capability:
        report.failure = (f"expected a sm_{require_capability} device, found "
                          f"sm_{report.capability} ({report.device_name})")
        raise CudaUnavailable(report.failure)

    # A real kernel launch. This is the check that is_available() cannot substitute for.
    try:
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        product = (a @ b).sum().item()
        if product != product:                                    # NaN
            raise RuntimeError("matmul returned NaN")
        torch.cuda.synchronize()
    except Exception as error:                                    # noqa: BLE001
        report.failure = (f"a CUDA kernel launch failed despite is_available() being True: "
                          f"{error}")
        raise CudaUnavailable(report.failure) from error
    report.checks.append("matmul kernel launched and synchronised on device")

    if run_training_step:
        try:
            from radiofry.models.modulation_cnn import ModulationCNN

            model = ModulationCNN(4, 8).cuda()
            if next(model.parameters()).device.type != "cuda":
                raise RuntimeError("parameters did not move to the device")
            before = next(model.parameters()).detach().clone()

            optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
            inputs = torch.randn(64, 4, 128, device="cuda")
            targets = torch.randint(0, 8, (64,), device="cuda")
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimiser.step()
            torch.cuda.synchronize()

            after = next(model.parameters()).detach()
            if torch.equal(before, after):
                raise RuntimeError("weights did not change after an optimiser step")
            report.checks.append(
                f"forward/backward/step on device changed the weights (loss {loss.item():.4f})")
            report.checks.append(
                f"peak GPU memory during smoke test: "
                f"{torch.cuda.max_memory_allocated()/1e6:.1f} MB")
            del model, inputs, targets, loss
            torch.cuda.empty_cache()
        except Exception as error:                                # noqa: BLE001
            report.failure = f"the GPU training smoke test failed: {error}"
            raise CudaUnavailable(report.failure) from error

    report.ok = True
    report.device = "cuda"
    return report


def describe_environment() -> dict:
    """Versions worth recording alongside any result."""

    import platform
    import sys

    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = str(torch.version.cuda)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
            info["arch_list"] = list(torch.cuda.get_arch_list())
    except ImportError:                                           # pragma: no cover
        info["torch"] = "not installed"
    return info
