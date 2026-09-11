"""Guards on the CUDA gate and the RadioML training configuration.

The gate exists because `torch.cuda.is_available()` is not evidence that training will work:
on a Blackwell card a build without sm_120 kernels can report True and then fail at the first
kernel launch. The failure mode this prevents is the expensive one - a long run silently
completing on the CPU, or a run that dies an hour in.

These tests must pass on a machine with no GPU (CI has none), so anything needing a device
skips rather than failing.
"""

import pytest

pytest.importorskip("torch", reason="torch is part of the optional ml extras")

import torch  # noqa: E402

from radiofry.training.device import (  # noqa: E402
    CudaUnavailable,
    DeviceReport,
    _capability_supported,
    describe_environment,
    verify_cuda,
)
from radiofry.training.train_radioml import (  # noqa: E402
    RadioMLTrainingConfig,
    load_pool,
    train,
)

has_cuda = torch.cuda.is_available() and torch.version.cuda is not None
requires_cuda = pytest.mark.skipif(not has_cuda, reason="no CUDA device on this machine")


# --- capability matching: the part that catches a Blackwell mismatch ---------------------------


def test_an_exact_kernel_match_is_accepted() -> None:
    ok, note = _capability_supported("120", ("sm_90", "sm_120"))

    assert ok
    assert "exact" in note


def test_a_missing_architecture_with_no_forward_ptx_is_refused() -> None:
    """The RTX 50-series case: a cu124 build has no sm_120 kernels and no PTX to JIT."""

    ok, note = _capability_supported("120", ("sm_70", "sm_80", "sm_90"))

    assert not ok
    assert "do not cover sm_120" in note


def test_forward_compatible_ptx_is_accepted_but_reported_distinctly() -> None:
    ok, note = _capability_supported("120", ("sm_90", "compute_90"))

    assert ok
    assert "PTX" in note and "JIT" in note


def test_an_empty_architecture_list_is_refused() -> None:
    ok, _ = _capability_supported("120", ())

    assert not ok


# --- the report ----------------------------------------------------------------------------------


def test_a_report_serialises_for_the_metrics_file() -> None:
    report = DeviceReport(ok=True, device="cuda", torch_version="2.13.0+cu130",
                          device_name="RTX 5060", capability="120",
                          arch_list=("sm_120",), total_memory_gb=8.0)

    payload = report.as_dict()

    assert payload["ok"] is True
    assert payload["device_name"] == "RTX 5060"
    assert payload["arch_list"] == ["sm_120"]


def test_describe_environment_records_what_a_result_depends_on() -> None:
    info = describe_environment()

    assert "python" in info and "torch" in info
    assert "cuda_available" in info


# --- refusing the CPU ------------------------------------------------------------------------------


@pytest.mark.skipif(has_cuda, reason="this asserts the refusal path on a CPU-only machine")
def test_a_cpu_only_build_is_refused_rather_than_used() -> None:
    """The whole point: no silent CPU fallback."""

    with pytest.raises(CudaUnavailable) as error:
        verify_cuda()

    assert "CPU" in str(error.value) or "is_available" in str(error.value)


@requires_cuda
def test_verify_cuda_proves_a_kernel_launch_and_a_weight_update() -> None:
    report = verify_cuda()

    assert report.ok and report.device == "cuda"
    assert report.device_name
    assert any("kernel launched" in c for c in report.checks)
    assert any("changed the weights" in c for c in report.checks)


@requires_cuda
def test_a_wrong_required_capability_is_refused() -> None:
    with pytest.raises(CudaUnavailable):
        verify_cuda(require_capability="35")


# --- training configuration guards -----------------------------------------------------------------


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        RadioMLTrainingConfig(mode="whatever")


def test_transfer_modes_require_a_starting_checkpoint() -> None:
    for mode in ("finetune", "linear_probe"):
        with pytest.raises(ValueError):
            RadioMLTrainingConfig(mode=mode, initial_checkpoint=None)


def test_scratch_needs_no_checkpoint() -> None:
    assert RadioMLTrainingConfig(mode="scratch").initial_checkpoint is None


def test_an_output_path_is_required() -> None:
    with pytest.raises(ValueError):
        train(RadioMLTrainingConfig(mode="scratch", output=""))


def test_training_refuses_to_overwrite_the_production_checkpoint(tmp_path) -> None:
    config = RadioMLTrainingConfig(mode="scratch",
                                   output=str(tmp_path / "modulation_cnn_v3_spsaug.pt"))

    with pytest.raises(PermissionError):
        train(config)


def test_the_sealed_split_is_not_reachable_from_the_training_path() -> None:
    with pytest.raises(PermissionError):
        load_pool("", "test", classes=("BPSK",), snr_db=(30,), per_config=4, seed=0)


def test_the_default_label_space_is_the_datasets_own_24_classes() -> None:
    """Training on the 5 mappable classes alone would flatter the V3 comparison."""

    config = RadioMLTrainingConfig(mode="scratch")

    assert len(config.classes) == 24
    assert config.classes[0] == "OOK" and config.classes[3] == "BPSK"


def test_the_default_snr_span_covers_the_whole_dataset() -> None:
    config = RadioMLTrainingConfig(mode="scratch")

    assert min(config.snr_db) == -20 and max(config.snr_db) == 30
    assert len(config.snr_db) == 26
