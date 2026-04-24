"""Training-log parser regression tests."""

import numpy as np

from scripts.extract_training_log import parse_log


def test_parse_log_extracts_energy_mean(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "Epoch 100/45000: loss=7.123e-04, |grad|=0.0423, 0.18s/ep "
        "| E_mean=2.3240, E_std=0.0004 | min|det F|=0.9412, "
        "masked=0.0000%\n"
    )

    epochs, losses, e_mean = parse_log(log)

    np.testing.assert_array_equal(epochs, np.array([100]))
    np.testing.assert_allclose(losses, np.array([7.123e-04]))
    np.testing.assert_allclose(e_mean, np.array([2.3240]))


def test_parse_log_keeps_nan_when_diagnostics_absent(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Epoch 1/45000: loss=1.000000e+01, |grad|=5.0\n")

    epochs, losses, e_mean = parse_log(log)

    np.testing.assert_array_equal(epochs, np.array([1]))
    np.testing.assert_allclose(losses, np.array([10.0]))
    assert np.isnan(e_mean[0])
