from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from .slice_completion import simulate_slice_completion


def _blur_cfg(enabled):
    return SimpleNamespace(
        enabled=enabled,
        clean_thickness_mm=1.0,
        degraded_thickness_mm=None,
        profile_model="box_variance",
        gaussian_truncate=3.0,
    )


def _simulate(volume, factor, offset=0, blur=False):
    return simulate_slice_completion(
        tf.convert_to_tensor(volume, tf.float32),
        tf.ones_like(volume, tf.float32),
        spacing_mm_z=1.0,
        keep_every_n_values=[factor],
        validation_factor=factor,
        validation_offset=offset,
        slice_profile_blur=_blur_cfg(blur),
        is_training=False,
    )


def test_linear_ramp_is_reconstructed_between_observed_slices():
    ramp = np.arange(16, dtype=np.float32)[None, :, None, None, None]
    source, observed, missing, factors = _simulate(ramp, factor=4)

    np.testing.assert_allclose(source.numpy()[:, :13], ramp[:, :13], atol=1e-6)
    np.testing.assert_array_equal(
        observed.numpy()[0, :, 0, 0, 0], np.asarray([1, 0, 0, 0] * 4, np.float32)
    )
    np.testing.assert_array_equal(
        missing.numpy()[0, :, 0, 0, 0], 1.0 - observed.numpy()[0, :, 0, 0, 0]
    )
    np.testing.assert_array_equal(factors.numpy(), [4])


def test_withheld_target_values_do_not_leak_into_source():
    volume = np.arange(16, dtype=np.float32)[None, :, None, None, None]
    changed = volume.copy()
    changed[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]] += 1000.0

    source, _, _, _ = _simulate(volume, factor=4)
    changed_source, _, _, _ = _simulate(changed, factor=4)

    np.testing.assert_array_equal(source.numpy(), changed_source.numpy())


def test_offset_changes_the_observed_slice_phase():
    volume = np.arange(12, dtype=np.float32)[None, :, None, None, None]
    _, observed, _, _ = _simulate(volume, factor=5, offset=2)
    expected = np.zeros(12, np.float32)
    expected[[2, 7]] = 1.0
    np.testing.assert_array_equal(observed.numpy()[0, :, 0, 0, 0], expected)


def test_slice_profile_blur_is_applied_before_sampling():
    volume = np.zeros((1, 17, 1, 1, 1), np.float32)
    volume[:, 8] = 1.0
    source_without_blur, _, _, _ = _simulate(volume, factor=4, blur=False)
    source_with_blur, _, _, _ = _simulate(volume, factor=4, blur=True)

    assert source_without_blur.numpy()[0, 8, 0, 0, 0] == 1.0
    assert 0.0 < source_with_blur.numpy()[0, 8, 0, 0, 0] < 1.0


def test_simulator_traces_with_xla_compatible_shapes():
    volume = tf.reshape(tf.range(2 * 24, dtype=tf.float32), (2, 24, 1, 1, 1))
    mask = tf.ones_like(volume)

    @tf.function(jit_compile=True)
    def run(images, masks):
        return simulate_slice_completion(
            images,
            masks,
            spacing_mm_z=1.0,
            keep_every_n_values=[2, 5, 8],
            sampling_weights=[0.2, 0.3, 0.5],
            random_offset=True,
            validation_factor=8,
            validation_offset=0,
            slice_profile_blur=_blur_cfg(True),
            is_training=True,
        )

    source, observed, missing, factors = run(volume, mask)
    assert source.shape == volume.shape
    assert observed.shape == volume.shape
    assert missing.shape == volume.shape
    assert factors.shape == (2,)
