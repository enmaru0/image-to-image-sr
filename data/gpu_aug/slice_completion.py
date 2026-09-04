import math

import tensorflow as tf


def _slice_profile_sigma_vox(
    factors, spacing_mm_z, clean_thickness_mm, degraded_thickness_mm, profile_model
):
    """Return one additional Gaussian SSP sigma per batch item."""
    dtype = tf.float32
    spacing = tf.cast(spacing_mm_z, dtype)
    clean = tf.cast(clean_thickness_mm, dtype)
    if degraded_thickness_mm is None:
        degraded = tf.cast(factors, dtype) * spacing
    else:
        degraded = tf.fill(tf.shape(factors), tf.cast(degraded_thickness_mm, dtype))

    additional_width = tf.sqrt(tf.maximum(tf.square(degraded) - clean**2, 0.0))
    if str(profile_model) == "gaussian_fwhm":
        sigma_mm = additional_width / math.sqrt(8.0 * math.log(2.0))
    elif str(profile_model) == "box_variance":
        sigma_mm = additional_width / math.sqrt(12.0)
    else:
        raise ValueError(
            f"profile_model must be 'gaussian_fwhm' or 'box_variance': {profile_model}"
        )
    return sigma_mm / spacing


def _slice_profile_sigma_vox_float(
    spacing_mm_z, clean_thickness_mm, degraded_thickness_mm, profile_model
):
    additional_width = math.sqrt(
        max(float(degraded_thickness_mm) ** 2 - float(clean_thickness_mm) ** 2, 0.0)
    )
    if str(profile_model) == "gaussian_fwhm":
        sigma_mm = additional_width / math.sqrt(8.0 * math.log(2.0))
    elif str(profile_model) == "box_variance":
        sigma_mm = additional_width / math.sqrt(12.0)
    else:
        raise ValueError(
            f"profile_model must be 'gaussian_fwhm' or 'box_variance': {profile_model}"
        )
    return sigma_mm / float(spacing_mm_z)


def _variable_gaussian_blur_z(imgs, img_msks, sigma_vox, max_sigma_vox, truncate):
    """Mask-normalized Z blur with a different sigma for each batch item."""
    radius = max(int(math.ceil(float(truncate) * float(max_sigma_vox))), 1)
    coordinates = tf.range(-radius, radius + 1, dtype=tf.float32)
    sigma_vox = tf.cast(sigma_vox, tf.float32)
    safe_sigma = tf.maximum(sigma_vox, tf.constant(1e-6, tf.float32))
    weights = tf.exp(
        -tf.square(coordinates)[None, :] / (2.0 * tf.square(safe_sigma)[:, None])
    )
    weights = tf.where(
        sigma_vox[:, None] > 0, weights, tf.cast(coordinates[None, :] == 0, tf.float32)
    )
    weights /= tf.reduce_sum(weights, axis=1, keepdims=True)
    weights = tf.cast(weights, imgs.dtype)

    paddings = [[0, 0], [radius, radius], [0, 0], [0, 0], [0, 0]]
    padded_imgs = tf.pad(imgs * img_msks, paddings)
    padded_msks = tf.pad(img_msks, paddings)
    depth = tf.shape(imgs)[1]
    numerator = tf.zeros_like(imgs)
    denominator = tf.zeros_like(img_msks)
    for kernel_index in range(2 * radius + 1):
        weight = weights[:, kernel_index, None, None, None, None]
        shifted_imgs = padded_imgs[:, kernel_index : kernel_index + depth]
        shifted_msks = padded_msks[:, kernel_index : kernel_index + depth]
        numerator += shifted_imgs * weight
        denominator += shifted_msks * weight
    return numerator / tf.maximum(denominator, tf.cast(1e-6, imgs.dtype))


def _sample_factors(
    batch_size, factor_values, sampling_weights, is_training, validation_factor
):
    factors = tf.constant([int(value) for value in factor_values], tf.int32)
    if is_training:
        weights = (
            [1.0] * len(factor_values)
            if sampling_weights is None
            else [float(value) for value in sampling_weights]
        )
        probabilities = tf.constant(weights, tf.float32)
        cumulative = tf.cumsum(probabilities / tf.reduce_sum(probabilities))
        draws = tf.random.uniform((batch_size, 1), 0.0, 1.0)
        indices = tf.reduce_sum(tf.cast(draws > cumulative[None, :], tf.int32), axis=1)
        return tf.gather(factors, indices)
    return tf.fill((batch_size,), tf.cast(validation_factor, tf.int32))


def simulate_slice_completion(
    imgs,
    img_msks,
    spacing_mm_z,
    keep_every_n_values=(2,),
    sampling_weights=None,
    random_offset=True,
    validation_factor=2,
    validation_offset=0,
    fill_mode="linear",
    slice_profile_blur=None,
    is_training=True,
):
    """Create sparse-to-dense training inputs from a dense B-Z-Y-X-C volume.

    The returned source stays on the original dense grid. Only slices selected by
    ``factor`` and ``offset`` contribute to its linear interpolation, so withheld
    target slices cannot leak into the input.
    """
    if str(fill_mode) != "linear":
        raise ValueError(f"Only fill_mode='linear' is supported: {fill_mode}")

    imgs = tf.convert_to_tensor(imgs)
    img_msks = tf.cast(img_msks, imgs.dtype)
    batch_size = tf.shape(imgs)[0]
    depth = tf.shape(imgs)[1]
    factors = _sample_factors(
        batch_size,
        keep_every_n_values,
        sampling_weights,
        bool(is_training),
        int(validation_factor),
    )

    if is_training and bool(random_offset):
        offsets = tf.cast(
            tf.floor(
                tf.random.uniform((batch_size,), 0.0, 1.0)
                * tf.cast(factors, tf.float32)
            ),
            tf.int32,
        )
    else:
        offsets = tf.math.floormod(
            tf.fill((batch_size,), tf.cast(validation_offset, tf.int32)), factors
        )

    sampled_imgs = imgs
    blur_cfg = slice_profile_blur
    if blur_cfg is not None and bool(blur_cfg.enabled):
        degraded_thickness = getattr(blur_cfg, "degraded_thickness_mm", None)
        sigma_vox = _slice_profile_sigma_vox(
            factors,
            spacing_mm_z,
            float(blur_cfg.clean_thickness_mm),
            degraded_thickness,
            str(blur_cfg.profile_model),
        )
        if degraded_thickness is None:
            max_factor = max(
                max(int(value) for value in keep_every_n_values), int(validation_factor)
            )
            max_degraded = max_factor * float(spacing_mm_z)
        else:
            max_degraded = float(degraded_thickness)
        max_sigma = _slice_profile_sigma_vox_float(
            spacing_mm_z,
            float(blur_cfg.clean_thickness_mm),
            max_degraded,
            str(blur_cfg.profile_model),
        )
        sampled_imgs = _variable_gaussian_blur_z(
            imgs, img_msks, sigma_vox, max_sigma, float(blur_cfg.gaussian_truncate)
        )

    z = tf.range(depth, dtype=tf.int32)[None, :]
    factors_2d = factors[:, None]
    offsets_2d = offsets[:, None]
    remainder = tf.math.floormod(z - offsets_2d, factors_2d)
    observed_slices = remainder == 0

    first_observed = offsets_2d
    last_observed = (
        first_observed
        + tf.math.floordiv((depth - 1) - first_observed, factors_2d) * factors_2d
    )
    lower = tf.clip_by_value(z - remainder, first_observed, last_observed)
    upper = tf.clip_by_value(lower + factors_2d, first_observed, last_observed)

    lower_values = tf.gather(sampled_imgs, lower, axis=1, batch_dims=1)
    upper_values = tf.gather(sampled_imgs, upper, axis=1, batch_dims=1)
    lower_msks = tf.gather(img_msks, lower, axis=1, batch_dims=1)
    upper_msks = tf.gather(img_msks, upper, axis=1, batch_dims=1)
    interval = upper - lower
    alpha = tf.where(
        interval > 0,
        tf.cast(z - lower, imgs.dtype) / tf.cast(tf.maximum(interval, 1), imgs.dtype),
        tf.zeros_like(tf.cast(z, imgs.dtype)),
    )[..., None, None, None]

    lower_weight = (1.0 - alpha) * lower_msks
    upper_weight = alpha * upper_msks
    source = (lower_values * lower_weight + upper_values * upper_weight) / tf.maximum(
        lower_weight + upper_weight, tf.cast(1e-6, imgs.dtype)
    )
    source *= img_msks

    observed_slice_mask = tf.cast(observed_slices, imgs.dtype)[..., None, None, None]
    observed_voxel_mask = observed_slice_mask * img_msks
    missing_voxel_mask = (1.0 - observed_slice_mask) * img_msks
    return source, observed_voxel_mask, missing_voxel_mask, factors
