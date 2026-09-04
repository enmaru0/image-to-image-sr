import keras
import tensorflow as tf

from data.gpu_aug import normalize


class ImageLogger(keras.callbacks.Callback):
    def __init__(
        self,
        val_data,
        log_dir,
        jit_compile,
        test_data=None,
        test_seed=0,
        val_seed=0,
        num_output_channels=1,
        max_test_images=3,
    ):
        super().__init__()
        self.writer = tf.summary.create_file_writer(str(log_dir))
        self.val_data = val_data
        self.test_data = test_data
        self.first_log = True
        self.first_test_log = True
        self.max_test_images = max_test_images

        val_noise_shape = tf.concat(
            [
                tf.shape(val_data["imgs"])[:-1],
                tf.constant([num_output_channels], tf.int32),
            ],
            axis=0,
        )
        self.val_initial_noise = tf.random.stateless_normal(
            val_noise_shape, seed=[int(val_seed), 0], dtype=tf.float32
        )

        def predict_step(data):
            return self.model.predict_step(
                data,
                return_aux=True,
                apply_self_supervised_blur=True,
                initial_noise=self.val_initial_noise,
            )

        self.one_step = tf.function(
            predict_step, reduce_retracing=True, jit_compile=jit_compile
        )

        if test_data is not None:
            noise_shape = tf.concat(
                [
                    tf.shape(test_data["imgs"])[:-1],
                    tf.constant([num_output_channels], tf.int32),
                ],
                axis=0,
            )
            self.test_initial_noise = tf.random.stateless_normal(
                noise_shape, seed=[int(test_seed), 0], dtype=tf.float32
            )

            def predict_test_step(data):
                img_msks = self.model._get_img_msks(
                    data["msks"], self.model.cfg.bit_info.padding_bit
                )
                imgs = normalize(
                    data["imgs"], data["min_clip_vals"], data["max_clip_vals"]
                )
                imgs = imgs * img_msks
                preds = self.model.predict_step(
                    data, initial_noise=self.test_initial_noise
                )
                return preds, imgs

            self.test_one_step = tf.function(
                predict_test_step, reduce_retracing=True, jit_compile=jit_compile
            )

    @staticmethod
    def _orthogonal_slices(volume, axial_index=None):
        """Extract center AX/COR/SAG planes from a [B, Z, Y, X, C] volume."""
        if axial_index is None:
            axial_index = volume.shape[1] // 2
        coronal_index = volume.shape[2] // 2
        sagittal_index = volume.shape[3] // 2
        return {
            "AX": volume[:, axial_index, :, :, :],
            "COR": volume[:, :, coronal_index, :, :],
            "SAG": volume[:, :, :, sagittal_index, :],
        }

    @classmethod
    def _log_orthogonal_images(
        cls, name, volume, step, axial_index=None, max_outputs=3
    ):
        for plane, image in cls._orthogonal_slices(volume, axial_index).items():
            tf.summary.image(
                f"{name}/{plane}",
                image,
                step=step,
                max_outputs=max_outputs,
            )

    def on_test_batch_end(self, batch, logs=None):
        """
        Logs the first batch (images and predictions) during validation.
        Only triggered during the first validation step to avoid logging all validation data.
        """
        # Only log during the first validation batch (batch=0)
        if batch > 0:
            return

        outputs = self.one_step(self.val_data)
        _, preds, imgs, target_imgs = outputs[:4]
        observed_slice_msks = outputs[4] if len(outputs) > 4 else None
        missing_slice_msks = outputs[5] if len(outputs) > 5 else None

        with self.writer.as_default():
            step = self.model.optimizer.iterations
            slice_num = imgs.shape[1] // 2  # center of z
            if missing_slice_msks is not None:
                missing_indices = tf.where(missing_slice_msks[0, :, 0, 0, 0] > 0)[:, 0]
                if int(tf.size(missing_indices)) > 0:
                    distances = tf.abs(missing_indices - int(slice_num))
                    slice_num = int(missing_indices[tf.argmin(distances)].numpy())
            if self.first_log:
                self._log_orthogonal_images(
                    "Validation/Source",
                    imgs,
                    step,
                    axial_index=slice_num,
                )
                self._log_orthogonal_images(
                    "Validation/Target",
                    target_imgs,
                    step,
                    axial_index=slice_num,
                )
                if observed_slice_msks is not None:
                    self._log_orthogonal_images(
                        "Validation/Observed Slice Mask",
                        observed_slice_msks,
                        step,
                        axial_index=slice_num,
                    )
                self.first_log = False
            self._log_orthogonal_images(
                "Validation/Prediction",
                preds,
                step,
                axial_index=slice_num,
            )

            if self.test_data is not None:
                test_preds, test_imgs = self.test_one_step(self.test_data)
                test_slice_num = test_imgs.shape[1] // 2
                if self.first_test_log:
                    self._log_orthogonal_images(
                        "Test/Source",
                        test_imgs,
                        step,
                        axial_index=test_slice_num,
                        max_outputs=self.max_test_images,
                    )
                    self.first_test_log = False
                self._log_orthogonal_images(
                    "Test/Prediction",
                    test_preds,
                    step,
                    axial_index=test_slice_num,
                    max_outputs=self.max_test_images,
                )

        self.writer.flush()
