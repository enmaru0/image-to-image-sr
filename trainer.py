import keras
from keras import Model
from keras.api.metrics import Mean
from keras.src import ops
import tensorflow as tf
from tensorflow import GradientTape

from data.gpu_aug import (
    apply_random_gaussian_noise,
    apply_random_sharpness_or_gaussian_filter,
    cardiac_motion_blur,
    gaussian_filter,
    normalize,
    random_gaussian_filter,
    random_gamma_correction,
    random_normalize,
    simulate_slice_completion,
    simulate_slice_thickness,
)
from evaluation import (
    masked_mae_with_scale,
    masked_psnr,
    masked_ssim_xy,
    masked_xy_edge_strength_ratio,
    masked_z_gradient_mae,
)
from losses.image import masked_xy_gradient_loss


def _concat_mask(msk_list):
    # check if all masks are the same dimensions
    if not all(m.ndim == msk_list[0].ndim for m in msk_list):
        raise ValueError("All masks must have the same dimensions.")
    out = msk_list[0]
    for msk in msk_list[1:]:
        out = out * msk
    return out


def _bitwise_and_float(x, bit_msk):
    return ops.cast((x & bit_msk) > 0, "float32")


def _bit_mask_to_one_hot(x, bit_list):
    """
    x: Tensor of shape (batch_size, height, width, channels)
    bit_msk: int or list of int
    """
    or_msk = sum(1 << bit for bit in bit_list)
    msk_all = _bitwise_and_float(x, or_msk)
    background = ops.where(msk_all, ops.zeros_like(x), ops.ones_like(x))
    msk_list = [background]
    for bit in bit_list:
        msk_list.append(_bitwise_and_float(x, 1 << (bit)))
    return ops.concatenate(msk_list, axis=-1)


@keras.saving.register_keras_serializable()
class CustomModel(Model):
    """
    https://keras.io/guides/custom_train_step_in_tensorflow/
    上記リンクを参考に作成した。GANなど少し複雑なモデルの学習方法も書いてあります。
    """

    EVALUATION_METRIC_NAMES = (
        "ssim_xy_global",
        "ssim_xy_heart",
        "psnr_heart",
        "mae_hu_heart",
        "z_gradient_mae",
        "xy_edge_strength_ratio",
    )
    SLICE_COMPLETION_EVALUATION_METRIC_NAMES = ("linear_baseline_mae_hu",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = None

    @property
    def metrics_dict(self):
        if not hasattr(self, "_metrics_dict"):
            self._metrics_dict = {}
        metric_names = [
            "rfr_loss",
            "gradient_loss",
            "z_gradient_loss",
            "mae",
            "mse",
            "psnr",
            "total_loss",
            *self.EVALUATION_METRIC_NAMES,
        ]
        if self.cfg is not None and self.is_slice_completion(self.cfg):
            metric_names.extend(self.SLICE_COMPLETION_EVALUATION_METRIC_NAMES)
        for name in metric_names:
            if name not in self._metrics_dict:
                self._metrics_dict[name] = Mean(name=name)
        return self._metrics_dict

    @staticmethod
    def _get_img_msks(msks, padding_bit):
        # padding_bitが立っていない部分は画像のマスクとして使う
        img_msks = ops.cast(msks & (1 << padding_bit) == 0, "float32")
        return img_msks

    @staticmethod
    def _get_heart_msks(msks, heart_bit):
        """Extract the heart ROI stored in the configured bit of the mask."""
        return ops.cast((msks & (1 << heart_bit)) > 0, "float32")

    @staticmethod
    def center_crop_to_model_size(tensor, cfg):
        """Crop a degradation-context tensor back to the network input size."""
        tensor = tf.convert_to_tensor(tensor)
        target_size = tf.constant(
            [int(value) for value in cfg.aug.crop_size_zyx], tf.int32
        )
        input_size = tf.shape(tensor)[1:4]
        static_input_size = tensor.shape.as_list()[1:4]
        if all(value is not None for value in static_input_size):
            if any(
                int(input_value) < int(target_value)
                for input_value, target_value in zip(
                    static_input_size, cfg.aug.crop_size_zyx
                )
            ):
                raise ValueError(
                    "degradation context is smaller than aug.crop_size_zyx: "
                    f"{static_input_size} < {list(cfg.aug.crop_size_zyx)}"
                )
        else:
            tf.debugging.assert_greater_equal(
                input_size,
                target_size,
                message="degradation context is smaller than aug.crop_size_zyx",
            )
        start = (input_size - target_size) // 2
        begin = tf.concat([[0], start, [0]], axis=0)
        size = tf.concat(
            [[tf.shape(tensor)[0]], target_size, [tf.shape(tensor)[4]]], axis=0
        )
        cropped = tf.slice(tensor, begin, size)
        static_shape = tensor.shape.as_list()
        if static_shape is not None and len(static_shape) == 5:
            cropped.set_shape(
                [
                    static_shape[0],
                    *[int(v) for v in cfg.aug.crop_size_zyx],
                    static_shape[4],
                ]
            )
        return cropped

    def train_step(self, data):
        """
        ここのデータ名であったりselfに渡す引数を変えた場合は、
        callbacks/image_logger.pyのpredict_stepやon_test_batch_endも変更すること
        ここはjit_compileされているのでtensorboardを含むCPUを使う処理はかけない
        """
        imgs = data["imgs"]
        msks = data["msks"]
        img_msks = self._get_img_msks(msks, self.cfg.bit_info.padding_bit)
        heart_msks = self._get_heart_msks(msks, self.cfg.bit_info.heart_bit)
        target_imgs = data["target_imgs"]

        min_clip_vals = data["min_clip_vals"]
        max_clip_vals = data["max_clip_vals"]
        target_min_clip_vals = data["target_min_clip_vals"]
        target_max_clip_vals = data["target_max_clip_vals"]
        # ここでGPUを使ったデータの正規化やデータ拡張を行う
        (imgs, target_imgs, observed_slice_msks, missing_slice_msks, loss_msks) = (
            self.prepare_training_batch(
                imgs,
                target_imgs,
                img_msks,
                min_clip_vals,
                max_clip_vals,
                target_min_clip_vals,
                target_max_clip_vals,
                self.cfg,
                heart_msks=heart_msks,
            )
        )
        img_msks = self.center_crop_to_model_size(img_msks, self.cfg)
        heart_msks = self.center_crop_to_model_size(heart_msks, self.cfg)
        with GradientTape() as tape:
            rfr_target = self.make_rfr_target(imgs, target_imgs, self.cfg)
            t = self.sample_rfr_time(target_imgs, self.cfg)
            eps = tf.random.normal(tf.shape(rfr_target), dtype=rfr_target.dtype)
            noisy_target = (1.0 - t) * rfr_target + t * eps
            state_msk = self.get_rfr_state_mask(img_msks, observed_slice_msks, self.cfg)
            noisy_target *= state_msk

            # モデルのフォワード&バックワードパス
            pred_state = self(
                [
                    self.concat_i2i_input(
                        imgs, noisy_target, observed_slice_msks=observed_slice_msks
                    ),
                    img_msks,
                ],
                training=True,
            )
            pred_state *= state_msk
            preds = self.reconstruct_rfr_prediction(imgs, pred_state, self.cfg)
            preds = self.preserve_observed_slices(
                imgs, preds, observed_slice_msks, self.cfg
            )

            total_loss = self._compute_rfr_total_loss(
                target_imgs,
                preds,
                loss_msks,
                t,
                img_msks=img_msks,
                missing_slice_msks=missing_slice_msks,
            )
        trainable_weights = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        metric_msks = (
            missing_slice_msks if self.is_slice_completion(self.cfg) else img_msks
        )
        self._update_metrics(
            target_imgs,
            preds,
            metric_msks,
            t=t,
            valid_img_msks=img_msks,
            missing_slice_msks=missing_slice_msks,
        )

        return self._get_metrics_result(include_evaluation=False)

    def test_step(self, data):
        """
        ここはjit_compileされているのでtensorboardを含むCPUを使う処理はかけない
        ./callbacks/image_logger.pyを参考にコールバックを実装する
        """

        img_msks = self.center_crop_to_model_size(
            self._get_img_msks(data["msks"], self.cfg.bit_info.padding_bit), self.cfg
        )
        heart_msks = self.center_crop_to_model_size(
            self._get_heart_msks(data["msks"], self.cfg.bit_info.heart_bit), self.cfg
        )
        initial_noise = self._make_validation_initial_noise(data["imgs"])
        prediction_outputs = self.predict_step(
            data,
            return_aux=True,
            apply_self_supervised_blur=True,
            initial_noise=initial_noise,
        )
        logits, _, source_imgs, target_imgs = prediction_outputs[:4]
        missing_slice_msks = (
            prediction_outputs[5] if self.is_slice_completion(self.cfg) else img_msks
        )
        metric_msks = (
            missing_slice_msks if self.is_slice_completion(self.cfg) else img_msks
        )
        self._update_metrics(
            target_imgs,
            logits,
            metric_msks,
            t=None,
            heart_msks=heart_msks,
            intensity_range=(
                data["target_max_clip_vals"] - data["target_min_clip_vals"]
            ),
            update_evaluation=True,
            valid_img_msks=img_msks,
            missing_slice_msks=missing_slice_msks,
        )
        if self.is_slice_completion(self.cfg):
            baseline_mae = self.masked_mean_values_per_sample(
                tf.abs(target_imgs - source_imgs), missing_slice_msks, target_imgs
            )
            intensity_range = tf.reshape(
                tf.cast(
                    data["target_max_clip_vals"] - data["target_min_clip_vals"],
                    baseline_mae.dtype,
                ),
                (-1,),
            )
            self.metrics_dict["linear_baseline_mae_hu"].update_state(
                baseline_mae * intensity_range
            )

        return self._get_metrics_result(include_evaluation=True)

    def _compute_rfr_total_loss(
        self, target_imgs, preds, loss_msks, t, img_msks=None, missing_slice_msks=None
    ):
        p = float(self.cfg.i2i_rfr.p)
        abs_error = ops.abs(target_imgs - preds)
        if p == 1.0:
            pixel_error = abs_error
        elif p == 2.0:
            pixel_error = ops.square(abs_error)
        else:
            pixel_error = ops.power(abs_error, p)
        weighted_error = pixel_error / ops.power(t, p)
        if self.is_slice_completion(self.cfg):
            rfr_loss = self.masked_mean_per_sample(
                weighted_error, loss_msks, target_imgs
            )
        else:
            denominator = self.masked_denominator(loss_msks, target_imgs)
            rfr_loss = ops.sum(weighted_error * loss_msks) / denominator
        gradient_loss = self.compute_gradient_loss(target_imgs, preds, loss_msks, t=t)
        gradient_weight = float(
            getattr(getattr(self.cfg.loss, "gradient", None), "weight", 0.0)
        )
        total = self.cfg.loss.rfr.weight * rfr_loss + gradient_weight * gradient_loss
        if self.is_slice_completion(self.cfg):
            total += float(self.cfg.loss.slice_completion.z_gradient_weight) * (
                self.compute_z_gradient_loss(
                    target_imgs, preds, img_msks, missing_slice_msks
                )
            )
        return total

    def _update_metrics(
        self,
        target_imgs,
        preds,
        img_msks,
        t=None,
        heart_msks=None,
        intensity_range=None,
        update_evaluation=False,
        valid_img_msks=None,
        missing_slice_msks=None,
    ):
        target_imgs = tf.stop_gradient(target_imgs)
        preds = tf.stop_gradient(preds)
        img_msks = tf.stop_gradient(img_msks)
        if t is not None:
            t = tf.stop_gradient(t)

        denominator = self.masked_denominator(img_msks, target_imgs)
        abs_error = ops.abs(target_imgs - preds)
        sq_error = ops.square(target_imgs - preds)

        mae = ops.sum(abs_error * img_msks) / denominator
        mse = ops.sum(sq_error * img_msks) / denominator
        psnr = -10.0 * tf.math.log(tf.maximum(mse, 1e-8)) / tf.math.log(10.0)

        if t is None:
            rfr_loss = mae
        else:
            p = float(self.cfg.i2i_rfr.p)
            if p == 1.0:
                pixel_error = abs_error
            elif p == 2.0:
                pixel_error = sq_error
            else:
                pixel_error = ops.power(abs_error, p)
            rfr_loss = ops.sum(pixel_error / ops.power(t, p) * img_msks) / denominator

        gradient_loss = self.compute_gradient_loss(target_imgs, preds, img_msks, t=t)
        gradient_weight = float(
            getattr(getattr(self.cfg.loss, "gradient", None), "weight", 0.0)
        )
        total_loss = (
            self.cfg.loss.rfr.weight * rfr_loss + gradient_weight * gradient_loss
        )
        z_gradient_loss = tf.cast(0.0, target_imgs.dtype)
        if self.is_slice_completion(self.cfg):
            z_gradient_loss = self.compute_z_gradient_loss(
                target_imgs, preds, valid_img_msks, missing_slice_msks
            )
            total_loss += (
                float(self.cfg.loss.slice_completion.z_gradient_weight)
                * z_gradient_loss
            )
        self.metrics_dict["rfr_loss"].update_state(rfr_loss)
        self.metrics_dict["gradient_loss"].update_state(gradient_loss)
        self.metrics_dict["z_gradient_loss"].update_state(z_gradient_loss)
        self.metrics_dict["mae"].update_state(mae)
        self.metrics_dict["mse"].update_state(mse)
        self.metrics_dict["psnr"].update_state(psnr)
        self.metrics_dict["total_loss"].update_state(total_loss)

        if update_evaluation:
            self._update_evaluation_metrics(
                target_imgs, preds, img_msks, heart_msks, intensity_range
            )

        return total_loss

    def _update_evaluation_metrics(
        self, target_imgs, preds, img_msks, heart_msks, intensity_range
    ):
        """Update final-image validation metrics; these are not train-step metrics."""
        if heart_msks is None or intensity_range is None:
            raise ValueError(
                "heart_msks and intensity_range are required for evaluation metrics"
            )

        metric_cfg = self.cfg.evaluation_metrics
        heart_msks = tf.cast(heart_msks, tf.float32) * tf.cast(img_msks, tf.float32)

        def update(name, values, valid):
            self.metrics_dict[name].update_state(
                values, sample_weight=tf.cast(valid, tf.float32)
            )

        global_ssim, global_valid = masked_ssim_xy(
            target_imgs,
            preds,
            img_msks,
            max_val=1.0,
            filter_size=int(metric_cfg.ssim_filter_size),
            filter_sigma=float(metric_cfg.ssim_filter_sigma),
        )
        heart_ssim, heart_ssim_valid = masked_ssim_xy(
            target_imgs,
            preds,
            heart_msks,
            max_val=1.0,
            filter_size=int(metric_cfg.ssim_filter_size),
            filter_sigma=float(metric_cfg.ssim_filter_sigma),
        )
        heart_psnr, heart_psnr_valid = masked_psnr(
            target_imgs, preds, heart_msks, max_val=1.0
        )
        heart_mae_hu, heart_mae_valid = masked_mae_with_scale(
            target_imgs, preds, heart_msks, intensity_range
        )
        z_gradient_mae, z_gradient_valid = masked_z_gradient_mae(
            target_imgs, preds, heart_msks
        )
        edge_ratio, edge_ratio_valid = masked_xy_edge_strength_ratio(
            target_imgs, preds, heart_msks, epsilon=float(metric_cfg.edge_epsilon)
        )

        update("ssim_xy_global", global_ssim, global_valid)
        update("ssim_xy_heart", heart_ssim, heart_ssim_valid)
        update("psnr_heart", heart_psnr, heart_psnr_valid)
        update("mae_hu_heart", heart_mae_hu, heart_mae_valid)
        update("z_gradient_mae", z_gradient_mae, z_gradient_valid)
        update("xy_edge_strength_ratio", edge_ratio, edge_ratio_valid)

    def _make_validation_initial_noise(self, imgs):
        """Use identical validation noise across epochs for comparable metrics."""
        target_shape = tf.concat(
            [tf.shape(imgs)[:-1], tf.constant([self.cfg.model.num_channel], tf.int32)],
            axis=0,
        )
        seed = int(self.cfg.evaluation_metrics.validation_seed)
        return tf.random.stateless_normal(
            target_shape, seed=[seed, 0], dtype=tf.float32
        )

    def predict_step(
        self,
        data,
        return_aux=False,
        apply_self_supervised_blur=False,
        initial_noise=None,
    ):
        imgs = data["imgs"]
        msks = data["msks"]
        img_msks = self._get_img_msks(msks, self.cfg.bit_info.padding_bit)

        min_clip_vals = data["min_clip_vals"]
        max_clip_vals = data["max_clip_vals"]

        # 画像の正規化
        imgs = normalize(imgs, min_clip_vals, max_clip_vals)
        imgs = imgs * img_msks
        observed_slice_msks = None
        missing_slice_msks = img_msks
        if self.is_slice_completion(self.cfg):
            if "observed_slice_msks" in data:
                observed_slice_msks = tf.cast(data["observed_slice_msks"], imgs.dtype)
                missing_slice_msks = (1.0 - observed_slice_msks) * img_msks
            else:
                (imgs, observed_slice_msks, missing_slice_msks, _) = (
                    self.apply_self_supervised_slice_completion(
                        imgs, img_msks, self.cfg, is_training=False
                    )
                )
        elif apply_self_supervised_blur:
            heart_msks = self._get_heart_msks(msks, self.cfg.bit_info.heart_bit)
            imgs = self.apply_self_supervised_deblur(
                imgs, img_msks, self.cfg, is_training=False, heart_msks=heart_msks
            )
        imgs = self.center_crop_to_model_size(imgs, self.cfg)
        img_msks = self.center_crop_to_model_size(img_msks, self.cfg)
        if observed_slice_msks is not None:
            observed_slice_msks = self.center_crop_to_model_size(
                observed_slice_msks, self.cfg
            )
        missing_slice_msks = self.center_crop_to_model_size(
            missing_slice_msks, self.cfg
        )
        if initial_noise is not None:
            initial_noise = self.center_crop_to_model_size(initial_noise, self.cfg)
        preds = self.i2i_rfr_inference(
            imgs,
            img_msks,
            initial_noise=initial_noise,
            observed_slice_msks=observed_slice_msks,
        )

        if return_aux:
            target_imgs = self.normalize_target(
                self.center_crop_to_model_size(data["target_imgs"], self.cfg),
                img_msks,
                data["target_min_clip_vals"],
                data["target_max_clip_vals"],
            )
            # callbacks/image_logger.pyで必要とするものも返す
            if self.is_slice_completion(self.cfg):
                return (
                    preds,
                    preds,
                    imgs,
                    target_imgs,
                    observed_slice_msks,
                    missing_slice_msks,
                )
            return preds, preds, imgs, target_imgs
        else:
            return preds

    @staticmethod
    def concat_i2i_input(imgs, target_state, observed_slice_msks=None):
        inputs = [imgs]
        if observed_slice_msks is not None:
            inputs.append(tf.cast(observed_slice_msks, imgs.dtype))
        inputs.append(target_state)
        return ops.concatenate(inputs, axis=-1)

    @staticmethod
    def get_prediction_type(cfg):
        """Resolve the RFR prediction space while accepting old output configs."""
        return str(getattr(cfg.i2i_rfr, "prediction_type", "image"))

    @classmethod
    def make_rfr_target(cls, imgs, target_imgs, cfg):
        """Return the clean endpoint represented as an image or source residual."""
        if cls.get_prediction_type(cfg) == "residual":
            return target_imgs - imgs
        return target_imgs

    @classmethod
    def reconstruct_rfr_prediction(cls, imgs, prediction_state, cfg):
        """Map the model prediction space back to a reconstructed target image."""
        if cls.get_prediction_type(cfg) == "residual":
            return imgs + prediction_state
        return prediction_state

    def compute_gradient_loss(self, target_imgs, preds, img_msks, t=None):
        """Compute the auxiliary gradient loss with optional legacy t weighting."""
        gradient_cfg = getattr(self.cfg.loss, "gradient", None)
        time_reweight = bool(getattr(gradient_cfg, "time_reweight", True))
        time_weight = t if time_reweight else None
        if self.is_slice_completion(self.cfg):
            return self.compute_xy_gradient_loss_per_sample(
                target_imgs, preds, img_msks, time_weight=time_weight
            )
        return masked_xy_gradient_loss(
            target_imgs, preds, img_msks, time_weight=time_weight
        )

    @staticmethod
    def compute_xy_gradient_loss_per_sample(
        target_imgs, preds, img_msks, time_weight=None
    ):
        """Per-case XY gradient loss used when batches mix sparse factors."""

        def axis_loss(axis):
            if axis == 2:
                target_gradient = target_imgs[:, :, 1:] - target_imgs[:, :, :-1]
                pred_gradient = preds[:, :, 1:] - preds[:, :, :-1]
                pair_mask = img_msks[:, :, 1:] * img_msks[:, :, :-1]
            else:
                target_gradient = target_imgs[:, :, :, 1:] - target_imgs[:, :, :, :-1]
                pred_gradient = preds[:, :, :, 1:] - preds[:, :, :, :-1]
                pair_mask = img_msks[:, :, :, 1:] * img_msks[:, :, :, :-1]
            error = tf.abs(target_gradient - pred_gradient)
            if time_weight is not None:
                error /= tf.maximum(time_weight, tf.cast(1e-6, error.dtype))
            return CustomModel.masked_mean_per_sample(error, pair_mask, target_gradient)

        return 0.5 * (axis_loss(2) + axis_loss(3))

    @staticmethod
    def compute_z_gradient_loss(target_imgs, preds, img_msks, missing_slice_msks):
        """Compare adjacent Z gradients for pairs touching a missing slice."""
        target_gradient = target_imgs[:, 1:] - target_imgs[:, :-1]
        pred_gradient = preds[:, 1:] - preds[:, :-1]
        valid_pair = img_msks[:, 1:] * img_msks[:, :-1]
        missing_pair = tf.maximum(missing_slice_msks[:, 1:], missing_slice_msks[:, :-1])
        pair_msk = valid_pair * missing_pair
        return CustomModel.masked_mean_per_sample(
            ops.abs(target_gradient - pred_gradient), pair_msk, target_gradient
        )

    @staticmethod
    def masked_mean_per_sample(values, msks, target_imgs):
        """Normalize each case independently so sparse factors have equal weight."""
        return tf.reduce_mean(
            CustomModel.masked_mean_values_per_sample(values, msks, target_imgs)
        )

    @staticmethod
    def masked_mean_values_per_sample(values, msks, target_imgs):
        """Return one independently normalized masked mean per batch item."""
        values = tf.convert_to_tensor(values)
        msks = tf.cast(msks, values.dtype)
        num_channels = tf.cast(tf.shape(target_imgs)[-1], values.dtype)
        reduce_axes = tuple(range(1, values.shape.rank))
        numerator = tf.reduce_sum(values * msks, axis=reduce_axes)
        denominator = tf.maximum(
            tf.reduce_sum(msks, axis=reduce_axes) * num_channels,
            tf.cast(1.0, values.dtype),
        )
        return numerator / denominator

    @staticmethod
    def masked_denominator(img_msks, target_imgs):
        num_channel = tf.cast(tf.shape(target_imgs)[-1], tf.float32)
        return tf.maximum(tf.reduce_sum(img_msks) * num_channel, 1.0)

    @staticmethod
    def is_slice_completion(cfg):
        return (
            str(getattr(cfg, "training_mode", "paired"))
            == "self_supervised_slice_completion"
        )

    @classmethod
    def get_rfr_state_mask(cls, img_msks, observed_slice_msks, cfg):
        if not cls.is_slice_completion(cfg):
            return img_msks
        if observed_slice_msks is None:
            raise ValueError("slice completion requires observed_slice_msks")
        if bool(cfg.self_supervised_slice_completion.preserve_observed_slices):
            return (1.0 - observed_slice_msks) * img_msks
        return img_msks

    @classmethod
    def preserve_observed_slices(cls, source, preds, observed_slice_msks, cfg):
        if not cls.is_slice_completion(cfg) or not bool(
            cfg.self_supervised_slice_completion.preserve_observed_slices
        ):
            return preds
        if observed_slice_msks is None:
            raise ValueError("slice completion requires observed_slice_msks")
        return observed_slice_msks * source + (1.0 - observed_slice_msks) * preds

    @staticmethod
    def normalize_target(target_imgs, img_msks, min_clip_vals, max_clip_vals):
        target_imgs = normalize(target_imgs, min_clip_vals, max_clip_vals)
        target_imgs = target_imgs * img_msks
        return target_imgs

    @staticmethod
    def sample_rfr_time(target_imgs, cfg):
        batch_size = tf.shape(target_imgs)[0]
        u = tf.random.uniform((batch_size, 1, 1, 1, 1), 0.0, 1.0)
        p = tf.cast(cfg.i2i_rfr.p, target_imgs.dtype)
        t = tf.pow(u, 1.0 / (p + 1.0))
        t_min = tf.cast(cfg.i2i_rfr.t_min, target_imgs.dtype)
        return tf.maximum(t, t_min)

    def i2i_rfr_inference(
        self, imgs, img_msks, initial_noise=None, observed_slice_msks=None
    ):
        steps = int(self.cfg.i2i_rfr.inference_steps)
        dt = tf.cast(1.0 / steps, imgs.dtype)
        target_shape = tf.concat(
            [tf.shape(imgs)[:-1], tf.constant([self.cfg.model.num_channel])], axis=0
        )
        if initial_noise is None:
            target_state = tf.random.normal(target_shape, dtype=imgs.dtype)
        else:
            target_state = tf.cast(initial_noise, imgs.dtype)
            tf.debugging.assert_equal(tf.shape(target_state), target_shape)
        state_msk = self.get_rfr_state_mask(img_msks, observed_slice_msks, self.cfg)
        target_state *= state_msk

        for n in range(steps):
            t = tf.cast(1.0 - n / steps, imgs.dtype)
            pred_x0 = self(
                [
                    self.concat_i2i_input(
                        imgs, target_state, observed_slice_msks=observed_slice_msks
                    ),
                    img_msks,
                ],
                training=False,
            )
            pred_x0 *= state_msk
            velocity = (target_state - pred_x0) / t
            target_state = (target_state - dt * velocity) * state_msk

        preds = self.reconstruct_rfr_prediction(imgs, target_state, self.cfg)
        if self.cfg.i2i_rfr.clip_output:
            preds = ops.clip(preds, 0, 1)
        preds = self.preserve_observed_slices(
            imgs, preds, observed_slice_msks, self.cfg
        )
        return preds * img_msks

    @staticmethod
    def gpu_shared_signal_aug(imgs, img_msks, min_clip_vals, max_clip_vals, cfg):
        """Apply signal transforms that are safe to share between source/target."""
        imgs = random_normalize(
            imgs, min_clip_vals, max_clip_vals, **cfg.aug.random_normalize
        )
        imgs = random_gamma_correction(imgs, **cfg.aug.random_gamma_correction)
        imgs = ops.clip(imgs, 0, 1)
        return imgs * img_msks

    @staticmethod
    def gpu_source_artifact_aug(imgs, img_msks, cfg):
        """Apply sharpness/blur/noise to source only, never to a clean target."""
        imgs = apply_random_sharpness_or_gaussian_filter(
            imgs,
            cfg.aug.random_sharpness.prob,
            cfg.aug.random_sharpness.sigma,
            cfg.aug.random_sharpness.alpha_range,
            cfg.aug.random_gauss_filter.prob,
            cfg.aug.random_gauss_filter.sigma_range,
        )

        imgs = apply_random_gaussian_noise(imgs, **cfg.aug.random_gauss_noise)
        imgs = ops.clip(imgs, 0, 1)
        return imgs * img_msks

    @classmethod
    def gpu_aug(cls, imgs, img_msks, min_clip_vals, max_clip_vals, cfg):
        """Apply the complete source augmentation pipeline for paired training."""
        imgs = cls.gpu_shared_signal_aug(
            imgs, img_msks, min_clip_vals, max_clip_vals, cfg
        )
        return cls.gpu_source_artifact_aug(imgs, img_msks, cfg)

    @staticmethod
    def mix_identity_samples(degraded_imgs, clean_imgs, cfg, is_training):
        """Replace random training samples with exact clean/source identity pairs."""
        probability = float(
            getattr(cfg.self_supervised_deblur, "identity_probability", 0.0)
        )
        if not is_training or probability <= 0:
            return degraded_imgs
        if probability >= 1:
            return clean_imgs
        batch_size = tf.shape(clean_imgs)[0]
        use_identity = tf.random.uniform((batch_size, 1, 1, 1, 1)) < probability
        return tf.where(use_identity, clean_imgs, degraded_imgs)

    @classmethod
    def prepare_training_batch(
        cls,
        imgs,
        target_imgs,
        img_msks,
        min_clip_vals,
        max_clip_vals,
        target_min_clip_vals,
        target_max_clip_vals,
        cfg,
        heart_msks=None,
    ):
        """Return model images and task masks for one training batch."""
        if cls.is_slice_completion(cfg):
            target_imgs = cls.gpu_shared_signal_aug(
                target_imgs, img_msks, target_min_clip_vals, target_max_clip_vals, cfg
            )
            (imgs, observed_slice_msks, missing_slice_msks, _) = (
                cls.apply_self_supervised_slice_completion(
                    target_imgs, img_msks, cfg, is_training=True
                )
            )
            imgs = cls.center_crop_to_model_size(imgs, cfg)
            target_imgs = cls.center_crop_to_model_size(target_imgs, cfg)
            observed_slice_msks = cls.center_crop_to_model_size(
                observed_slice_msks, cfg
            )
            missing_slice_msks = cls.center_crop_to_model_size(missing_slice_msks, cfg)
            missing_weight = float(cfg.loss.slice_completion.missing_weight)
            observed_weight = float(cfg.loss.slice_completion.observed_weight)
            loss_msks = (
                missing_weight * missing_slice_msks
                + observed_weight * observed_slice_msks
            )
            return (
                imgs,
                target_imgs,
                observed_slice_msks,
                missing_slice_msks,
                loss_msks,
            )

        imgs, target_imgs = cls.prepare_training_images(
            imgs,
            target_imgs,
            img_msks,
            min_clip_vals,
            max_clip_vals,
            target_min_clip_vals,
            target_max_clip_vals,
            cfg,
            heart_msks=heart_msks,
        )
        img_msks = cls.center_crop_to_model_size(img_msks, cfg)
        return imgs, target_imgs, None, img_msks, img_msks

    @classmethod
    def prepare_training_images(
        cls,
        imgs,
        target_imgs,
        img_msks,
        min_clip_vals,
        max_clip_vals,
        target_min_clip_vals,
        target_max_clip_vals,
        cfg,
        heart_msks=None,
    ):
        """Prepare source/target while keeping self-supervised signals aligned."""
        training_mode = str(getattr(cfg, "training_mode", "paired"))
        if training_mode == "self_supervised_deblur":
            # 正規化/gammaだけをclean targetとsourceで共有する。Sharpness、追加blur、
            # noiseはclean targetを汚さないよう、劣化後のsourceだけへ適用する。
            target_imgs = cls.gpu_shared_signal_aug(
                target_imgs, img_msks, target_min_clip_vals, target_max_clip_vals, cfg
            )
            imgs = tf.identity(target_imgs)
            imgs = cls.apply_self_supervised_deblur(
                imgs, img_msks, cfg, is_training=True, heart_msks=heart_msks
            )
            imgs = cls.gpu_source_artifact_aug(imgs, img_msks, cfg)
            imgs = cls.mix_identity_samples(imgs, target_imgs, cfg, is_training=True)
            imgs = cls.center_crop_to_model_size(imgs, cfg)
            target_imgs = cls.center_crop_to_model_size(target_imgs, cfg)
        else:
            imgs = cls.gpu_aug(imgs, img_msks, min_clip_vals, max_clip_vals, cfg)
            target_imgs = cls.normalize_target(
                target_imgs, img_msks, target_min_clip_vals, target_max_clip_vals
            )
        return imgs, target_imgs

    @staticmethod
    def apply_self_supervised_deblur(imgs, img_msks, cfg, is_training, heart_msks=None):
        """Synthesize a blurred source while leaving the target unchanged."""
        training_mode = str(getattr(cfg, "training_mode", "paired"))
        if training_mode != "self_supervised_deblur":
            return imgs

        degradation_type = str(
            getattr(cfg.self_supervised_deblur, "degradation_type", "gaussian")
        )
        if degradation_type in ["cardiac_motion", "cardiac_motion_gaussian"]:
            imgs = cardiac_motion_blur(
                imgs,
                img_msks,
                spacing_mm_yx=cfg.aug.affine.norm_spacing_zyx[1:3],
                motion_msks=heart_msks,
                is_training=is_training,
                **cfg.self_supervised_deblur.cardiac_motion,
            )
        if degradation_type in ["gaussian", "cardiac_motion_gaussian"]:
            if is_training:
                imgs = random_gaussian_filter(
                    imgs, sigma_range=cfg.self_supervised_deblur.sigma_range
                )
            else:
                imgs = gaussian_filter(
                    imgs, sigma=float(cfg.self_supervised_deblur.validation_sigma)
                )
        slice_thickness_cfg = getattr(
            cfg.self_supervised_deblur, "slice_thickness", None
        )
        if slice_thickness_cfg is not None and slice_thickness_cfg.enabled:
            imgs = simulate_slice_thickness(
                imgs,
                img_msks,
                spacing_mm_z=cfg.aug.affine.norm_spacing_zyx[0],
                **slice_thickness_cfg,
            )
        return imgs * img_msks

    @staticmethod
    def apply_self_supervised_slice_completion(imgs, img_msks, cfg, is_training):
        """Remove dense Z slices and interpolate only from retained slices."""
        if not CustomModel.is_slice_completion(cfg):
            raise ValueError(
                "apply_self_supervised_slice_completion requires "
                "training_mode=self_supervised_slice_completion"
            )
        completion_cfg = cfg.self_supervised_slice_completion
        return simulate_slice_completion(
            imgs,
            img_msks,
            spacing_mm_z=float(cfg.aug.affine.norm_spacing_zyx[0]),
            keep_every_n_values=completion_cfg.keep_every_n_values,
            sampling_weights=getattr(completion_cfg, "sampling_weights", None),
            random_offset=bool(completion_cfg.random_offset),
            validation_factor=int(completion_cfg.validation_factor),
            validation_offset=int(completion_cfg.validation_offset),
            fill_mode=str(completion_cfg.fill_mode),
            slice_profile_blur=completion_cfg.slice_profile_blur,
            is_training=is_training,
        )

    def _get_metrics_result(self, include_evaluation=True):
        """
        Return the results of all metrics as a dictionary.
        """
        return {
            metric.name: metric.result()
            for metric in self.metrics_dict.values()
            if include_evaluation
            or metric.name
            not in (
                *self.EVALUATION_METRIC_NAMES,
                *self.SLICE_COMPLETION_EVALUATION_METRIC_NAMES,
            )
        }

    @property
    def metrics(self):
        """
        We list our `Metric` objects here so that `reset_states()` can be
        called automatically at the start of each epoch
        or at the start of `evaluate()`.
        """
        return self.metrics_dict.values()
