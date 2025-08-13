import tensorflow as tf
from keras.layers import Layer
import keras_cv


class CustomGaussianNoise(Layer):
    def __init__(self, stddev=None, min_stddev=None, max_stddev=None, **kwargs):
        super().__init__(**kwargs)

        if stddev is not None:
            if min_stddev is not None or max_stddev is not None:
                raise ValueError("If `stddev` is defined, `min_stddev` and `max_stddev` cannot be defined.")
            if not 0 <= stddev <= 1:
                raise ValueError(f"Invalid value for `stddev`. Expected a float between 0 and 1. Received: {stddev}")
            self.stddev = stddev
            self.min_stddev = None
            self.max_stddev = None
        elif max_stddev is not None:
            self.min_stddev = 0.0 if min_stddev is None else min_stddev
            self.max_stddev = max_stddev
            if not 0.0 <= self.min_stddev <= 1.0 or not 0.0 <= self.max_stddev <= 1.0:
                raise ValueError("`min_stddev` and `max_stddev` must be in [0.0, 1.0]")
            if self.min_stddev > self.max_stddev:
                raise ValueError("`min_stddev` must be less than or equal to `max_stddev`.")
            self.stddev = None
        else:
            raise ValueError("You must provide either `stddev` or `max_stddev` (and optionally `min_stddev`).")

        self.random_generator = tf.random.Generator.from_non_deterministic_state()
        self.supports_masking = True
        self.built = True
        self.last_stddev = None

    def call(self, inputs, training=False):
        if not training:
            return inputs
        return self._apply_gaussian_noise(inputs)

    @tf.function
    def _apply_gaussian_noise(self, inputs):
        stddev = (
            self.stddev
            if self.stddev is not None
            else self.random_generator.uniform([], self.min_stddev, self.max_stddev)
        )

        self.last_stddev = stddev

        noise = tf.random.normal(
            shape=tf.shape(inputs),
            mean=0.0,
            stddev=stddev,
            dtype=inputs.dtype,
        )
        return inputs + noise

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "stddev": self.stddev,
            "min_stddev": getattr(self, "min_stddev", None),
            "max_stddev": getattr(self, "max_stddev", None),
        }
        return {**base_config, **config}


class RandomSaltAndPepper(Layer):
    def __init__(self, factor=None, min_factor=None, max_factor=None, seed=None, **kwargs):
        super(RandomSaltAndPepper, self).__init__(**kwargs)
        if factor is not None:
            if min_factor is not None or max_factor is not None:
                raise ValueError("If 'factor' is defined, 'min_factor' and 'max_factor' cannot be defined.")
            self.factor = factor
            self.min_factor = None
            self.max_factor = None
        elif max_factor is not None:
            self.min_factor = 0.0 if min_factor is None else min_factor
            self.max_factor = max_factor
            if not 0.0 <= self.min_factor <= 1.0 or not 0.0 <= self.max_factor <= 1.0:
                raise ValueError("'min_factor' and 'max_factor' must be in the range [0.0, 1.0].")
            if self.min_factor > self.max_factor:
                raise ValueError("'min_factor' must be less than or equal to 'max_factor'.")
            self.factor = None
        else:
            raise ValueError("Either 'factor' must be defined, or 'max_factor' must be.")

        self.seed = seed
        self.random_generator = tf.random.Generator.from_seed(seed) if seed else tf.random.Generator.from_non_deterministic_state()

    def call(self, inputs, training=None):
        if not training:
            return inputs
        return self._add_salt_and_pepper(inputs)

    @tf.function
    def _add_salt_and_pepper(self, inputs):
        factor = (
            self.factor
            if self.factor is not None
            else self.random_generator.uniform(shape=[], minval=self.min_factor, maxval=self.max_factor)
        )

        shape = tf.shape(inputs)
        random_values = self.random_generator.uniform(shape, minval=0.0, maxval=1.0)

        salt_mask = random_values > (1 - factor / 2)
        pepper_mask = random_values < (factor / 2)

        outputs = tf.where(salt_mask, 1.0, inputs)
        outputs = tf.where(pepper_mask, 0.0, outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "factor": self.factor,
            "min_factor": self.min_factor,
            "max_factor": self.max_factor,
            "seed": self.seed,
        })
        return config


class RandAugmentGaussian(Layer):
    def __init__(self, num_layers=2, magnitude=9, gaussian_stddev=0.05, **kwargs):
        """
        Combines Keras RandAugment with CustomGaussianNoise.

        Args:
            num_layers (int): Number of RandAugment transformations.
            magnitude (int): Magnitude of RandAugment.
            gaussian_stddev (float): Stddev for Gaussian noise.
        """
        super().__init__(**kwargs)
        self.randaugment = keras_cv.layers.RandAugment(num_layers=num_layers, magnitude=magnitude)
        self.gaussian_noise = CustomGaussianNoise(stddev=gaussian_stddev)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        x = self.randaugment(inputs, training=training)
        x = self.gaussian_noise(x, training=training)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "magnitude": self.randaugment.magnitude,
            "gaussian_stddev": self.gaussian_noise.stddev,
        }
        return {**base_config, **config}
