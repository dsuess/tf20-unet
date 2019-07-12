from typing import List, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras as k


class UnetEncoder(k.Model):
    # TODO Make downsampling layer configurable

    def __init__(self, num_filters: List[int], **kwargs: Any):
        super().__init__()
        assert len(num_filters) >= 1
        self.blocks = [
            self.build_block(f, name=f'block_{i}', **kwargs)
            for i, f in enumerate(num_filters)]
        self.downsamplers = [
            k.layers.MaxPool2D(padding='same', name=f'downsample_{i}')
            for i in range(len(num_filters))]

    @staticmethod
    def build_block(num_filters: int, name: str, **kwargs: Any) -> k.Sequential:
        return k.Sequential([
            k.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', **kwargs),
            k.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', **kwargs)],
            name=name)

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        >>> x = tf.zeros((1, 32, 32, 3))
        >>> model = UnetEncoder(num_filters=[64])
        >>> y, ys = model(x)
        >>> tuple(y.shape), [tuple(yi.shape) for yi in ys]
        ((1, 16, 16, 64), [(1, 32, 32, 64)])

        >>> x = tf.zeros((1, 32, 32, 3))
        >>> model = UnetEncoder(num_filters=[64, 32])
        >>> y, ys = model(x)
        >>> tuple(y.shape), [tuple(yi.shape) for yi in ys]
        ((1, 8, 8, 32), [(1, 32, 32, 64), (1, 16, 16, 32)])
        """
        intermediate = []
        y = x
        for block, downsample in zip(self.blocks, self.downsamplers):
            y = block(y)
            intermediate.append(y)
            y = downsample(y)
        return y, intermediate


class UnetDecoder(k.Model):
    # TODO Make upsamplers configurable

    def __init__(self, num_filters: List[int], **kwargs: Any):
        super().__init__()
        assert len(num_filters) >= 1
        self.blocks = [
            self.build_block(f, name=f'block_{i}', **kwargs)
            for i, f in enumerate(num_filters)]
        # We consider a block to be consisting of conv + upsample (instead of
        # upsample + decode) since otherwise we'd need to pass through the
        # number of features for the previous channel as well when switching
        # to transpose convolutions
        self.upsamplers = [
            k.layers.UpSampling2D(name=f'upsample_{i}')
            for i in range(len(num_filters) - 1)]
        self.upsamplers += [lambda x: x]

    @staticmethod
    def build_block(num_filters: int, name: str, **kwargs: Any) -> k.Sequential:
        return k.Sequential([
            k.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', **kwargs),
            k.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', **kwargs)],
            name=name)

    def call(self, initial: tf.Tensor, intermediates: List[tf.Tensor]) \
        -> Tuple[tf.Tensor, tf.Tensor]:
        """
        >>> x = (tf.zeros((1, 32, 32, 3)), [tf.zeros((1, 32, 32, 3))])
        >>> model = UnetDecoder(num_filters=[64])
        >>> tuple(model(*x).shape)
        (1, 32, 32, 64)

        >>> x = (tf.zeros((1, 32, 32, 3)), [tf.zeros((1, 32, 32, 3)), tf.zeros((1, 64, 64, 3))])
        >>> model = UnetDecoder(num_filters=[4, 8])
        >>> tuple(model(*x).shape)
        (1, 64, 64, 8)
        """
        assert len(intermediates) == len(self.blocks) == len(self.upsamplers)

        y_u = initial
        iterator = zip(intermediates, self.blocks, self.upsamplers)
        for intermediate, block, upsampler in iterator:
            x = k.backend.concatenate((intermediate, y_u))
            y = block(x)
            y_u = upsampler(y)

        return y  # == y_u


class Unet(k.Model):

    def __init__(self, output_channels, num_filters: List[int], **kwargs):
        super().__init__()
        if 'activation' not in kwargs:
            kwargs['activation'] = tf.nn.leaky_relu
        *enc_features, bottleneck_features = num_filters
        self.encoder = UnetEncoder(list(enc_features), **kwargs)
        self.decoder = UnetDecoder(list(reversed(enc_features)), **kwargs)
        self.bottleneck = self.build_bottleneck(bottleneck_features, **kwargs)
        self.tail = self.build_tail(enc_features[0], output_channels, **kwargs)

    @staticmethod
    def build_bottleneck(features: int, **kwargs: Any) -> k.Sequential:
        return k.Sequential([
            k.layers.Conv2D(filters=features, kernel_size=3, padding='same', **kwargs),
            k.layers.Conv2D(filters=features, kernel_size=3, padding='same', **kwargs),
            k.layers.UpSampling2D()],
            name='bottleneck')

    @staticmethod
    def build_tail(features: int, output_channels: int, **kwargs: Any) -> k.Sequential:
        layers = [
            k.layers.Conv2D(filters=features, kernel_size=3, padding='same', **kwargs),
            k.layers.Conv2D(filters=features, kernel_size=3, padding='same', **kwargs)]
        kwargs = {**kwargs, 'activation': None}
        layers += [
            k.layers.Conv2D(filters=output_channels, kernel_size=3, padding='same', **kwargs)]
        return k.Sequential(layers, name='tail')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        >>> x = tf.zeros((1, 32, 32, 3))
        >>> model = Unet(output_channels=4, num_filters=[64, 32])
        >>> tuple(model(x).shape)
        (1, 32, 32, 4)

        >>> x = tf.zeros((1, 416, 416, 1))
        >>> model = Unet(output_channels=3, num_filters=[64, 128, 256])
        >>> tuple(model(x).shape)
        (1, 416, 416, 3)
        """
        y, intermediates = self.encoder(x)
        y = self.bottleneck(y)
        y = self.decoder(y, intermediates[::-1])
        return self.tail(y)
