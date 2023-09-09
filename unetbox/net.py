import torch
import timm

import torch.nn.functional as F

from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    ModuleList,
    Conv2d,
    ConvTranspose2d,
    BatchNorm2d,
    SiLU,
    Identity
)
from typing import Callable
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation


class UnetBox(Module):
    r"""

    Args:
        base_chn: number of channels of the input images.
        activation: the activation function to use. Must subclass
            ``torch.Module``.
        encoder: the encoder to use to compute the features at each level of
            the pyramid. It can be set to ``default`` or whatever string
            accepted by the timm library. If the latter is used, each level
            in the features pyramid must correspond to a whole octave
            (half octaves are not accepted by the decoder).
        depth: number of levels of the features pyramid. If ``encoder`` is
            different from ``default`` this values is ignored and a decoder
            is created to match the number of levels of the encoder.
        expansion: number of output channels from the first convolution layer.
            The number of output channels of all the following activations
            are proportional to this parameter. If ``encoder`` is different
            from ``default`` this values is ignored.
        expansion_layer: when set to ``True``, adds a 3x3 convolution that
            expands the number of activation's channels by 4x before
            down-sampling and up-sampling.
        norm_layer: when set to ``True``, adds a BatchNorm layer after every
            convolution.
        convup_layer: when set to ``True``, uses transposed convolutions in the
            up-sampling stream, instead of bilinear interpolations.
        se_block: when set to ``True``, adds a Squeeze Excitation block to
            every last layer before down-sampling and up-sampling.

    Shape:
        - Input: :math:`(N, C{_in}, H, W)`
        - Output: :math:`(N, 1, H, W)`

    """
    def __init__(
        self,
        base_chn: int = 3,
        activation: Module = SiLU,
        encoder: str = 'default',
        depth: int = 4,
        expansion: int = 16,
        expansion_layer: bool = True,
        norm_layer: bool = True,
        convup_layer: bool = True,
        se_block: bool = True
    ):
        assert (depth > 0)
        assert (expansion >= 2 and expansion % 2 == 0)
        assert (base_chn > 0)
        super().__init__()

        self.depth = depth
        self.expansion = expansion
        self.base_chn = base_chn
        self.encoder_name = encoder

        norm_layer = BatchNorm2d if norm_layer else None

        if self.encoder_name == 'default':

            # First layer in the encoder block is a 7x7 convolution
            # (no down-sampling); later we add an encoder block for
            # each level
            modules = [
                Conv2dNormActivation(
                    base_chn,
                    expansion,
                    kernel_size=7,
                    padding=3,
                    norm_layer=norm_layer,
                    activation_layer=activation
                )
            ]

            ceb = partial(
                UnetBox.create_encoder_block,
                activation=activation,
                expansion_layer=expansion_layer,
                norm_layer=norm_layer
            )

            # Wrap encoder layer in a Squeeze Excitation block
            if se_block:
                ceb = partial(
                    UnetBox.create_se_block,
                    create_block_func=ceb,
                    se_ratio=4,
                    activation=activation
                )

            # Input and output channels for each level in the pyramid
            encoder_chns = [expansion * (2 ** i) for i in range(depth + 1)]

            # Append each level in the pyramid to a ModuleList with
            # the determined number of input/output channels
            for in_chn, out_chn in zip(encoder_chns[:-1], encoder_chns[1:]):
                modules.append(ceb(in_chn, out_chn))

            self.encoder = ModuleList(modules)

        else:
            # Use an encoder available from the timm library
            self.encoder = timm.create_model(encoder, features_only=True)

            # Number of output channels for each layer of the encoder,
            # which is equivalent to number of input channels be used
            # by the decoder
            encoder_chns = self.encoder.feature_info.channels()
            encoder_chns.insert(0, self.base_chn)

        cdb = partial(
            UnetBox.create_decoder_block,
            activation=activation,
            expansion_layer=expansion_layer,
            norm_layer=norm_layer,
            convup_layer=convup_layer
        )

        # Wrap decoder layer in a Squeeze Excitation block
        if se_block:
            cdb = partial(
                UnetBox.create_se_block,
                create_block_func=cdb,
                se_ratio=4,
                activation=activation
            )

        # Invert number of channels in the encoder for convenience
        encoder_chns = encoder_chns[::-1]

        # Bottom-level layer has just 1 input (no skip connection)
        encoder_chns[0] //= 2

        in_out_chns = zip(encoder_chns[:-1], encoder_chns[1:])

        # Add encoders to a ModuleList for each level of the pyramid
        modules = [
            cdb(2 * in_chn, out_chn) for in_chn, out_chn in in_out_chns
        ]

        # Top-level layer has no BatchNorm or Squeeze Excitation
        modules.append(
            Conv2d(2 * encoder_chns[-1], 1, kernel_size=3, padding=1)
        )

        self.decoder = ModuleList(modules)

    def forward(self, x: Tensor) -> Tensor:
        
        x_encoded = [x]

        # Encoder
        if self.encoder_name == 'default':
            for encoder in self.encoder:
                x_encoded.insert(0, encoder(x_encoded[0]))
        else:
            for xe in self.encoder(x_encoded[0]):
                x_encoded.insert(0, xe)

        # Decoder
        x = []
        for xe, decoder in zip(x_encoded, self.decoder):
            channels = torch.cat([xe] + x, dim=1)
            x = [decoder(channels)]

        return x[0]

    @staticmethod
    def create_encoder_block(
            in_chn: int,
            out_chn: int,
            activation: Module,
            expansion_layer: bool,
            norm_layer: Module
    ) -> Sequential:

        mid_chn = 2 * out_chn if expansion_layer else out_chn

        layers = [
            Conv2dNormActivation(
                in_chn,
                mid_chn,
                kernel_size=3,
                padding=1,
                norm_layer=norm_layer,
                activation_layer=activation
            ),
            Conv2dNormActivation(
                mid_chn,
                out_chn,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=norm_layer,
                activation_layer=activation
            )
        ]

        return Sequential(*layers)

    @staticmethod
    def create_decoder_block(
        in_chn: int,
        out_chn: int,
        activation: Module,
        expansion_layer: bool,
        norm_layer: Module,
        convup_layer: bool
    ) -> Sequential:

        class UpSample(Module):
            def forward(self, input):
                return F.interpolate(input, scale_factor=2, mode='bilinear')

        mid_chn = 2 * in_chn if expansion_layer else in_chn
        bias = norm_layer is None
        batch_norm = norm_layer or Identity

        if convup_layer:
            upsample = ConvTranspose2d(
                mid_chn,
                out_chn,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=bias
            )
        else:
            upsample = Sequential(
                Conv2d(mid_chn, out_chn, kernel_size=3, padding=1, bias=bias),
                UpSample()
            )

        layers = [
            Conv2dNormActivation(
                in_chn,
                mid_chn,
                kernel_size=3,
                padding=1,
                norm_layer=norm_layer,
                activation_layer=activation
            ),
            upsample,
            batch_norm(out_chn),
            activation(inplace=True)
        ]

        return Sequential(*layers)

    @staticmethod
    def create_se_block(
        in_chn: int,
        out_chn: int,
        create_block_func: Callable[[int, int], Sequential],
        se_ratio: int,
        activation: Module
    ) -> Sequential:

        block = create_block_func(in_chn, out_chn)
        block.append(
            SqueezeExcitation(out_chn, max(1, out_chn // se_ratio), activation)
        )

        return block
