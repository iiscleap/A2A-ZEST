# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import os
import json
from pathlib import Path
from typing import Optional, Union, Dict

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

import activations
from utils import init_weights, get_padding
from alias_free_activation.torch.act import Activation1d as TorchActivation1d
from utils import AttrDict

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()
        
        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()
        
        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="bigvgan",
    repo_url="https://github.com/NVIDIA/BigVGAN",
    docs_url="https://github.com/NVIDIA/BigVGAN/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2206.04658"],
):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).
    New in BigVGAN-v2: it can optionally use optimized CUDA kernels for AMP (anti-aliased multi-periodicity) blocks.

    Args:
        h (AttrDict): Hyperparameters.
        use_cuda_kernel (bool): If set to True, loads optimized CUDA kernels for AMP. This should be used for inference only, as training is not supported with CUDA kernels.

    Note:
        - The `use_cuda_kernel` parameter should be used for inference only, as training with CUDA kernels is not supported.
        - Ensure that the activation function is correctly specified in the hyperparameters (h.activation).
    """

    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(getattr(h, "model_in_dim", 128), h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x):
        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    # Additional methods for huggingface_hub support
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config.json from a Pytorch model to a local directory."""

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",  # Additional argument
        strict: bool = False,  # Additional argument
        use_cuda_kernel: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        h = load_hparams_from_json(config_file)

        # instantiate BigVGAN using h
        if use_cuda_kernel:
            print(
                f"[WARNING] You have specified use_cuda_kernel=True during BigVGAN.from_pretrained(). Only inference is supported (training is not implemented)!"
            )
            print(
                f"[WARNING] You need nvcc and ninja installed in your system that matches your PyTorch build is using to build the kernel. If not, the model will fail to initialize or generate incorrect waveform!"
            )
            print(
                f"[WARNING] For detail, see the official GitHub repository: https://github.com/NVIDIA/BigVGAN?tab=readme-ov-file#using-custom-cuda-kernel-for-synthesis"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="bigvgan_generator.pt",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        checkpoint_dict = torch.load(model_file, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model

    
class F0Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim//16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim//16, out_channels=hidden_dim//8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim//8, out_channels=hidden_dim//4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=hidden_dim//4, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.blstm = nn.GRU(hidden_dim, hidden_dim//2, num_layers=2, bidirectional=True, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, f0):
        f0 = self.relu(self.conv1(f0))
        f0 = self.relu(self.conv2(f0))
        f0 = self.relu(self.conv3(f0))
        f0 = self.relu(self.conv4(f0))
        f0 = f0.permute(0, 2, 1)
        f0, _ = self.blstm(f0)
        f0 = f0.permute(0, 2, 1)

        return f0



class CodeGenerator(BigVGAN):
    def __init__(self, h):
        super().__init__(h)
        self.dict = nn.Embedding(h.num_embeddings, h.embedding_dim)
        self.f0 = h.get('f0', None)
        self.energy = h.get('energy', None)
        self.multispkr = h.get('multispkr', None)
        self.encodeunits = h.get('encodeunits', None)
        self.encodef0 = h.get('encodef0', None)
        self.encodeenergy = h.get('encodeenergy', None)
        if self.encodeunits:
            self.unitencoder = UnitEncoder(h.embedding_dim)
        if self.encodef0:
            self.f0encoder = F0Encoder(128)
        if self.encodeenergy:
            self.energyencoder = F0Encoder(128)
        if self.multispkr:
            self.spkr = nn.Embedding(200, h.embedding_dim)

        self.encoder = None
        self.vq = None
        if h.get("lambda_commit", None):
            assert self.f0, "Requires F0 set"
            self.encoder = Encoder(**h.f0_encoder_params)
            self.vq = Bottleneck(**h.f0_vq_params)

        self.code_encoder = None
        self.code_vq = None
        if h.get('lambda_commit_code', None):
            self.code_encoder = Encoder(**h.code_encoder_params)
            self.code_vq = Bottleneck(**h.code_vq_params)
            self.dict = None

        self.quantizer = None
        if h.get('f0_quantizer_path', None):
            assert self.f0, "Requires F0 set"
            self.quantizer = Quantizer(AttrDict(h.f0_quantizer))
            quantizer_state = torch.load(h.f0_quantizer_path, map_location='cpu')
            self.quantizer.load_state_dict(quantizer_state['generator'])
            self.quantizer.eval()
            self.f0_dict = nn.Embedding(h.f0_quantizer['f0_vq_params']['l_bins'], h.embedding_dim)

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError('Padding condition signal - misalignment between condition features.')

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, **kwargs):
        code_commit_losses = None
        code_metrics = None
        if self.code_vq and kwargs['code'].dtype is torch.int64:
            x = self.code_vq.level_blocks[0].k[kwargs['code']].transpose(1, 2)
        elif self.code_vq:
            code_h = self.code_encoder(kwargs['code'])
            _, code_h_q, code_commit_losses, code_metrics = self.code_vq(code_h)
            x = code_h_q[0]
        else:
            x = self.dict(kwargs['code']).transpose(1, 2)
        if self.encodeunits:
            x = self.unitencoder(x)
        f0_commit_losses = None
        f0_metrics = None
        if self.vq:
            f0_h = self.encoder(kwargs['f0'])
            _, f0_h_q, f0_commit_losses, f0_metrics = self.vq(f0_h)
            kwargs['f0'] = f0_h_q[0]
        elif self.quantizer:
            self.quantizer.eval()
            assert not self.quantizer.training, "VQ is in training status!!!"
            f0_h = self.quantizer.encoder(kwargs['f0'])
            f0_h = [x.detach() for x in f0_h]
            zs, _, _, _ = self.quantizer.vq(f0_h)
            zs = [x.detach() for x in zs]
            f0_h_q = self.f0_dict(zs[0].detach()).transpose(1, 2)
            kwargs['f0'] = f0_h_q
        if self.encodef0:
            kwargs['f0'] = self.f0encoder(kwargs['f0'].float())
        if self.encodeenergy:
            kwargs['energy'] = self.f0encoder(kwargs['energy'].float())
        if self.f0:
            # print(x.shape, kwargs['f0'].shape)
            if x.shape[-1] < kwargs['f0'].shape[-1]:
                x = self._upsample(x, kwargs['f0'].shape[-1])
            else:
                kwargs['f0'] = self._upsample(kwargs['f0'], x.shape[-1])
            x = torch.cat([x, kwargs['f0']], dim=1)
        if self.energy:
            # print(x.shape, kwargs['f0'].shape)
            if x.shape[-1] < kwargs['energy'].shape[-1]:
                x = self._upsample(x, kwargs['energy'].shape[-1])
            else:
                kwargs['energy'] = self._upsample(kwargs['energy'], x.shape[-1])
            x = torch.cat([x, kwargs['energy']], dim=1)

        if self.multispkr:
            # spkr = self.spkr(kwargs['spkr']).transpose(1, 2)
            # print(spkr.shape)
            spkr = self._upsample(kwargs['spkr'], x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

        for k, feat in kwargs.items():
            if k in ['spkr', 'code', 'f0', 'energy']:
                continue
            # print(feat.shape, x.shape)
            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)
        if self.vq or self.code_vq:
            return super().forward(x), (code_commit_losses, f0_commit_losses), (code_metrics, f0_metrics)
        else:
            #print(x)
            return super().forward(x)
