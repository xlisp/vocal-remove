"""音源分离模型定义 + 检查点读写 + 分段推理。

两种可训练结构:
    mini      —— 自己从零实现的时域 U-Net（Demucs v2 风格的简化版），代码全在本文件
    htdemucs  —— Meta 预训练的 Hybrid Transformer Demucs，用于在自己的数据上微调

所有模型都满足同一个接口:
    model(mix)  mix: (B, C, T)  ->  (B, S, C, T)
    model.sources / model.samplerate / model.audio_channels / model.segment
"""
from pathlib import Path

import torch
from torch import nn

DEFAULT_SOURCES = ("drums", "bass", "other", "vocals")


class BLSTM(nn.Module):
    """瓶颈处的双向 LSTM：让每个时间步“看到”整段片段的上下文。"""

    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, num_layers=layers, bidirectional=True)
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (T, B, C) -> LSTM -> (B, C, T)，残差连接
        y = x.permute(2, 0, 1)
        y, _ = self.lstm(y)
        y = self.linear(y).permute(1, 2, 0)
        return x + y


class MiniDemucs(nn.Module):
    """时域 U-Net 音源分离器。

    编码器每层把时间分辨率压缩 stride 倍、通道数扩大 growth 倍；
    解码器对称地还原，并通过跳跃连接拿回编码器同层的细节。
    kernel=8, stride=4, padding=2 时，只要输入长度是 stride**depth 的整数倍，
    每层的长度都正好是整除/整乘关系，不需要任何裁剪。
    """

    def __init__(self, sources=DEFAULT_SOURCES, audio_channels: int = 2,
                 channels: int = 48, growth: float = 2.0, depth: int = 5,
                 kernel: int = 8, stride: int = 4, lstm_layers: int = 2,
                 samplerate: int = 44100, segment: float = 6.0):
        super().__init__()
        self._kwargs = dict(sources=list(sources), audio_channels=audio_channels,
                            channels=channels, growth=growth, depth=depth,
                            kernel=kernel, stride=stride, lstm_layers=lstm_layers,
                            samplerate=samplerate, segment=segment)
        self.sources = list(sources)
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        self.segment = segment
        self.depth = depth
        self.stride = stride
        pad = (kernel - stride) // 2

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        in_ch, ch = audio_channels, channels
        for i in range(depth):
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_ch, ch, kernel, stride, padding=pad),
                nn.ReLU(),
                nn.Conv1d(ch, 2 * ch, 1),
                nn.GLU(dim=1),
            ))
            out_ch = len(self.sources) * audio_channels if i == 0 else in_ch
            dec = [
                nn.Conv1d(ch, 2 * ch, 3, padding=1),
                nn.GLU(dim=1),
                nn.ConvTranspose1d(ch, out_ch, kernel, stride, padding=pad),
            ]
            if i > 0:  # 最外层输出是波形，不能加 ReLU
                dec.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*dec))
            in_ch, ch = ch, int(ch * growth)
        self.lstm = BLSTM(in_ch, lstm_layers)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        B, C, T = mix.shape
        # 按混合信号自身的均值/方差归一化，输出时再还原 —— 模型对音量不敏感
        mono = mix.mean(1, keepdim=True)
        mean = mono.mean(-1, keepdim=True)
        std = mono.std(-1, keepdim=True) + 1e-5
        x = (mix - mean) / std

        unit = self.stride ** self.depth
        padded = (T + unit - 1) // unit * unit
        x = nn.functional.pad(x, (0, padded - T))

        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
        x = self.lstm(x)
        for dec in self.decoder:
            x = dec(x + skips.pop())

        x = x[..., :T].view(B, len(self.sources), C, T)
        return x * std[:, None] + mean[:, None]


def build_model(arch: str = "mini", **kwargs) -> nn.Module:
    if arch == "mini":
        return MiniDemucs(**kwargs)
    if arch == "htdemucs":
        from demucs.pretrained import get_model
        # get_model 返回 BagOfModels，htdemucs 里只有一个模型
        model = get_model(kwargs.get("base", "htdemucs")).models[0]
        model._kwargs = {"base": kwargs.get("base", "htdemucs")}
        return model
    raise ValueError(f"未知结构: {arch}")


def save_checkpoint(path: Path, model: nn.Module, arch: str, **extra) -> None:
    torch.save({"arch": arch, "kwargs": model._kwargs,
                "state": model.state_dict(), **extra}, path)


def load_model(path: Path, map_location="cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ckpt["arch"], **ckpt["kwargs"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


@torch.no_grad()
def separate_long(model: nn.Module, wav: torch.Tensor, device="cpu",
                  overlap: float = 0.25, progress: bool = True) -> torch.Tensor:
    """对整首歌分段推理，重叠部分用三角窗加权平均（overlap-add）。

    wav: (C, T)  ->  (S, C, T)
    """
    model.to(device).eval()
    C, T = wav.shape
    length = int(float(model.segment) * model.samplerate)
    hop = max(1, int(length * (1 - overlap)))
    window = torch.cat([torch.arange(1, length // 2 + 1),
                        torch.arange(length - length // 2, 0, -1)]).float()
    out = torch.zeros(len(model.sources), C, T)
    weight = torch.zeros(T)

    starts = list(range(0, max(T - length, 0) + hop, hop))
    for n, start in enumerate(starts, 1):
        chunk = wav[:, start:start + length]
        size = chunk.shape[-1]
        chunk = nn.functional.pad(chunk, (0, length - size))
        est = model(chunk[None].to(device))[0, ..., :size].float().cpu()
        w = window[:size]
        out[..., start:start + size] += est * w
        weight[start:start + size] += w
        if progress:
            print(f"\r  分段 {n}/{len(starts)}", end="", flush=True)
    if progress:
        print()
    return out / weight.clamp_min(1e-8)
