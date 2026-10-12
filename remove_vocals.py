#!/usr/bin/env python3
"""去除歌曲人声，保留伴奏（基于 Demucs AI 音源分离）。

用法:
    python remove_vocals.py "周慧敏+-+最爱.mp3"
    python remove_vocals.py song.mp3 -o out_dir --model htdemucs_ft --keep-vocals
    python remove_vocals.py song.mp3 --checkpoint runs/mini/best.pt   # 用自己训练的模型

输出:
    <out_dir>/<歌名>_伴奏.mp3   （去人声的伴奏）
    <out_dir>/<歌名>_人声.mp3   （可选，--keep-vocals）
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from model import load_model, separate_long


def load_audio(path: Path, sr: int, channels: int) -> torch.Tensor:
    """用 ffmpeg 解码任意音频为 float32，返回 (channels, samples)。"""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).T
    return torch.from_numpy(audio.copy())


def save_mp3(wav: torch.Tensor, sr: int, path: Path, bitrate: str) -> None:
    """先写临时 wav，再用 ffmpeg 转成 mp3。"""
    wav = wav.clamp(-1, 1).cpu().numpy().T
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, wav, sr, subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", tmp.name,
             "-codec:a", "libmp3lame", "-b:a", bitrate, str(path)],
            check=True,
        )


def separate(input_path: Path, out_dir: Path, model_name: str,
             keep_vocals: bool, bitrate: str, shifts: int,
             checkpoint: Path = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint:
        print(f"加载自训练模型 {checkpoint} (device={device}) ...")
        model = load_model(checkpoint)
    else:
        print(f"加载模型 {model_name} (device={device}) ...")
        model = get_model(model_name)
        model.eval()

    print(f"读取音频 {input_path.name} ...")
    wav = load_audio(input_path, model.samplerate, model.audio_channels)

    # 标准化，分离后再还原音量
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    wav = (wav - mean) / std

    print("分离中（CPU 上可能需要几分钟）...")
    if checkpoint:
        sources = separate_long(model, wav, device=device, overlap=0.25)
    else:
        with torch.no_grad():
            sources = apply_model(model, wav[None], device=device,
                                  shifts=shifts, split=True, overlap=0.25,
                                  progress=True)[0]
    sources = sources * std + mean

    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx]
    # 伴奏 = 除人声外所有音轨之和（drums + bass + other）
    accompaniment = sources.sum(0) - vocals

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    acc_path = out_dir / f"{stem}_伴奏.mp3"
    save_mp3(accompaniment, model.samplerate, acc_path, bitrate)
    print(f"✅ 伴奏已保存: {acc_path}")

    if keep_vocals:
        voc_path = out_dir / f"{stem}_人声.mp3"
        save_mp3(vocals, model.samplerate, voc_path, bitrate)
        print(f"✅ 人声已保存: {voc_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="去除歌曲人声，保留伴奏")
    p.add_argument("input", type=Path, help="输入音频文件（mp3/wav/flac...）")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("output"))
    p.add_argument("-m", "--model", default="htdemucs",
                   help="htdemucs(默认) / htdemucs_ft(更精细但慢4倍) / mdx_extra")
    p.add_argument("--keep-vocals", action="store_true", help="同时导出人声轨")
    p.add_argument("--bitrate", default="320k")
    p.add_argument("--shifts", type=int, default=1,
                   help="随机平移次数，越大质量略好但更慢")
    p.add_argument("-c", "--checkpoint", type=Path,
                   help="train.py 训练出的 .pt 检查点（指定后忽略 --model/--shifts）")
    args = p.parse_args()

    separate(args.input, args.out_dir, args.model,
             args.keep_vocals, args.bitrate, args.shifts, args.checkpoint)


if __name__ == "__main__":
    main()
