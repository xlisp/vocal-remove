#!/usr/bin/env python3
"""训练 / 微调音源分离模型。

数据集目录结构（MUSDB18-HQ 即是此格式，每首歌一个目录，44.1kHz wav）:
    data/
      train/<歌名>/{drums,bass,other,vocals}.wav
      test/<歌名>/{drums,bass,other,vocals}.wav

用法:
    # 从零训练自己的 MiniDemucs
    python train.py --data data --out runs/mini
    # 在自己的数据上微调预训练 htdemucs
    python train.py --data data --arch htdemucs --lr 1e-5 --batch-size 2 --out runs/ft
    # 断点续训
    python train.py --data data --out runs/mini --resume runs/mini/last.pt
    # 用训练好的模型去人声
    python remove_vocals.py song.mp3 --checkpoint runs/mini/best.pt
"""
import argparse
import math
import random
import time
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from model import build_model, save_checkpoint


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------
def find_tracks(root: Path, sources) -> list:
    tracks = sorted(p for p in root.iterdir()
                    if p.is_dir() and all((p / f"{s}.wav").exists() for s in sources))
    if not tracks:
        raise SystemExit(f"{root} 下没有找到包含 {sources} 全部 wav 的歌曲目录")
    return tracks


def read_chunk(path: Path, offset: int, length: int, channels: int) -> torch.Tensor:
    """读 wav 的 [offset, offset+length) 片段，返回 (channels, length)，不足补零。"""
    data, _ = sf.read(path, start=offset, frames=length, dtype="float32", always_2d=True)
    x = torch.from_numpy(data.T.copy())
    if x.shape[0] == 1 and channels == 2:
        x = x.expand(2, -1).clone()
    elif x.shape[0] != channels:
        x = x.mean(0, keepdim=True).expand(channels, -1).clone()
    if x.shape[-1] < length:
        x = torch.nn.functional.pad(x, (0, length - x.shape[-1]))
    return x


class StemDataset(Dataset):
    """训练集：每次随机抽片段，并做“重混”数据增强。

    因为混合信号 = 各音轨之和（线性叠加），可以把 A 歌的鼓 + B 歌的贝斯
    + C 歌的人声……随意组合，生成无穷多“新歌”，而标签天然正确。
    """

    def __init__(self, root: Path, sources, samplerate: int, channels: int,
                 segment: float, epoch_size: int, remix: bool = True,
                 augment: bool = True):
        self.sources = list(sources)
        self.channels = channels
        self.length = int(round(segment * samplerate))
        self.epoch_size = epoch_size
        self.remix = remix
        self.augment = augment
        self.tracks = find_tracks(root, self.sources)
        self.frames = {}
        for t in self.tracks:
            info = sf.info(t / f"{self.sources[0]}.wav")
            if info.samplerate != samplerate:
                raise SystemExit(f"{t} 采样率 {info.samplerate} != {samplerate}，"
                                 "请先用 ffmpeg -ar 重采样")
            self.frames[t] = info.frames

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx):
        # 用 random 而非 numpy：DataLoader 会为每个 worker 单独设置 random 的种子
        track = random.choice(self.tracks)
        stems = []
        for s in self.sources:
            t = random.choice(self.tracks) if self.remix else track
            offset = random.randint(0, max(0, self.frames[t] - self.length))
            x = read_chunk(t / f"{s}.wav", offset, self.length, self.channels)
            if self.augment:
                x = x * random.uniform(0.25, 1.25)          # 随机音量
                if random.random() < 0.5:
                    x = -x                                   # 极性翻转
                if self.channels == 2 and random.random() < 0.5:
                    x = x.flip(0)                            # 左右声道互换
            stems.append(x)
        stems = torch.stack(stems)                           # (S, C, T)
        return stems.sum(0), stems                           # 混合, 目标


class ValidDataset(Dataset):
    """验证集：每首歌按固定步长切片，不做增强，结果可复现。"""

    def __init__(self, root: Path, sources, samplerate: int, channels: int,
                 segment: float, max_chunks_per_track: int = 4):
        self.sources = list(sources)
        self.channels = channels
        self.length = int(round(segment * samplerate))
        self.items = []
        for t in find_tracks(root, self.sources):
            frames = sf.info(t / f"{self.sources[0]}.wav").frames
            n = max(1, min(max_chunks_per_track, frames // self.length))
            step = max(self.length, (frames - self.length) // n)
            self.items += [(t, i * step) for i in range(n)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        t, offset = self.items[idx]
        stems = torch.stack([read_chunk(t / f"{s}.wav", offset, self.length, self.channels)
                             for s in self.sources])
        return stems.sum(0), stems


# ----------------------------------------------------------------------------
# 损失与指标
# ----------------------------------------------------------------------------
def multi_resolution_stft_loss(est: torch.Tensor, ref: torch.Tensor,
                               fft_sizes=(512, 1024, 2048)) -> torch.Tensor:
    """在多个时频分辨率下比较幅度谱：谱收敛 + 对数幅度 L1。"""
    est = est.reshape(-1, est.shape[-1])
    ref = ref.reshape(-1, ref.shape[-1])
    loss = 0.0
    for n in fft_sizes:
        win = torch.hann_window(n, device=est.device)
        E = torch.stft(est, n, n // 4, window=win, return_complex=True).abs()
        R = torch.stft(ref, n, n // 4, window=win, return_complex=True).abs()
        sc = torch.linalg.norm(R - E) / (torch.linalg.norm(R) + 1e-7)
        mag = (torch.log(E + 1e-7) - torch.log(R + 1e-7)).abs().mean()
        loss = loss + sc + mag
    return loss / len(fft_sizes)


def compute_loss(est, ref, stft_weight: float):
    est, ref = est.float(), ref.float()   # 混合精度下也用 fp32 算损失
    loss = (est - ref).abs().mean()
    if stft_weight > 0:
        loss = loss + stft_weight * multi_resolution_stft_loss(est, ref)
    return loss


def sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """信号失真比 (dB)，返回 (B, S)。越高越好。"""
    num = ref.pow(2).sum((-2, -1))
    den = (ref - est).pow(2).sum((-2, -1))
    return 10 * torch.log10((num + 1e-8) / (den + 1e-8))


# ----------------------------------------------------------------------------
# 训练循环
# ----------------------------------------------------------------------------
def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device, stft_weight):
    model.eval()
    total, scores, n = 0.0, [], 0
    for mix, stems in loader:
        mix, stems = mix.to(device), stems.to(device)
        est = model(mix)
        total += compute_loss(est, stems, stft_weight).item() * mix.shape[0]
        scores.append(sdr(est.float(), stems).cpu())
        n += mix.shape[0]
    return total / n, torch.cat(scores).mean(0)


def main() -> None:
    p = argparse.ArgumentParser(description="训练音源分离模型")
    p.add_argument("--data", type=Path, required=True, help="含 train/ 与 test/ 的数据目录")
    p.add_argument("--out", type=Path, default=Path("runs/mini"))
    p.add_argument("--arch", choices=["mini", "htdemucs"], default="mini")
    p.add_argument("--channels", type=int, default=48, help="mini: 第一层通道数")
    p.add_argument("--depth", type=int, default=5, help="mini: 编码器层数")
    p.add_argument("--segment", type=float, default=6.0,
                   help="训练片段秒数（htdemucs 固定用其自带的 7.8s）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--epoch-size", type=int, default=2000, help="每轮抽多少个片段")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500, help="学习率预热步数")
    p.add_argument("--stft-weight", type=float, default=0.5, help="多分辨率 STFT 损失权重")
    p.add_argument("--clip", type=float, default=5.0, help="梯度裁剪")
    p.add_argument("--no-remix", action="store_true", help="关闭跨歌曲重混增强")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto / cuda / cpu / mps")
    p.add_argument("--amp", action="store_true", help="CUDA 混合精度")
    p.add_argument("--resume", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.arch == "mini":
        model = build_model("mini", channels=args.channels, depth=args.depth,
                            segment=args.segment)
        segment = args.segment
    else:
        model = build_model("htdemucs")
        segment = float(model.segment)   # 其 Transformer 位置编码按此长度训练
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型 {args.arch}: {n_params:.1f}M 参数, 音轨 {model.sources}, device={device}")

    common = dict(sources=model.sources, samplerate=model.samplerate,
                  channels=model.audio_channels, segment=segment)
    train_set = StemDataset(args.data / "train", epoch_size=args.epoch_size,
                            remix=not args.no_remix, **common)
    valid_set = ValidDataset(args.data / "test", **common)
    print(f"训练歌曲 {len(train_set.tracks)} 首, 验证片段 {len(valid_set)} 个")

    loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                        drop_last=True, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)

    def lr_lambda(step):  # 线性预热 + 余弦退火
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt["best"]
        print(f"从第 {start_epoch} 轮继续, 当前最佳验证损失 {best:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for step, (mix, stems) in enumerate(loader, 1):
            mix, stems = mix.to(device), stems.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                est = model(mix)
            loss = compute_loss(est, stems, args.stft_weight)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += loss.item()
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} "
                      f"loss {running / args.log_every:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
                running = 0.0

        val_loss, val_sdr = evaluate(model, valid_loader, device, args.stft_weight)
        sdr_str = "  ".join(f"{s}={v:.2f}dB" for s, v in zip(model.sources, val_sdr.tolist()))
        print(f"== epoch {epoch} 验证损失 {val_loss:.4f}  SDR: {sdr_str}")

        is_best = val_loss < best
        best = min(best, val_loss)
        state = dict(epoch=epoch, best=best, optimizer=opt.state_dict(),
                     scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                     val_sdr=dict(zip(model.sources, val_sdr.tolist())))
        save_checkpoint(args.out / "last.pt", model, args.arch, **state)
        if is_best:
            save_checkpoint(args.out / "best.pt", model, args.arch, **state)
            print(f"   ↳ 新的最佳模型已保存: {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
