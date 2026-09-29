
```
我写好了程序，也已经对这首歌跑了一遍，生成了伴奏文件：

- output/周慧敏+-+最爱_伴奏.mp3：去掉人声后的伴奏
- output/周慧敏+-+最爱_人声.mp3：分离出来的人声

我只用音量做了粗略检查：原曲约 -13.7 dB，人声和伴奏各约 -17 dB，说明歌曲确实被拆成了两部分。实际效果我还没法试听，请你听一下伴奏里有没有残留人声。

原理：程序是 remove_vocals.py，用的是 Meta 开源的 Demucs 模型（htdemucs）。它把歌曲拆成人声、鼓、贝斯和其他乐器四轨，再把除人声外的三轨合成伴奏。这比传统的"左右声道相减"效果好很多，不会把居中的贝斯、鼓一起消掉。

环境：你的电脑是 Intel Mac，PyTorch 在这类机器上最高只支持 2.2.2 版和 Python 3.12，现在默认的 Python 3.13 装不上。所以我用 conda 新建了一个叫 vocal-remove 的环境（Python 3.11 + torch 2.2.2 + demucs 4.0.1）。第一次运行时会自动下载模型文件（约 80MB）。

用法：
conda activate vocal-remove
python remove_vocals.py "周慧敏+-+最爱.mp3"                 # 只输出伴奏
python remove_vocals.py song.mp3 --keep-vocals              # 同时输出人声
python remove_vocals.py song.mp3 -m htdemucs_ft --shifts 5  # 质量更好，速度更慢
python remove_vocals.py song.mp3 -o my_dir --bitrate 192k   # 指定输出目录和码率

速度：用 CPU 处理这首歌花了大约 6 分钟。如果伴奏里还能听到一点人声，可以换 -m htdemucs_ft 模型，或者把 --shifts 调大，但会慢好几倍。

```
