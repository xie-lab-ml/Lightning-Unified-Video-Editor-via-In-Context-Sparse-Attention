"""Video I/O utilities."""

import subprocess
import numpy as np
import torch


def save_video_ffmpeg(tensor, save_path, fps=16, normalize=True, value_range=(-1, 1)):
    """
    Save a video tensor [C, T, H, W] to an mp4 file via ffmpeg.
    Values are mapped from value_range to [0, 255] uint8.
    """
    if normalize:
        lo, hi = value_range
        tensor = (tensor - lo) / (hi - lo)
    tensor = tensor.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    # [C, T, H, W] -> [T, H, W, C]
    frames = tensor.transpose(1, 2, 3, 0).astype(np.uint8)
    T, H, W, C = frames.shape

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}', '-pix_fmt', 'rgb24' if C == 3 else 'gray',
        '-r', str(fps), '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', save_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(T):
        frame = frames[i] if C == 3 else frames[i, ..., 0]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def read_video_ffmpeg(path, target_fps=16):
    """Read a video file into a tensor [C, T, H, W] (float, 0-1)."""
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,nb_frames', '-of', 'csv=p=0', path],
        capture_output=True, text=True
    )
    W, H, num_frames = map(int, probe.stdout.strip().split(',')[:3])

    cmd = [
        'ffmpeg', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-r', str(target_fps), '-'
    ]
    out = subprocess.run(cmd, capture_output=True)
    frames = np.frombuffer(out.stdout, dtype=np.uint8).reshape(-1, H, W, 3)
    frames = frames.transpose(3, 0, 1, 2)  # [C, T, H, W]
    return torch.from_numpy(frames).float() / 255.0
