# 远程文件同步指南

Triton 内核来自远程训练服务器，后续若远程代码更新，可通过以下方式同步到本地。

## 远程服务器信息

| 项目 | 值 |
|------|-----|
| 地址 | `sshao213@hpc3login.hpc.hkust-gz.edu.cn` |
| 密码 | `1620111262shI!` |
| 远程路径 | `/data/user/sshao213/zk_workspace/wanx-v2v/model_dit/models/wanx/modules/` |

## 拉取文件

需要安装 `sshpass`（已安装）：

```bash
# 安装 sshpass（如未安装）
apt-get install -y sshpass

# 拉取文件到本地
sshpass -p '1620111262shI!' scp -o StrictHostKeyChecking=no \
  'sshao213@hpc3login.hpc.hkust-gz.edu.cn:/data/user/sshao213/zk_workspace/wanx-v2v/model_dit/models/wanx/modules/video_sparse_attn_triton_3.py' \
  ./sparse_attn_tilelang/triton_kernels_remote.py

# 如需原样保留（不改名）
sshpass -p '1620111262shI!' scp -o StrictHostKeyChecking=no \
  'sshao213@hpc3login.hpc.hkust-gz.edu.cn:/data/user/sshao213/zk_workspace/wanx-v2v/model_dit/models/wanx/modules/video_sparse_attn_triton_3.py' \
  ./sparse_attn_tilelang/
```

## 拉取后处理

拉取后需拆分文件为 `triton_kernels.py` + `triton_host.py`：

```bash
cd sparse_attn_tilelang
# 内核部分（前 599 行）
head -599 video_sparse_attn_triton_3.py > triton_kernels.py
# 入口部分（第 600 行起）
tail -n +600 video_sparse_attn_triton_3.py > triton_host_raw.py
# 然后手动清理 triton_host_raw.py，只保留 sparse_piecewise_attn_1st_intervals 函数
```

## 环境对齐

远程服务器 triton 版本为 **3.5.1**，本地需保持一致：

```bash
pip install triton==3.5.1
```

## 当前本地文件对应关系

| 远程文件 | 本地文件 |
|---------|---------|
| `video_sparse_attn_triton_3.py` (L1-599) | `triton_kernels.py` |
| `video_sparse_attn_triton_3.py` (L600-601) | `triton_host.py` |
