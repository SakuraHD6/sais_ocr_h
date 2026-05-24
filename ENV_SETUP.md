# Ubuntu 工作站环境配置说明

本文档用于配置本项目的本地/工作站训练环境。当前假设：

- 系统：Ubuntu 22.04/24.04
- 环境管理：不使用 conda，使用 Python `venv`
- 显卡：NVIDIA GPU
- 本机 CUDA：12.8
- 任务：训练 YOLO 检测器、训练字符识别器、离线评估、替换提交镜像中的权重

注意：本机 CUDA 12.8 不要求 PyTorch wheel 也必须完全等于 12.8。PyTorch pip 包自带 CUDA runtime，只要 NVIDIA 驱动足够新，`cu124/cu126/cu128` 通常都可以运行。为了本机训练，优先使用 `cu128`；为了尽量贴近比赛提交镜像，可使用 `cu124`。

## 1. 检查显卡和驱动

```bash
nvidia-smi
```

需要确认：

- 能看到 GPU 型号和显存。
- `CUDA Version` 显示为 `12.8` 或更高。
- 如果 `nvidia-smi` 不存在，需要先安装或修复 NVIDIA 驱动。

## 2. 安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev \
  build-essential \
  git \
  wget \
  curl \
  unzip \
  tmux \
  htop \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6
```

其中 `libgl1` 和 `libglib2.0-0` 用于解决 OpenCV 常见报错：

```text
ImportError: libGL.so.1: cannot open shared object file
```

## 3. 创建 Python 虚拟环境

建议把训练脚本放在：

```bash
mkdir -p ~/sais_ocr
cd ~/sais_ocr
```

创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

升级基础工具：

```bash
python -m pip install -U pip setuptools wheel
```

后续每次重新登录服务器，都需要先执行：

```bash
cd ~/sais_ocr
source .venv/bin/activate
```

## 4. 安装 PyTorch

### 推荐方案：CUDA 12.8 工作站训练

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

只要 `cuda available: True`，就可以继续。

### 兼容方案：贴近比赛提交镜像

当前提交镜像使用的是 CUDA 12.4 版 PyTorch：

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

这个方案更贴近 Dockerfile 中的运行环境，但不一定是本机训练速度最优方案。

## 5. 安装项目依赖

如果当前目录有 `requirements.txt`：

```bash
pip install -r requirements.txt
```

如果只是配置训练脚本环境，可以直接安装：

```bash
pip install \
  ultralytics==8.4.51 \
  "opencv-python-headless>=4.8.0" \
  "pillow>=10.0.0" \
  "pyyaml>=6.0" \
  "tqdm>=4.64.0" \
  "numpy>=1.24.0" \
  "timm>=0.9.0" \
  "scikit-learn>=1.0.0"
```

验证关键库：

```bash
python - <<'PY'
import torch
import cv2
import ultralytics
import timm
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("cv2:", cv2.__version__)
print("ultralytics:", ultralytics.__version__)
print("timm:", timm.__version__)
PY
```

## 6. 建议目录结构

推荐在工作站上整理为：

```text
~/sais_ocr/
  .venv/
  prepare_yolo_dataset.py
  train_yolo.py
  evaluate_detection.py
  watch_and_evaluate.py
  prepare_classifier_dataset.py
  train_classifier.py
  evaluate_pipeline_offline.py

/root/data/
  out_of_domain/
  yolo_data/
  classifier_crops/

/root/yolo_dataset/
  detect_yolo11l_1536/

/root/classifier_output/
  best.pth
```

如果不是 root 用户，把 `/root/...` 换成自己的路径，例如：

```text
/home/用户名/data/out_of_domain
/home/用户名/yolo_dataset
/home/用户名/classifier_output
```

## 7. 准备检测器数据

原始训练数据路径示例：

```text
/root/data/out_of_domain
```

生成 YOLO 数据：

```bash
python prepare_yolo_dataset.py \
  --src /root/data/out_of_domain \
  --out /root/data/yolo_data \
  --val-ratio 0.2 \
  --seed 42
```

检查结果：

```bash
ls /root/data/yolo_data
ls /root/data/yolo_data/images/train | head
ls /root/data/yolo_data/labels/train | head
cat /root/data/yolo_data/data.yaml
```

正常情况下应看到：

```text
conversion_warnings.txt
data.yaml
images/
labels/
```

## 8. 训练 YOLO 检测器

```bash
python train_yolo.py \
  --data /root/data/yolo_data/data.yaml \
  --model yolo11l.pt \
  --imgsz 1536 \
  --epochs 150 \
  --batch 8 \
  --device 0 \
  --project /root/yolo_dataset \
  --name detect_yolo11l_1536
```

训练结果通常在：

```text
/root/yolo_dataset/detect_yolo11l_1536/weights/best.pt
/root/yolo_dataset/detect_yolo11l_1536/results.csv
```

如果显存不足，优先调整：

```bash
--batch 4
```

如果还不够，再考虑：

```bash
--imgsz 1280
```

## 9. 检测器离线评估

```bash
python evaluate_detection.py \
  --weights /root/yolo_dataset/detect_yolo11l_1536/weights/best.pt \
  --img-dir /root/data/yolo_data/images/val \
  --xml-dir /root/data/out_of_domain \
  --conf 0.20 \
  --imgsz 1536 \
  --device 0 \
  --save-json /root/yolo_dataset/eval_logs/detect_eval_conf020.json
```

当前实验中，检测器在验证集上大致表现为：

```text
conf=0.20
precision=0.5813
recall=0.5601
f1=0.5705
```

因此提交镜像中建议使用：

```text
YOLO_IMGSZ=1536
CONFIDENCE_THRESHOLD=0.20
```

## 10. 准备识别器数据

检测器已经能较好地产生候选框，当前线上分数低的主要原因是字符识别器不准。因此下一步应训练识别器。

生成字符裁剪数据：

```bash
python prepare_classifier_dataset.py \
  --src /root/data/out_of_domain \
  --out /root/data/classifier_crops \
  --val-ratio 0.1 \
  --padding 0.15 \
  --min-samples-per-class 1 \
  --seed 42
```

检查：

```bash
ls /root/data/classifier_crops
ls /root/data/classifier_crops/train | head
cat /root/data/classifier_crops/class_mapping.json | head
```

对于训练集中出现的非 Unicode 脏数据，直接过滤/剔除，不需要修复。

## 11. 训练识别器

```bash
python train_classifier.py \
  --data /root/data/classifier_crops \
  --output /root/classifier_output \
  --backbone efficientnet_b0 \
  --img-size 128 \
  --epochs 50 \
  --batch-size 256 \
  --device cuda:0
```

如果显存不足：

```bash
--batch-size 128
```

如果仍然不足：

```bash
--batch-size 64
```

训练输出：

```text
/root/classifier_output/best.pth
```

这个文件后续需要替换提交仓库里的：

```text
classifier_output/best.pth
```

## 12. 完整离线评估

识别器训练完成后，建议在验证集上跑完整 pipeline：

```bash
python evaluate_pipeline_offline.py \
  --repo /root/sais_ocr_h \
  --img-dir /root/data/yolo_data/images/val \
  --xml-dir /root/data/out_of_domain \
  --det-weights /root/yolo_dataset/detect_yolo11l_1536/weights/best.pt \
  --cls-weights /root/classifier_output/best.pth \
  --conf 0.20 \
  --imgsz 1536 \
  --device cuda:0 \
  --save-json /root/yolo_dataset/eval_logs/pipeline_eval_conf020.json
```

重点看：

```text
precision
recall
f1
tp
fp
fn
```

如果检测 F1 不低，但完整 pipeline F1 仍然极低，说明识别器仍是主要瓶颈。

## 13. 替换提交权重

检测器权重：

```bash
cp /root/yolo_dataset/detect_yolo11l_1536/weights/best.pt \
  /root/sais_ocr_h/yolo_dataset/detect_yolo11l/weights/best.pt
```

识别器权重：

```bash
cp /root/classifier_output/best.pth \
  /root/sais_ocr_h/classifier_output/best.pth
```

确认 Dockerfile 环境变量：

```text
YOLO_IMGSZ=1536
CONFIDENCE_THRESHOLD=0.20
MAX_DET=100
NMS_IOU_THRESHOLD=0.45
MERGED_CLASS_POLICY=common
```

## 14. 本地构建提交镜像

在提交仓库根目录：

```bash
docker build -t sais_ocr_h:v1 .
```

简单运行检查：

```bash
docker run --rm --gpus all \
  -v /root/data/test_images:/saisdata/13/eval/images:ro \
  -v /root/sais_result:/saisresult \
  sais_ocr_h:v1
```

检查输出：

```bash
ls /root/sais_result
cat /root/sais_result/prediction.json | head
```

比赛需要的输出文件是：

```text
/saisresult/prediction.json
```

## 15. 常见问题

### `torch.cuda.is_available()` 是 `False`

先检查：

```bash
nvidia-smi
```

如果 `nvidia-smi` 正常，重新安装 PyTorch CUDA 版：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### `libGL.so.1` 缺失

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

### `libpng warning: iCCP`

这是 PNG 色彩配置警告，通常不影响训练和评估，可以忽略。

### cuDNN plan warning

如果看到类似：

```text
cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED
```

通常不影响训练。如果日志太多，可以尝试：

```bash
export TORCH_CUDNN_V8_API_DISABLED=1
```

然后重新运行训练命令。

### 训练中断后继续

Ultralytics 通常支持 resume：

```bash
yolo detect train resume model=/root/yolo_dataset/detect_yolo11l_1536/weights/last.pt
```

或者根据 `train_yolo.py` 的参数支持情况添加 `--resume`。

## 16. 参考链接

- PyTorch 官方安装页：https://pytorch.org/get-started/locally/
- PyTorch 历史版本安装命令：https://pytorch.org/get-started/previous-versions/
- Ultralytics 文档：https://docs.ultralytics.com/
