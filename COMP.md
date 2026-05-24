# 古文字识别赛道：少样本跨域古文字识别

更新时间：2026-05-23

官方页面：https://competition.ai4s.com.cn/race/11/introduction

## 1. 赛题概况

赛道名称：古文字识别赛道：少样本跨域古文字识别  
英文名称：Ancient Script Recognition Track: Few-shot Cross-domain Ancient Script Recognition  
比赛类型：算法竞赛  
当前状态：进行中  
奖金池：CNY 200,000

初赛阶段：

- 开始时间：2026-04-01 10:00:00，UTC+8
- 结束时间：2026-05-27 14:00:00，UTC+8
- 初赛采用 Docker 镜像提交。
- 每支队伍每天有 3 次提交机会，提交失败也会消耗次数。
- 在线推理和评测时间限制为 4 小时。
- 排行榜展示本阶段历史最好成绩。
- 初赛结束后，排行榜前 150 支队伍需要提交代码接受审核。
- 通过代码审核的前 100 支队伍晋级复赛。

## 2. 任务目标

给定完整的古文字拓片图片，模型需要自动完成两件事：

1. 检测图片中所有古文字的位置，输出字符检测框。
2. 识别每个检测框内的字符内容，输出对应 Unicode 字符文本。

最终结果需要写入：

```text
/saisresult/prediction.json
```

每个预测项格式：

```json
{
  "bbox": [843, 2087, 93, 89],
  "text": "天"
}
```

注意：

- `bbox` 必须是 `[x, y, w, h]`。
- `x, y` 是左上角坐标。
- `w, h` 是宽和高。
- 不要输出 `[x1, y1, x2, y2]`。
- `text` 必须是字符串，并且需要和标准答案完全一致。

## 3. 官方数据

官方数据集接口显示初赛数据包：

| 文件 | 大小 |
| --- | ---: |
| `train.tar` | 39,269,209,064 bytes，约 39.27 GB |

官方数据说明如下：

| 数据集 | 用途 | 数量 | 格式 | 来源 |
| --- | --- | ---: | --- | --- |
| 甲骨文预训练数据 | 识别模型预训练 | 约 77,000 | PNG + JSON | HUST-OBC |
| 跨域训练数据 | 端到端检测和识别训练 | 6,000 | PNG + XML | 比赛平台 |
| PDF 文献数据 | 额外训练增强资源 | 约 500 | PDF | ModelScope |
| 评测数据 | 最终在线评测 | 约 1,000 | PNG | 比赛平台 |

### 3.1 甲骨文预训练数据

官方来源：HUST-OBC。

典型结构：

```text
images/        # 单字裁剪图片
annotations/   # JSON 标注，包含 char、char_unicode 等字段
```

用途：

- 训练或预训练单字分类模型。
- 学习古文字字形特征。
- 可以作为识别模型的基础数据。

### 3.2 跨域训练数据

官方来源：复旦大学出土文献与古文字研究中心。

典型结构：

```text
train/out_of_domain/
├── *.png     # 完整拓片图片
└── *.xml     # 标注文件
```

XML 关键字段：

| 字段 | 含义 |
| --- | --- |
| `page/id` | 图片文件名 |
| `page/width`, `page/height` | 图片尺寸 |
| `char/text` | 字符内容 |
| `char/position` | 字符位置，可能是矩形或多边形 |

坐标转换：

- 矩形 `"x1,y1,x2,y2"` 转为 `[x1, y1, x2 - x1, y2 - y1]`
- 多边形 `"x1,y1;x2,y2;..."` 转为 `[min_x, min_y, max_x - min_x, max_y - min_y]`

### 3.3 PDF 文献数据

官方来源：ModelScope ancient_char_doc 数据集。

用途：

- 可作为额外训练资源。
- 可从 PDF 中提取古文字图片，用于数据增强。

### 3.4 评测数据

比赛平台会自动把评测图片挂载到容器内：

```text
/saisdata/13/eval/images/
├── image_id_1.png
├── image_id_2.png
└── ...
```

说明：

- 图片格式为 PNG。
- 图片数量约 1000 张。
- 图片文件名去掉 `.png` 后缀后就是 `prediction.json` 中的 `image_id`。
- 评测数据不包含标注文件。
- 不要向 `/saisdata` 写入任何结果。

当前项目默认读取：

```text
/saisdata/13/eval/images
```

如果该目录不存在，代码会递归搜索 `/saisdata` 下的图片。

## 4. 评分规则

最终分数为整体 F1。

匹配规则：

1. 预测框和真实框的 IoU 必须大于等于 0.5，才可能匹配成功。
2. 检测框匹配成功后，预测 `text` 必须和真实 `text` 完全一致，才记为 TP。
3. 检测位置错误、重复检测、误检或文字识别错误都会计入 FP。
4. 漏检或未被正确识别的真实字符计入 FN。

指标公式：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

这个评分方式意味着：

- 只有检测框准、字符也识别对，才算真正得分。
- 框准但文字错，不得分。
- 文字对但框不准，也不得分。
- 重复框会增加 FP。

## 5. 初赛代码提交规范

初赛提交 Docker 镜像。平台启动镜像后，镜像需要自动完成推理并写出结果。

输入目录：

```text
/saisdata/13/eval/images/
```

输出文件：

```text
/saisresult/prediction.json
```

输出 JSON 要求：

- UTF-8 编码。
- 顶层为字典。
- 每个图片 ID 对应一个列表。
- 每个预测项包含 `bbox` 和 `text`。
- 如果某张图没有检测结果，建议保留该图片 ID，并写为空列表 `[]`。

示例：

```json
{
  "ZHJWD060612-000001-GUJINGJUYING195": [
    {"bbox": [843, 2087, 93, 89], "text": "天"},
    {"bbox": [808, 2147, 97, 96], "text": "王"}
  ],
  "ZHXXXXXXXX-XXXXXX-GUJIXXXXUYINGXXXX": []
}
```

官方评测环境：

| 项目 | 配置 |
| --- | --- |
| GPU | Tesla V100-SXM2-16GB |
| CUDA | 12.4 |
| PyTorch | 2.2.0+ |

当前项目镜像使用 CUDA 12.4 runtime，并安装 PyTorch CUDA 12.4 版本。

## 6. 当前项目方案

当前项目采用两阶段 OCR 流程：

```text
完整拓片图片
  -> YOLO 检测字符框
  -> 裁剪字符区域
  -> 分类模型识别字符
  -> 写出 prediction.json
```

当前镜像提交地址：

```text
crpi-cnoiz6qm6ajuv6bh.cn-shenzhen.personal.cr.aliyuncs.com/deng_h/saic_ocr_h:v1
```

当前 GitHub 自动构建行为：

- 推送到 `main` 后自动构建 Docker 镜像。
- 登录阿里云 ACR。
- 推送镜像 tag：`v1`。

当前镜像内重要路径：

| 路径 | 含义 |
| --- | --- |
| `/app/src/` | 推理代码 |
| `/app/yolo_dataset/weights/best.pt` | YOLO 检测权重 |
| `/app/classifier_output/best.pth` | 字符分类权重 |
| `/app/char_mapping.json` | 字符映射文件 |
| `/saisdata/13/eval/images` | 默认输入图片目录 |
| `/saisresult/prediction.json` | 比赛要求输出文件 |

当前推理入口：

```text
run.sh
src/run_inference.py
```

## 7. 官方问答和注意事项

### 7.1 古文字显示乱码是正常现象

部分古文字在本地编辑器中可能显示为乱码、方框或无法识别的符号。这通常是字体缺失导致的显示问题。

处理原则：

- 代码中按 Unicode 字符串或 Unicode code point 处理。
- 不要根据编辑器显示效果判断字符是否有效。

### 7.2 `prediction.json` 必须是 UTF-8

官方说明：UTF-8 可以表示比赛涉及的 Unicode 字符，不存在“某些字符只能用 UTF-16 表示”的问题。

因此：

- 输出文件必须使用 UTF-8。
- 写 JSON 时使用 `encoding="utf-8"`。
- 建议 `json.dump(..., ensure_ascii=False)`。

### 7.3 训练集中的脏数据

官方公告指出，训练集 XML 中有少量非 Unicode 脏数据。

例如：

- 以 `ZH-` 开头的异常字符串。
- `None` 等占位值。
- 无法解析成有效 Unicode code point 的标签。

建议：

- 训练数据预处理时直接过滤这些脏标签。
- 官方说明最终测试集不存在这类非 Unicode 脏数据。

## 8. 当前分数分析

最近一次线上结果：

```text
TP: 12
FP: 17644
FN: 11151
Precision: 0.000680
Recall: 0.001075
F1: 0.000833
GT chars: 11163
Pred chars: 17656
```

结论：

- 仅靠调输出格式和后处理，收益很小。
- 目前 TP 极低，说明检测命中和识别正确同时成立的样本太少。
- 下一步需要拆分评估检测错误和识别错误。

建议优先顺序：

1. 先评估检测器：只看 IoU >= 0.5，不看文字。
2. 如果检测 recall 很低，优先重训 YOLO。
3. 如果检测能命中但文字错误多，再重训分类器。
4. 分类器训练应使用原始拓片图中的 GT 框裁剪，而不是只用干净单字图。

## 9. 云端训练建议

第一优先级：重训或微调 YOLO 检测器。

推荐起步命令：

```bash
yolo detect train \
  model=yolo11l.pt \
  data=data.yaml \
  imgsz=1536 \
  epochs=150 \
  batch=8 \
  device=0 \
  optimizer=AdamW \
  cos_lr=True \
  patience=30 \
  workers=8 \
  project=yolo_dataset \
  name=detect_yolo11l_1536
```

如果显存足够，可以尝试：

```text
model=yolo11x.pt
imgsz=1536 或 1920
batch=4
```

训练完成后替换：

```text
yolo_dataset/detect_yolo11l/weights/best.pt
```

然后提交：

```bash
git add yolo_dataset/detect_yolo11l/weights/best.pt
git commit -m "Update detector weights"
git push origin main
```

GitHub 自动构建流程会重新构建并推送 ACR 镜像。

第二优先级：重训分类器。

建议分类训练数据来源：

```text
原始拓片图片 + GT bbox 裁剪 + 与推理一致的 padding/resize
```

分类标签建议直接使用最终 `text`，而不是过度依赖目录 ID。因为评分只看 `text` 是否完全一致。
