# Aircraft AOV Image Classification

这个项目使用 PyTorch 训练一个简单的 CNN，对飞机图片进行二分类。当前代码主要围绕 active learning 流程：先用已有标注图片训练模型，再从未标注缩略图中挑选模型最不确定的图片，打开标注窗口让用户标注，最后把新标注样本加入训练集继续训练。

## Project Structure

```text
.
├── train.py              # active learning 主训练流程
├── test.py               # 加载 model.pth 并在测试集上评估
├── model.py              # CNN 模型结构
├── dataset.py            # 数据集和 DataLoader 工具
├── aov.py                # Tkinter 图片标注窗口
├── gettopindex.py        # 从 JetPhotos 抓取缩略图和 catalog 数据
├── model.pth             # 已保存的模型权重
├── train/
│   ├── class_0/          # 已标注训练图片：类别 0
│   └── class_1/          # 已标注训练图片：类别 1
└── thumbnails/
    ├── catalog.csv       # 未标注/半标注图片信息，包含 aov 标签列
    └── *.jpg             # 待标注缩略图
```

## Requirements

建议使用 Python 3.10+。需要的主要依赖：

```bash
pip install torch torchvision pandas numpy matplotlib pillow selenium requests beautifulsoup4
```

如果只训练和测试模型，通常需要：

```bash
pip install torch torchvision pandas numpy matplotlib pillow
```

`gettopindex.py` 使用 Selenium 和 Microsoft Edge WebDriver。如果不需要重新抓取图片，可以不用运行它。

## Dataset Format

训练集使用 `torchvision.datasets.ImageFolder`，所以目录必须保持这种格式：

```text
train/
├── class_0/
│   └── image1.jpg
└── class_1/
    └── image2.jpg
```

未标注图片放在 `thumbnails/` 目录下，同时需要 `thumbnails/catalog.csv`。代码会读取其中的 `id` 和 `aov` 列：

- `aov = 0` 表示类别 0
- `aov = 1` 表示类别 1
- 空值表示还没有标注

## How to Train

运行 active learning 训练：

```bash
python train.py
```

训练流程大致是：

1. 从 `train/class_0` 和 `train/class_1` 加载已有标注图片。
2. 初始化 `CNN` 模型。
3. 训练若干 epoch。
4. 对 `thumbnails/` 中的图片计算不确定性。
5. 选择最不确定的 10 张图片。
6. 打开 Tkinter 标注窗口，通过键盘输入 `0` 或 `1` 标注。
7. 把新标注样本加入训练集，继续下一轮。
8. 训练结束后保存为 `model.pth`。

注意：`train.py` 中目前写了本机绝对路径：

```python
data_dir_L = r'C:\Users\todds\Desktop\Final\train'
data_dir_U = r'C:\Users\todds\Desktop\Final\thumbnails'
```

如果项目移动到其他位置，需要改成新的路径，或者改成相对路径：

```python
data_dir_L = "./train"
data_dir_U = "./thumbnails"
```

## How to Test

运行：

```bash
python test.py
```

`test.py` 会加载 `model.pth`，然后在 `get_loaders()` 划分出的测试集上输出：

- overall accuracy
- class 0 accuracy
- class 1 accuracy
- 每个类别的样本数量和正确数量

注意：`test.py` 里也有一个绝对路径：

```python
data_dir_L = '/mnt/c/Users/todds/Desktop/env/Final/train'
```

在 Windows PowerShell 中运行时，建议改成：

```python
data_dir_L = r'C:\Users\todds\Desktop\Final\train'
```

或者：

```python
data_dir_L = "./train"
```

## Labeling Controls

`aov.py` 中的标注窗口支持键盘操作：

- `Right`: 下一张图片
- `Left`: 上一张图片
- `0`: 标为类别 0
- `1`: 标为类别 1

关闭窗口时会询问是否保存更改。如果选择保存，会写回 `thumbnails/catalog.csv`。

## Model

模型定义在 `model.py`：

- 输入：RGB 图片，经过 resize 后为 `120 x 80`
- 两层卷积层
- 两层最大池化
- 一层全连接隐藏层
- Dropout
- 输出 2 个类别 logits

## Notes

- `dataset.py` 中固定了随机种子，方便复现实验结果。
- 当前 `get_loaders()` 每个类别只取约 20% 作为训练集，剩余作为测试集。
- `thumbnails/catalog.csv` 当前包含多列重复的 `Unnamed` 索引列，可能是多次保存 CSV 时产生的；如果后续整理数据，可以清理这些列。
- 上传 GitHub 时，建议不要上传 `__pycache__/`。如果数据集或模型文件太大，也可以把 `train/`、`thumbnails/`、`*.pth` 加入 `.gitignore`。
