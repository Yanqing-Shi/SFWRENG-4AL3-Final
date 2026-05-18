# Aircraft AOV Image Classification

This project uses PyTorch to train a simple CNN for binary classification on aircraft images. The main workflow is active learning: train on the existing labeled images, select uncertain unlabeled thumbnails, label those images with a small Tkinter GUI, then add the newly labeled samples back into the training set.

## Project Structure

```text
.
|-- train.py              # Main active learning training script
|-- test.py               # Evaluates the saved model
|-- model.py              # CNN model definition
|-- dataset.py            # Dataset and DataLoader helpers
|-- aov.py                # Tkinter labeling tool
|-- gettopindex.py        # Downloads thumbnails and metadata from JetPhotos
|-- model.pth             # Saved model weights
|-- train/
|   |-- class_0/          # Labeled training images for class 0
|   `-- class_1/          # Labeled training images for class 1
`-- thumbnails/
    |-- catalog.csv       # Thumbnail metadata and labels
    `-- *.jpg             # Unlabeled or partially labeled thumbnails
```

## Requirements

Python 3.10+ is recommended.

For training and testing:

```bash
pip install torch torchvision pandas numpy matplotlib pillow
```

For downloading new images with `gettopindex.py`, also install:

```bash
pip install selenium requests beautifulsoup4
```

`gettopindex.py` uses Selenium with Microsoft Edge WebDriver. You do not need to run it if the thumbnail images already exist.

## Dataset Format

The labeled dataset uses `torchvision.datasets.ImageFolder`, so the training folder must follow this structure:

```text
train/
|-- class_0/
|   `-- image1.jpg
`-- class_1/
    `-- image2.jpg
```

Unlabeled images are stored in `thumbnails/`. The file `thumbnails/catalog.csv` is expected to contain at least these columns:

- `id`: image id, used to locate `<id>.jpg`
- `aov`: label value

Label meanings:

- `aov = 0`: class 0
- `aov = 1`: class 1
- empty value: not labeled yet

## Training

Run the active learning training script:

```bash
python train.py
```

The training script does the following:

1. Loads labeled images from `train/class_0` and `train/class_1`.
2. Initializes the CNN model.
3. Trains the model for several epochs.
4. Computes uncertainty scores for images in `thumbnails/`.
5. Selects the 10 most uncertain images.
6. Opens the labeling window so the selected images can be labeled.
7. Adds the newly labeled samples to the training set.
8. Saves the final weights to `model.pth`.

Note: `train.py` currently contains local absolute paths:

```python
data_dir_L = r'C:\Users\todds\Desktop\Final\train'
data_dir_U = r'C:\Users\todds\Desktop\Final\thumbnails'
```

If the project is moved, update these paths or replace them with relative paths:

```python
data_dir_L = "./train"
data_dir_U = "./thumbnails"
```

## Testing

Run:

```bash
python test.py
```

`test.py` loads `model.pth` and evaluates it on the test split created by `get_loaders()`. It prints:

- overall accuracy
- class 0 accuracy
- class 1 accuracy
- sample counts and correct counts for each class

Note: `test.py` also contains an absolute path:

```python
data_dir_L = '/mnt/c/Users/todds/Desktop/env/Final/train'
```

For Windows PowerShell, change it to:

```python
data_dir_L = r'C:\Users\todds\Desktop\Final\train'
```

Or use a relative path:

```python
data_dir_L = "./train"
```

## Labeling Controls

The labeling GUI in `aov.py` supports these keyboard controls:

- `Right`: next image
- `Left`: previous image
- `0`: label as class 0
- `1`: label as class 1

When the window is closed, it asks whether to save changes. If saved, the labels are written back to `thumbnails/catalog.csv`.

## Model

The CNN is defined in `model.py`.

Input images are resized to `120 x 80` and normalized. The model contains:

- two convolution layers
- two max pooling layers
- one fully connected hidden layer
- dropout
- a final output layer with 2 logits

## Notes

- `dataset.py` sets random seeds for more reproducible results.
- `get_loaders()` currently uses about 20% of each class as the training split and the remaining images as the test split.
- `thumbnails/catalog.csv` currently includes repeated `Unnamed` index columns, likely created by saving the CSV multiple times. These columns can be cleaned later.
- Before uploading to GitHub, avoid committing cache folders such as `__pycache__/`. Large model weights such as `*.pth` and archives such as `*.zip` are already ignored by `.gitignore`.
