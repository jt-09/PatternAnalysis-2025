# Siamese Network for ISIC 2020 — s4786177

Student: Jay Thakkar  
Student number: s4786177
## Abstract

The goal was to build a binary classifier (normal vs melanoma) for the ISIC 2020 dermoscopic image dataset using a Siamese-style approach: a shared ResNet50 backbone produces embeddings that are passed to a small MLP classification head. The classifier was trained with cross-entropy on image-level labels and evaluated on a held-out test set. This repository contains the dataset handling, training, evaluation and plotting utilities used to reproduce experiments and generate the key figures.
## How it works (brief)

The Siamese-style network uses a shared feature extractor (ResNet50) to produce low-dimensional embeddings for input images. The embeddings are passed through a small MLP head to produce 2-class logits (normal / melanoma). During training the model is optimized with standard cross-entropy loss. Training, evaluation and inference scripts (`train.py`, `predict.py`) produce plots (loss, accuracy, AUC, confusion matrices, and t-SNE) saved under the `reports/` folder.
insert figure: high-level architecture diagram (two images → shared ResNet50 → embedding → MLP classifier) (yet to do)

> insert image of architecture diagram here (yet to do)
## Files of interest

- `dataset.py`: data loading, preprocessing, augmentation, stratified train/val/test split and balanced sampler.
- `modules.py`: model components: `FeatureExtractor` (ResNet50 + MLP projection head), `ClassifierHead` (linear classification head), and `SiameseNet` wrapper.
- `train.py`: training loop (cross-entropy), metrics logging, save-best checkpoint by validation AUC, and training plots.
- `predict.py`: load best checkpoint, evaluate on test set, compute AUC/accuracy/Youden threshold, and save ROC/confusion/t-SNE plots.
## Algorithm and design choices

The primary goal was to leverage a pretrained ResNet50 backbone so the model could learn strong visual features from dermoscopic images with limited training time. The siamese-style design (shared backbone producing embeddings) was chosen because it naturally extends to metric-learning approaches (triplet loss or contrastive loss) if needed later. The small MLP projection head (2048 → 512 → 256 → emb_dim) was chosen to reduce the backbone output to a compact embedding while keeping sufficient capacity for classification.
In `modules.py` it can be seen that the feature extractor uses a pretrained ResNet50 whose final fc was replaced by an identity and followed by a sequential projection head. The MLP head uses ReLU non-linearities and dropout; linear layers were initialized with Kaiming normal initialization. The classifier head is a single linear layer mapping the embedding to 2 logits.

## Dataset and preprocessing
The initial ISIC 2020: Skin Cancer Detection challenge provided a large collection of high-resolution dermoscopic images. However, a pre-processed version was adopted to facilitate faster training and resource efficiency. This version contains image files resized to a fixed resolution of $224 \times 224$ pixels, along with the associated metadata in `train-metadata.csv`. The important fields in the metadata are the unique image identifier (`isic_id`), a randomized `patient_id`, and the `target` column, which provides the binary class label: benign (0) or malignant (1). This dataset was then further divided in `dataset.py` using stratified sampling to create the $\mathbf{80/10/10}$ train, validation, and test splits. Given the class imbalance, the training set employs a weighted sampler to ensure equal representation of both classes in each batch, mitigating the issue.
### Image Augmentations

The Siamese Network was trained using various image augmentation techniques, defined within the `get_transforms` function in `dataset.py`. This was important for improving the model's resilience and preventing overfitting which is a valid concern given the high class imbalance in the dataset. The main training transform, `train_t`, applies sequential augmentation steps: It uses a RandomResizedCrop to $224 \times 224$ pixels, which applies a scale variation (between $0.85$ and $1.0$) and positional shifts. This is followed by RandomHorizontalFlip and RandomVerticalFlip (both with a $p=0.5$ probability), and RandomRotation (up to $15^\circ$). These spatial transformations ensure the model learns to recognise the correct class independent of the lesion's orientation. Additionally, ColorJitter is applied with small variations in brightness ($0.10$), contrast ($0.10$), saturation ($0.05$), and hue ($0.02$). The final steps convert the image to a PyTorch Tensor and apply standard ImageNet normalization.

```python
# from dataset.py -> get_transforms
train_t = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15, fill=0),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
    transforms.ToTensor(),   # convert PIL image to torch.FloatTensor [0,1]
    norm                     # normalize channels to mean/std
])

Insert image: dataset samples (example benign / malignant images)

## Model architecture details

- Backbone: ResNet50 (pretrained on ImageNet, weights used when `pretrained=True` in `FeatureExtractor`). The last fc layer was removed (set to Identity) and the 2048-dim feature vector is passed to the projection head.
- Projection head: Linear(2048,512) → ReLU → Dropout → Linear(512,256) → ReLU → Dropout → Linear(256,emb_dim)
- Embedding dimension: configurable (default 128). L2 normalization of embeddings was enabled for metric-learning experiments.
- Classifier: single linear layer from embedding to 2 logits. Cross-entropy loss used for training.

Insert figure: projection head diagram and embedding visualization placeholder

## Training, evaluation and saved outputs

Running `train.py` will train the model and save the best checkpoint (by validation AUC) to `siamese_ce.pt` (or the `save_path` you pass). The script also writes per-epoch plots to `reports/` (loss, accuracy, AUC). Example outputs generated by the repository include:

- `reports/ce_loss.png` — Cross-entropy loss curves (train vs val)
- `reports/ce_acc.png` — Accuracy curves
- `reports/ce_auc.png` — AUC curves
- `reports/test_roc.png` — ROC plot from `predict.py` evaluation
- `reports/test_confusion_matrix_argmax.png` — Confusion matrix at argmax
- `reports/test_confusion_matrix_threshold.png` — Confusion matrix at Youden threshold
- `reports/test_tsne.png` — t-SNE of test embeddings (optional)

Example usage (from project folder):
yet to do

## Dependencies (recommended)


- Python >= 3.8
- torch >= 1.11 (CUDA-enabled build if using GPU)
- torchvision >= 0.12
- pandas >= 1.3
- scikit-learn >= 1.0
- pillow >= 8.0
- matplotlib >= 3.4
- tqdm >= 4.60

The reproducible environment can be recreated using:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install pip==23.0.1
pip install torch torchvision pandas scikit-learn pillow matplotlib tqdm
```

Set the random seed (the repo already sets seeds) by calling `set_seed(s)` — scripts call `set_seed(42)` by default to improve reproducibility. Note that GPU/cuDNN non-determinism can still cause minor run-to-run variation; to increase determinism further you can disable `torch.backends.cudnn.benchmark` and set `torch.backends.cudnn.deterministic = True` (may reduce performance).


## Results and expected performance

The project was designed to target around 0.8 test accuracy (when trained end-to-end with the provided augmentations and balanced sampling). Final results depend on training hyperparameters, number of epochs, GPU availability and dataset version.

Insert image: example training plots (loss/acc/auc)

r to `train.py`/`predict.py` so the commands above can be run without modifying the scripts. Tell me which you'd like next.
# Siamese Network - s4786177

Student: Jay Thakkar
Student number: s4786177
