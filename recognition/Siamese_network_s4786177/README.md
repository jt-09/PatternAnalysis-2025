# Siamese Network for ISIC 2020 — s4786177

Student: Jay Thakkar  
Student number: s4786177
## Abstract

The goal was to build a binary classifier (normal vs melanoma) for the ISIC 2020 dermoscopic image dataset using a Siamese-style approach: a shared ResNet50 backbone produces embeddings that are passed to a small MLP classification head. The classifier was trained with cross-entropy on image-level labels and evaluated on a held-out test set. This repository contains the dataset handling, training, evaluation and plotting utilities used to reproduce experiments and generate the key figures.

## Files of interest

- `dataset.py`: data loading, preprocessing, augmentation, stratified train/val/test split and balanced sampler.
- `modules.py`: model components: `FeatureExtractor` (ResNet50 + MLP projection head), `ClassifierHead` (linear classification head), and `SiameseNet` wrapper.
- `train.py`: training loop (cross-entropy), metrics logging, save-best checkpoint by validation AUC, and training plots.
- `predict.py`: load best checkpoint, evaluate on test set, compute AUC/accuracy/Youden threshold, and save ROC/confusion/t-SNE plots.

## How it works 

The Siamese-style network uses a shared feature extractor (ResNet50) to produce low-dimensional embeddings for input images. The embeddings are passed through a small MLP head to produce 2-class logits (normal / melanoma). During training the model is optimized with standard cross-entropy loss. Training, evaluation and inference scripts (`train.py`, `predict.py`) produce plots (loss, accuracy, AUC, confusion matrices, and t-SNE) saved under the `reports/` folder.



###  Model Architecture Details: The Siamese Network Design
The network is structured as a Siamese architecture, utilizing a single, shared set of weights across two conceptually parallel branches (for metric learning). The primary goal is to simultaneously optimize for image classification and a highly structured, discriminative feature space.

The model, wrapped by SiameseNet, comprises a sequential arrangement of two modules: the Feature Extractor and the Classifier Head.


insert figure: high-level architecture diagram (two images → shared ResNet50 → embedding → MLP classifier) (yet to do)

> insert image of architecture diagram here (yet to do)

 #### Feature Extractor
 This module is responsible for projecting the raw input image into a compact, low-dimensional vector space.Backbone (ResNet50): A ResNet50 network, pretrained on ImageNet, serves as the foundational backbone. Transfer learning is employed by loading the ImageNet weights to leverage learned hierarchical visual features. The standard 1,000-class classification layer is replaced with nn.Identity().Output: The output from the global average pooling layer yields a 2048-dimensional feature vector.
 #### Projection Head (MLP): 
 This head transforms the backbone's features into the final, lower-dimensional embedding. It is a three-layer Multi-Layer Perceptron (MLP):$$\mathbf{2048} \xrightarrow{\text{Linear}} 512 \xrightarrow{\text{Linear}} 256 \xrightarrow{\text{Linear}} \mathbf{\text{emb\_dim}}$$Structure: The sequence includes ReLU activation and Dropout p=0.6 layers following the first two linear transformations, serving as non-linearities and regularization.Initialization: The linear layers within the MLP head are initialized using Kaiming normal initialization (He initialization), which is appropriate for layers followed by a ReLU non-linearity.Embedding: The final output is a 128-dim vector (default emb_dim. This vector undergoes L2-normalization (torch.nn.functional.normalize) before being outputted, which is a prerequisite for effective metric-learning losses that rely on angular or cosine distance in a normalized space.

## Dataset and preprocessing
The initial ISIC 2020: Skin Cancer Detection challenge provided a large collection of high-resolution dermoscopic images. However, a pre-processed version was adopted to facilitate faster training and resource efficiency. This version contains image files resized to a fixed resolution of 224x224 along with the associated metadata in `train-metadata.csv`. The important fields in the metadata are the unique image identifier (`isic_id`), a randomized `patient_id`, and the `target` column, which provides the binary class label: benign (0) or malignant (1). This dataset was then further divided in `dataset.py` using stratified sampling to create the 80/10/10 train, validation, and test splits. Given the class imbalance, the training set employs a weighted sampler to ensure equal representation of both classes in each batch, mitigating the issue.

### Oversampling minority class
Oversampling was used during training to deal with the strong class imbalance in the melanoma dataset, where there are many more benign (non-cancerous) images than melanoma (cancerous) ones. Without balancing, the model would mostly see benign examples and could easily learn to always predict "benign," which would look accurate but fail to detect real melanomas. To fix this, the data loader in dataset.py uses a `WeightedRandomSampler`, which increases the chances of selecting melanoma images so that each training batch contains a more even mix of both classes. This helps the model learn what makes melanomas different instead of being biased toward the majority class. Oversampling is especially important here because the model also uses metric learning (Triplet Loss), which compares examples from both classes. Balanced batches ensure that the model always has enough positive and negative examples to learn useful differences between them. 
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
```
Insert image: dataset samples (example benign / malignant images)

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
