# Hurricane Damage Detection

Binary classification of post-Hurricane-Harvey satellite tiles (damage vs no_damage).
Compares a from-scratch CNN against three ImageNet-pretrained backbones (ResNet-18,
EfficientNet-B0, MobileNetV3-Small) and includes Grad-CAM visualisations.

5800 ML final project, Spring 2026.

## Dataset

[Kaggle: kmader/satellite-images-of-hurricane-damage](https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage)

| split                | damage | no_damage |
|----------------------|-------:|----------:|
| `train_another`      | 5000   | 5000      |
| `validation_another` | 1000   | 1000      |
| `test`               | 1000   | 1000      |
| `test_another`       | 8000   | 1000      |

128x128 RGB tiles. The two test splits are reported on separately so the
imbalance behaviour is visible.

## Layout

```
main.py              run the whole pipeline
regen_gradcam.py     just rebuild the Grad-CAM figure
src/
  dataset.py         ImageFolder loaders, transforms
  models.py          SimpleCNN + 3 pretrained backbones
  train.py           training loop with checkpointing
  evaluate.py        metrics + plots
  gradcam.py         Grad-CAM overlays
```

## Run

```powershell
conda create -n hurricane_ml python=3.10 -y
conda activate hurricane_ml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install scikit-learn matplotlib seaborn pandas tqdm grad-cam kagglehub timm python-pptx

# data
$env:KAGGLEHUB_CACHE = "$PWD\datafolder\kagglehub_cache"
python -c "import kagglehub; kagglehub.dataset_download('kmader/satellite-images-of-hurricane-damage')"

# train + evaluate + grad-cam
python main.py
```

