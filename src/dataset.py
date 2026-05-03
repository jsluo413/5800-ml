from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 128
CLASS_NAMES = ("damage", "no_damage")


def build_transforms(train):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def find_split_dirs(root):
    wanted = ["train_another", "validation_another", "test", "test_another"]
    out = {}
    for name in wanted:
        matches = list(root.rglob(name))
        matches = [m for m in matches if m.is_dir()
                   and any((m / c).is_dir() for c in CLASS_NAMES)]
        if not matches:
            raise FileNotFoundError(f"could not find split '{name}' under {root}")
        out[name] = matches[0]
    return out


def make_loaders(root, batch_size=64, num_workers=4):
    splits = find_split_dirs(root)
    train_ds = datasets.ImageFolder(splits["train_another"], transform=build_transforms(True))
    val_ds = datasets.ImageFolder(splits["validation_another"], transform=build_transforms(False))
    test_bal_ds = datasets.ImageFolder(splits["test"], transform=build_transforms(False))
    test_unbal_ds = datasets.ImageFolder(splits["test_another"], transform=build_transforms(False))

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_bal_ds, shuffle=False, **common),
        DataLoader(test_unbal_ds, shuffle=False, **common),
    )
