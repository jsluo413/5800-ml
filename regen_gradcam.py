from pathlib import Path

import torch

from src.dataset import make_loaders
from src.gradcam import make_gradcam_grid
from src.models import build_model
from main import pick_gradcam_examples, DATA_ROOT, CKPTS, GRADCAM_DIR


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_bal, _ = make_loaders(DATA_ROOT, batch_size=128, num_workers=0)
    idx_to_class = {v: k for k, v in test_bal.dataset.class_to_idx.items()}
    pos_label = test_bal.dataset.class_to_idx["damage"]

    name = "efficientnet_b0"
    model, target_layer = build_model(name)
    model.load_state_dict(torch.load(CKPTS / f"{name}_best.pt", map_location=device))
    model.to(device)

    examples = pick_gradcam_examples(model, test_bal, device,
                                     pos_label=pos_label, n_per_group=2)
    print("examples:", len(examples), "tags:", [t for _, _, t in examples])

    make_gradcam_grid(model, target_layer, device, examples,
                      [idx_to_class[i] for i in range(2)],
                      GRADCAM_DIR / f"gradcam_best_{name}_wide.png",
                      title=f"Grad-CAM | {name} (best)",
                      cols=4, cell_size=3.0)
    print("saved")


if __name__ == "__main__":
    main()
