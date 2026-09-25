from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.augmentations import APPROACHES, make_contrastive_views
from src.dataloader import SmilesDataset
from src.loss import nt_xent_loss
from src.model import PolyCLModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "HAYDERphd/polyBERT"
CSV_PATH = r"data\train.csv"
SMILES_COL = "repeat_unit_smiles"

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-5
TEMPERATURE = 0.05
MASK_PROB = 0.10
MAX_LENGTH = 128


def pretrain(
    approach="original_mask",
    epochs=EPOCHS,
    max_steps=None,
    output_path=None,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = PolyCLModel(MODEL_NAME).to(DEVICE)

    dataset = SmilesDataset(CSV_PATH, SMILES_COL)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    history = []
    steps_done = 0

    for epoch in range(epochs):
        total_loss = 0.0
        epoch_steps = 0

        for smiles_batch in loader:
            (view1_ids, view1_mask), (view2_ids, view2_mask) = (
                make_contrastive_views(
                    smiles_batch=smiles_batch,
                    tokenizer=tokenizer,
                    approach=approach,
                    max_length=MAX_LENGTH,
                    mask_probability=MASK_PROB,
                    device=DEVICE,
                )
            )

            _, z1 = model(view1_ids, view1_mask)
            _, z2 = model(view2_ids, view2_mask)

            loss = nt_xent_loss(
                z1,
                z2,
                temperature=TEMPERATURE,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            history.append(loss.item())
            total_loss += loss.item()
            epoch_steps += 1
            steps_done += 1

            if max_steps is not None and steps_done >= max_steps:
                break

        if epoch_steps:
            print(
                f"{approach} | epoch {epoch + 1}/{epochs} "
                f"| loss {total_loss / epoch_steps:.4f}"
            )

        if max_steps is not None and steps_done >= max_steps:
            break

    if output_path is None:
        output_path = ROOT / "checkpoints" / f"polycl_{approach}.pt"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "approach",
        choices=APPROACHES,
        nargs="?",
        default="original_mask",
    )
    args = parser.parse_args()
    pretrain(args.approach)


if __name__ == "__main__":
    main()
