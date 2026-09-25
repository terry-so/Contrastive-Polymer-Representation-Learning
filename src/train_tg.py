from pathlib import Path
import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from src.dataloader import TgDataset
from src.downstream import TgRegressor
from src.pretrain import CSV_PATH, DEVICE, MAX_LENGTH, MODEL_NAME


BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
TEST_FRACTION = 0.20
SEED = 0


def _tokenize(smiles_batch, tokenizer):
    tokens = tokenizer(
        list(smiles_batch),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return (
        tokens["input_ids"].to(DEVICE),
        tokens["attention_mask"].to(DEVICE),
    )


def _metrics(y_true, y_pred):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)

    error = y_pred - y_true
    mae = error.abs().mean().item()
    rmse = error.square().mean().sqrt().item()

    ss_res = error.square().sum()
    ss_tot = (y_true - y_true.mean()).square().sum()
    r2 = (1.0 - ss_res / ss_tot).item()

    return {
        "mae_k": mae,
        "rmse_k": rmse,
        "r2": r2,
    }


def train_tg(
    checkpoint_path,
    epochs=EPOCHS,
    max_train_steps=None,
    max_test_steps=None,
):
    torch.manual_seed(SEED)

    dataset = TgDataset(CSV_PATH)
    test_size = int(len(dataset) * TEST_FRACTION)
    train_size = len(dataset) - test_size

    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    train_targets = torch.tensor(
        [dataset.targets[i] for i in train_set.indices],
        dtype=torch.float32,
    )
    target_mean = train_targets.mean().item()
    target_std = train_targets.std().item()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TgRegressor(
        MODEL_NAME,
        checkpoint_path,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=LR,
    )

    model.encoder.eval()
    model.head.train()

    train_steps = 0

    for epoch in range(epochs):
        total_loss = 0.0
        epoch_steps = 0

        for smiles_batch, target in train_loader:
            input_ids, attention_mask = _tokenize(
                smiles_batch,
                tokenizer,
            )
            target = target.float().to(DEVICE)
            target = (target - target_mean) / target_std

            prediction = model(
                input_ids,
                attention_mask,
            )

            loss = F.mse_loss(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_steps += 1
            train_steps += 1

            if (
                max_train_steps is not None
                and train_steps >= max_train_steps
            ):
                break

        if epoch_steps:
            print(
                f"Tg | epoch {epoch + 1}/{epochs} "
                f"| loss {total_loss / epoch_steps:.4f}"
            )

        if (
            max_train_steps is not None
            and train_steps >= max_train_steps
        ):
            break

    model.encoder.eval()
    model.head.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for step, (smiles_batch, target) in enumerate(test_loader):
            input_ids, attention_mask = _tokenize(
                smiles_batch,
                tokenizer,
            )

            prediction = model(
                input_ids,
                attention_mask,
            )
            prediction = prediction * target_std + target_mean

            y_true.extend(target.tolist())
            y_pred.extend(prediction.cpu().tolist())

            if (
                max_test_steps is not None
                and step + 1 >= max_test_steps
            ):
                break

    return _metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    metrics = train_tg(args.checkpoint)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
