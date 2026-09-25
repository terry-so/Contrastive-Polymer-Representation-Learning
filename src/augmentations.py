import torch
from rdkit import Chem


APPROACHES = ("none", "original_mask", "enumeration_mask")


def normalize_dummy_atoms(smiles: str) -> str:
    out = []
    bracket_depth = 0

    for char in smiles:
        if char == "[":
            bracket_depth += 1
            out.append(char)
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            out.append(char)
        elif char == "*" and bracket_depth == 0:
            out.append("[*]")
        else:
            out.append(char)

    return "".join(out)


def randomize_psmiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    randomized = Chem.MolToSmiles(
        mol,
        canonical=False,
        doRandom=True,
        isomericSmiles=True,
    )
    return normalize_dummy_atoms(randomized)


def mask_token_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    mask_token_id: int,
    probability: float = 0.10,
) -> torch.Tensor:
    if mask_token_id is None:
        raise ValueError("Tokenizer has no mask token.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")

    masked = input_ids.clone()

    for row in range(input_ids.size(0)):
        eligible = attention_mask[row].bool() & ~special_tokens_mask[row].bool()
        positions = torch.nonzero(eligible, as_tuple=False).flatten()

        if positions.numel() == 0:
            continue

        n_mask = int(round(positions.numel() * probability))
        if probability > 0:
            n_mask = max(1, n_mask)
        n_mask = min(n_mask, int(positions.numel()))

        if n_mask == 0:
            continue

        chosen = positions[
            torch.randperm(positions.numel(), device=positions.device)[:n_mask]
        ]
        masked[row, chosen] = mask_token_id

    return masked


def _tokenize(smiles_batch, tokenizer, max_length, device):
    tokens = tokenizer(
        list(smiles_batch),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    return (
        tokens["input_ids"].to(device),
        tokens["attention_mask"].to(device),
        tokens["special_tokens_mask"].to(device),
    )


def make_contrastive_views(
    smiles_batch,
    tokenizer,
    approach,
    max_length,
    mask_probability,
    device,
):
    if approach not in APPROACHES:
        raise ValueError(f"Unknown approach: {approach}")

    original_ids, original_mask, special_mask = _tokenize(
        smiles_batch,
        tokenizer,
        max_length,
        device,
    )

    if approach == "none":
        return (
            (original_ids, original_mask),
            (original_ids.clone(), original_mask.clone()),
        )

    masked_ids = mask_token_ids(
        input_ids=original_ids,
        attention_mask=original_mask,
        special_tokens_mask=special_mask,
        mask_token_id=tokenizer.mask_token_id,
        probability=mask_probability,
    )

    if approach == "original_mask":
        return (
            (original_ids, original_mask),
            (masked_ids, original_mask),
        )

    enumerated = [randomize_psmiles(s) for s in smiles_batch]
    enum_ids, enum_mask, _ = _tokenize(
        enumerated,
        tokenizer,
        max_length,
        device,
    )

    return (
        (enum_ids, enum_mask),
        (masked_ids, original_mask),
    )
