from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from shutil import rmtree
from time import perf_counter
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from sklearn.metrics import f1_score, r2_score, confusion_matrix


# =========================================================
# 1. Configuration
# =========================================================

ROOT = Path("/home/ubuntu/datasets")

SENSOR_OUTPUT_DIR = (
    ROOT / "outputs" / "sensor_force_y_hampel21_sigma4_rms100"
)
MANIFEST_PATH = SENSOR_OUTPUT_DIR / "manifest_sensor_preprocessed.csv"
NORMALIZATION_PATH = SENSOR_OUTPUT_DIR / "normalization.json"

IMAGE_DIR = ROOT / "image_inte_3"

IMAGE_CHECKPOINT = (
    ROOT / "regression_model"
    / "image_regression_model" / "best_resnet_regression.pth"
)
FY_CHECKPOINT = (
    ROOT / "regression_model"
    / "sensor_regression_model" / "best_1dcnn_regression.pth"
)

# "pretrained": 사전학습 이미지/Force Y 인코더 가중치 사용
# "random_init": 두 인코더 모두 무작위 초기화
INIT_MODE = "pretrained"

# "partial_finetune": 이미지 ResNet18의 초기 블록을 동결하고 나머지 학습
# "freeze": 이미지/Force Y 인코더 전체 동결
ENCODER_MODE = "partial_finetune"

OUTPUT_DIR = ROOT / "multimodal_model" / "multitask_weight_init"
BEST_MODEL_PATH = OUTPUT_DIR / "best_mm.pth"
PARETO_TEMP_DIR = OUTPUT_DIR / "_pareto_candidates"

SEED = 42
NUM_WORKERS = 0

WINDOW_SIZE = 512
IMAGE_SIZE = 224
BATCH_SIZE = 32

MAX_EPOCHS = 100
PATIENCE = 10

FUSION_EMBED_DIM = 256

MM_IMAGE_LR = 1e-4
MM_SENSOR_LR = 1e-3
MM_CLASSIFIER_LR = 1e-3
WEIGHT_DECAY = 1e-4

HUBER_DELTA = 20.0
CLS_LOSS_WEIGHT = 100.0
CLS_THRESHOLD = 0.50

SENSOR_CHANNEL = "force_y"
SENSOR_CHANNEL_INDEX = 3
USE_HAMPEL = True
HAMPEL_WINDOW_SIZE = 21
HAMPEL_N_SIGMAS = 4
RMS_WINDOW_MS = 100

WEAR_TYPE_TO_LABEL = {
    "flank_wear": 1,
    "adhesion": 0,
    "flank_wear+adhesion": 0,
}


# =========================================================
# 2. Reproducibility / Utility
# =========================================================

def set_seed(seed: int) -> None:
    #random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)

    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed(seed)
    #    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_data_path(path_value: str | Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else ROOT / path


def normalize_set_column(series: pd.Series) -> pd.Series:
    values = (
        series.astype(str)
        .str.strip()
        .str.replace(r"(?i)^set\s*", "", regex=True)
    )
    return pd.to_numeric(values).astype(int)


# =========================================================
# 3. Load Sensor Manifest and Image Metadata
# =========================================================

def load_paired_metadata() -> pd.DataFrame:
    sensor_data = pd.read_csv(MANIFEST_PATH)

    sensor_data["Set"] = normalize_set_column(sensor_data["Set"])
    sensor_data["split"] = (
        sensor_data["split"].astype(str).str.strip().str.lower()
    )
    sensor_data["SensorName"] = (
        sensor_data["SensorName"].astype(str).str.strip()
    )
    sensor_data["wear"] = pd.to_numeric(sensor_data["wear"])
    sensor_data["n_windows"] = pd.to_numeric(
        sensor_data["n_windows"]
    ).astype(int)

    sensor_data = sensor_data.loc[
        sensor_data["cut_status"].eq("detected")
        & sensor_data["windows_path"].notna()
        & sensor_data["n_windows"].gt(0)
    ].copy()

    sensor_data["npy_path"] = sensor_data["windows_path"].map(
        resolve_data_path
    )

    image_meta = pd.read_csv(
        IMAGE_DIR / "metadata.csv",
        usecols=["Set", "original_filename", "filename"],
    ).rename(columns={"filename": "image_filename"})

    image_meta["Set"] = normalize_set_column(image_meta["Set"])

    pairs = sensor_data.merge(
        image_meta,
        on=["Set", "original_filename"],
        how="inner",
    )

    wear_type = (
        pairs["type"].astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
    )
    pairs["type_target"] = wear_type.map(WEAR_TYPE_TO_LABEL).astype(int)

    pairs["image_path"] = [
        IMAGE_DIR / split / filename
        for split, filename in zip(
            pairs["split"], pairs["image_filename"]
        )
    ]

    return pairs.reset_index(drop=True)


# =========================================================
# 4. Dataset / DataLoader
# =========================================================

class MultimodalWindowDataset(Dataset):
    def __init__(self, split_rows: pd.DataFrame):
        self.records = list(split_rows.itertuples(index=False))
        self.images = []
        self.windows = []
        self.offsets = [0]

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        for record in self.records:
            with Image.open(record.image_path) as image:
                self.images.append(
                    self.image_transform(image.convert("RGB"))
                )

            windows = np.load(
                record.npy_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            windows_tensor = torch.from_numpy(
                np.array(windows, dtype=np.float32, copy=True)
            )

            self.windows.append(windows_tensor)
            self.offsets.append(
                self.offsets[-1] + len(windows_tensor)
            )

        self.total_windows = self.offsets[-1]

    def __len__(self):
        return self.total_windows

    def __getitem__(self, index):
        record_index = bisect_right(self.offsets, index) - 1
        window_index = index - self.offsets[record_index]

        record = self.records[record_index]

        return {
            "image": self.images[record_index],
            "sensor": self.windows[record_index][window_index],
            "wear": torch.tensor(
                [float(record.wear)], dtype=torch.float32
            ),
            "type_target": torch.tensor(
                [int(record.type_target)], dtype=torch.float32
            ),
            "set_number": int(record.Set),
            "sensor_name": record.SensorName,
            "n_windows": len(self.windows[record_index]),
        }


def build_dataloaders(pairs: pd.DataFrame):
    datasets = {}
    loaders = {}

    for split in ("train", "valid", "test"):
        dataset = MultimodalWindowDataset(
            pairs.loc[pairs["split"].eq(split)].reset_index(drop=True)
        )
        datasets[split] = dataset

        loaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            generator=(
                torch.Generator().manual_seed(SEED)
                if split == "train" else None
            ),
            drop_last=False,
        )

        print(
            f"{split:5s} | "
            f"기록 {len(dataset.records):,}개 | "
            f"윈도우 {len(dataset):,}개"
        )

    return datasets, loaders


# =========================================================
# 5. Model
# =========================================================

class ImageForceYFusionModel(nn.Module):
    def __init__(
        self,
        init_mode: str,
        encoder_mode: str,
        image_state: dict | None = None,
        fy_encoder_state: dict | None = None,
    ):
        super().__init__()

        self.init_mode = init_mode
        self.encoder_mode = encoder_mode

        self.image_encoder = resnet18(weights=None)
        self.image_encoder.fc = nn.Identity()

        self.fy_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        if init_mode == "pretrained":
            image_encoder_state = {
                key.removeprefix("image_encoder."): value
                for key, value in image_state.items()
                if key.startswith("image_encoder.")
            }
            self.image_encoder.load_state_dict(
                image_encoder_state,
                strict=True,
            )
            self.fy_encoder.load_state_dict(
                fy_encoder_state,
                strict=True,
            )

        self.frozen_image_blocks = (
            "conv1", "bn1", "layer1", "layer2"
        )

        if encoder_mode == "freeze":
            self.image_encoder.requires_grad_(False)
            self.fy_encoder.requires_grad_(False)
        elif encoder_mode == "partial_finetune":
            for block in self.frozen_image_blocks:
                getattr(self.image_encoder, block).requires_grad_(False)

        self.fusion_embedding = nn.Sequential(
            nn.Linear(512 + 128, FUSION_EMBED_DIM),
            nn.BatchNorm1d(FUSION_EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.regressor = nn.Sequential(
            nn.Linear(FUSION_EMBED_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(FUSION_EMBED_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)

        if self.encoder_mode == "freeze":
            self.image_encoder.eval()
            self.fy_encoder.eval()
        elif self.encoder_mode == "partial_finetune":
            for block in self.frozen_image_blocks:
                getattr(self.image_encoder, block).eval()

        return self

    def forward(self, image, sensor):
        image_features = self.image_encoder(image)
        sensor_features = self.fy_encoder(sensor).flatten(1)

        fused = torch.cat(
            (image_features, sensor_features),
            dim=1,
        )
        embedding = self.fusion_embedding(fused)

        wear_prediction = self.regressor(embedding)
        type_logit = self.classifier(embedding)

        return wear_prediction, type_logit


# =========================================================
# 6. Metrics / Prediction
# =========================================================

def regression_metrics(records: pd.DataFrame) -> dict:
    actual = records["wear"].to_numpy(dtype=np.float64)
    predicted = records["prediction"].to_numpy(dtype=np.float64)

    error = predicted - actual
    abs_error = np.abs(error)
    huber_values = np.where(
        abs_error <= HUBER_DELTA,
        0.5 * error**2,
        HUBER_DELTA * (abs_error - 0.5 * HUBER_DELTA),
    )

    return {
        "huber": float(huber_values.mean()),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(error.mean()),
    }


@torch.inference_mode()
def predict_records(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    window_rows = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        sensor = batch["sensor"].to(device, non_blocking=True)

        wear_prediction, type_logit = model(image, sensor)

        predictions = wear_prediction.squeeze(1).cpu().numpy()
        probabilities = type_logit.squeeze(1).sigmoid().cpu().numpy()

        window_rows.extend(zip(
            batch["set_number"].tolist(),
            batch["sensor_name"],
            predictions.tolist(),
            probabilities.tolist(),
        ))

    window_predictions = pd.DataFrame(
        window_rows,
        columns=[
            "Set", "SensorName", "prediction", "type_probability"
        ],
    )

    averaged = window_predictions.groupby(
        ["Set", "SensorName"],
        as_index=False,
    ).agg(
        prediction=("prediction", "mean"),
        type_probability=("type_probability", "mean"),
    )

    expected = pd.DataFrame([
        {
            "Set": int(record.Set),
            "SensorName": record.SensorName,
            "wear": float(record.wear),
            "type_target": int(record.type_target),
            "n_windows": len(windows),
        }
        for record, windows in zip(
            loader.dataset.records,
            loader.dataset.windows,
        )
    ])

    return expected.merge(
        averaged,
        on=["Set", "SensorName"],
        how="inner",
    )


def add_classification_metrics(
    records: pd.DataFrame,
    threshold: float = CLS_THRESHOLD,
) -> dict:
    actual = records["type_target"].to_numpy(dtype=int)
    predicted = (
        records["type_probability"].to_numpy(dtype=float) >= threshold
    ).astype(int)

    return {
        "macro_f1": float(f1_score(
            actual,
            predicted,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )),
        "flank_wear_f1": float(f1_score(
            actual,
            predicted,
            pos_label=1,
            average="binary",
            zero_division=0,
        )),
    }


# =========================================================
# 7. Pareto Checkpoint Utilities
# =========================================================

def dominates(left: dict, right: dict) -> bool:
    return (
        left["valid_mae"] <= right["valid_mae"]
        and left["valid_macro_f1"] >= right["valid_macro_f1"]
        and (
            left["valid_mae"] < right["valid_mae"]
            or left["valid_macro_f1"] > right["valid_macro_f1"]
        )
    )


def save_candidate_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    valid_metrics: dict,
    normalization: dict,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_metrics": valid_metrics,
        "init_mode": INIT_MODE,
        "encoder_mode": ENCODER_MODE,
        "fusion_embed_dim": FUSION_EMBED_DIM,
        "image_checkpoint": (
            str(IMAGE_CHECKPOINT) if INIT_MODE == "pretrained" else None
        ),
        "fy_checkpoint": (
            str(FY_CHECKPOINT) if INIT_MODE == "pretrained" else None
        ),
        "image_size": (IMAGE_SIZE, IMAGE_SIZE),
        "sensor_channel": SENSOR_CHANNEL,
        "sensor_channel_index": SENSOR_CHANNEL_INDEX,
        "use_hampel": USE_HAMPEL,
        "hampel_window_size": HAMPEL_WINDOW_SIZE,
        "hampel_n_sigmas": HAMPEL_N_SIGMAS,
        "rms_window_ms": RMS_WINDOW_MS,
        "window_size": WINDOW_SIZE,
        "fy_mean": float(normalization["sensor_mean"]),
        "fy_std": float(normalization["sensor_std"]),
        "cls_loss_weight": CLS_LOSS_WEIGHT,
        "cls_threshold": CLS_THRESHOLD,
        "wear_type_to_label": WEAR_TYPE_TO_LABEL,
    }

    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def select_pareto_checkpoint(
    candidates: list[dict],
    candidate_dir: Path,
    best_path: Path,
) -> tuple[dict, pd.DataFrame]:
    pareto_table = pd.DataFrame(candidates)

    mae = pareto_table["valid_mae"]
    f1 = pareto_table["valid_macro_f1"]

    mae_span = mae.max() - mae.min()
    f1_span = f1.max() - f1.min()

    mae_penalty = (
        (mae - mae.min()) / mae_span
        if mae_span > 0 else 0.0
    )
    f1_penalty = (
        (f1.max() - f1) / f1_span
        if f1_span > 0 else 0.0
    )

    pareto_table["balance_score"] = np.sqrt(
        mae_penalty**2 + f1_penalty**2
    )

    selected = pareto_table.sort_values(
        [
            "balance_score",
            "valid_mae",
            "valid_macro_f1",
            "epoch",
        ],
        ascending=[True, True, False, True],
    ).iloc[0]

    selected_path = candidate_dir / selected["checkpoint"]
    selected_checkpoint = torch.load(
        selected_path,
        map_location="cpu",
        weights_only=True,
    )

    best_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = best_path.with_suffix(".tmp")
    torch.save(selected_checkpoint, temporary_path)
    temporary_path.replace(best_path)

    return selected.to_dict(), pareto_table


# =========================================================
# 8. Training
# =========================================================

def train_model(
    model: nn.Module,
    loaders: dict,
    device: torch.device,
    normalization: dict,
):
    train_loader = loaders["train"]

    avg_train_windows = (
        len(train_loader.dataset)
        / len(train_loader.dataset.records)
    )

    image_params = [
        parameter
        for parameter in model.image_encoder.parameters()
        if parameter.requires_grad
    ]

    sensor_and_regression_params = [
        parameter
        for module in (
            model.fy_encoder,
            model.fusion_embedding,
            model.regressor,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    param_groups = []

    if image_params:
        param_groups.append({
            "name": "image",
            "params": image_params,
            "lr": MM_IMAGE_LR,
        })

    if sensor_and_regression_params:
        param_groups.append({
            "name": "sensor_and_regression",
            "params": sensor_and_regression_params,
            "lr": MM_SENSOR_LR,
        })

    param_groups.append({
        "name": "classifier",
        "params": model.classifier.parameters(),
        "lr": MM_CLASSIFIER_LR,
    })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PARETO_TEMP_DIR.exists():
        rmtree(PARETO_TEMP_DIR)
    PARETO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    pareto_candidates = []

    best_valid_mae = float("inf")
    best_valid_f1 = float("-inf")
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        start_time = perf_counter()
        model.train()

        huber_sum = 0.0
        bce_sum = 0.0
        seen_windows = 0

        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            sensor = batch["sensor"].to(device, non_blocking=True)
            wear = batch["wear"].to(device, non_blocking=True)
            type_target = batch["type_target"].to(
                device=device,
                dtype=wear.dtype,
                non_blocking=True,
            )
            n_windows = batch["n_windows"].to(
                device=device,
                dtype=wear.dtype,
                non_blocking=True,
            ).reshape(-1, 1)

            record_weight = avg_train_windows / n_windows

            optimizer.zero_grad(set_to_none=True)

            wear_prediction, type_logit = model(image, sensor)

            huber_per_window = F.huber_loss(
                wear_prediction,
                wear,
                delta=HUBER_DELTA,
                reduction="none",
            )
            bce_per_window = F.binary_cross_entropy_with_logits(
                type_logit,
                type_target,
                reduction="none",
            )

            huber_loss = (huber_per_window * record_weight).mean()
            bce_loss = (bce_per_window * record_weight).mean()
            total_loss = huber_loss + CLS_LOSS_WEIGHT * bce_loss

            total_loss.backward()
            optimizer.step()

            batch_size = wear.size(0)
            huber_sum += huber_loss.item() * batch_size
            bce_sum += bce_loss.item() * batch_size
            seen_windows += batch_size

        train_huber = huber_sum / seen_windows
        train_bce = bce_sum / seen_windows
        train_total_loss = (
            train_huber + CLS_LOSS_WEIGHT * train_bce
        )

        valid_records = predict_records(
            model,
            loaders["valid"],
            device,
        )
        valid_metrics = regression_metrics(valid_records)
        valid_metrics.update(
            add_classification_metrics(valid_records)
        )

        improved = (
            valid_metrics["mae"] < best_valid_mae
            or valid_metrics["macro_f1"] > best_valid_f1
        )

        best_valid_mae = min(
            best_valid_mae,
            valid_metrics["mae"],
        )
        best_valid_f1 = max(
            best_valid_f1,
            valid_metrics["macro_f1"],
        )
        stale_epochs = 0 if improved else stale_epochs + 1

        candidate_path = PARETO_TEMP_DIR / f"epoch_{epoch:03d}.pth"
        candidate = {
            "epoch": epoch,
            "valid_mae": valid_metrics["mae"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_huber": valid_metrics["huber"],
            "checkpoint": candidate_path.name,
        }

        dominated = any(
            dominates(previous, candidate)
            or (
                previous["valid_mae"] == candidate["valid_mae"]
                and previous["valid_macro_f1"]
                == candidate["valid_macro_f1"]
            )
            for previous in pareto_candidates
        )

        if not dominated:
            save_candidate_checkpoint(
                candidate_path,
                model,
                epoch,
                valid_metrics,
                normalization,
            )

            removed = [
                previous
                for previous in pareto_candidates
                if dominates(candidate, previous)
            ]

            pareto_candidates = [
                previous
                for previous in pareto_candidates
                if not dominates(candidate, previous)
            ]
            pareto_candidates.append(candidate)

            for previous in removed:
                (
                    PARETO_TEMP_DIR / previous["checkpoint"]
                ).unlink(missing_ok=True)

        scheduler.step(valid_metrics["huber"])

        elapsed = perf_counter() - start_time

        history.append({
            "epoch": epoch,
            "train_huber": train_huber,
            "train_bce": train_bce,
            "train_total_loss": train_total_loss,
            **{f"valid_{key}": value
               for key, value in valid_metrics.items()},
            "learning_rate_image": next(
                (
                    group["lr"] for group in optimizer.param_groups
                    if group["name"] == "image"
                ),
                None,
            ),
            "learning_rate_sensor": next(
                (
                    group["lr"] for group in optimizer.param_groups
                    if group["name"] == "sensor_and_regression"
                ),
                None,
            ),
            "learning_rate_classifier": next(
                group["lr"] for group in optimizer.param_groups
                if group["name"] == "classifier"
            ),
            "seconds": elapsed,
        })

        pd.DataFrame(history).to_csv(
            OUTPUT_DIR / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train Huber {train_huber:.3f} | "
            f"Train BCE {train_bce:.4f} | "
            f"Valid MAE {valid_metrics['mae']:.2f} μm | "
            f"Valid Macro F1 {valid_metrics['macro_f1']:.4f} | "
            f"Valid Huber {valid_metrics['huber']:.3f} | "
            f"Patience {stale_epochs}/{PATIENCE} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

        if stale_epochs >= PATIENCE:
            break

    selected, pareto_table = select_pareto_checkpoint(
        pareto_candidates,
        PARETO_TEMP_DIR,
        BEST_MODEL_PATH,
    )

    pareto_table.drop(columns=["checkpoint"]).to_csv(
        OUTPUT_DIR / "pareto_candidates.csv",
        index=False,
    )

    rmtree(PARETO_TEMP_DIR)

    print(
        f"\nBest Pareto Model | Epoch {int(selected['epoch'])} | "
        f"Valid MAE {selected['valid_mae']:.3f} μm | "
        f"Valid Macro F1 {selected['valid_macro_f1']:.4f}"
    )
    print(f"Saved: {BEST_MODEL_PATH}")

    return selected


# =========================================================
# 9. Test Evaluation
# =========================================================

def evaluate_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
):
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records = predict_records(model, test_loader, device)

    records["type_prediction"] = (
        records["type_probability"] >= CLS_THRESHOLD
    ).astype(int)
    records["error"] = records["prediction"] - records["wear"]
    records["abs_error"] = records["error"].abs()

    groups = [("Overall", records)]
    groups.extend(
        (f"Set{set_number}", group)
        for set_number, group in records.groupby("Set", sort=True)
    )

    metric_rows = []

    for group_name, group in groups:
        metrics = regression_metrics(group)
        metrics.update(add_classification_metrics(group))

        actual = group["type_target"].to_numpy(dtype=int)
        predicted = group["type_prediction"].to_numpy(dtype=int)
        cm = confusion_matrix(actual, predicted, labels=[0, 1])

        metric_rows.append({
            "scope": group_name,
            "records": len(group),
            **metrics,
            "nonflank_correct": int(cm[0, 0]),
            "nonflank_total": int(cm[0].sum()),
            "flank_correct": int(cm[1, 1]),
            "flank_total": int(cm[1].sum()),
        })

    metrics_df = pd.DataFrame(metric_rows)

    records.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_df.to_csv(
        OUTPUT_DIR / "test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[Test Results]")
    print(metrics_df.round(4).to_string(index=False))
    print(f"\nBest checkpoint: {BEST_MODEL_PATH}")


# =========================================================
# 10. Main
# =========================================================

def main():
    set_seed(SEED)
    device = get_device()

    with NORMALIZATION_PATH.open("r", encoding="utf-8") as file:
        normalization = json.load(file)

    image_state = None
    fy_encoder_state = None

    if INIT_MODE == "pretrained":
        image_checkpoint = torch.load(
            IMAGE_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
        fy_checkpoint = torch.load(
            FY_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

        image_state = image_checkpoint["model_state_dict"]
        fy_model_state = fy_checkpoint["model_state_dict"]

        fy_encoder_state = {
            key.removeprefix("sensor_encoder."): value
            for key, value in fy_model_state.items()
            if key.startswith("sensor_encoder.")
        }

    pairs = load_paired_metadata()
    _, loaders = build_dataloaders(pairs)

    model = ImageForceYFusionModel(
        init_mode=INIT_MODE,
        encoder_mode=ENCODER_MODE,
        image_state=image_state,
        fy_encoder_state=fy_encoder_state,
    ).to(device)

    print(f"Device: {device}")
    print(f"Initialization: {INIT_MODE}")
    print(f"Encoder training mode: {ENCODER_MODE}")

    train_model(
        model=model,
        loaders=loaders,
        device=device,
        normalization=normalization,
    )

    evaluate_test(
        model=model,
        test_loader=loaders["test"],
        device=device,
    )


if __name__ == "__main__":
    main()
