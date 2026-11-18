from pathlib import Path
from time import perf_counter
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# -------------------------
# Task
# -------------------------
TASK = "regression"  # "regression" 또는 "classification"

# -------------------------
# Paths
# -------------------------
ROOT_DIR = Path("/home/ubuntu/datasets")
DATA_DIR = ROOT_DIR / "image_inte_3"

# Task별 결과 저장 폴더
OUTPUT_DIRS = {
    "classification": (
        ROOT_DIR
        / "classification_model"
        / "image_classification_model"
    ),
    "regression": (
        ROOT_DIR
        / "regression_model"
        / "image_regression_model"
    ),
}

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
NUM_WORKERS = 0

# -------------------------
# Training
# -------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10

BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4

HUBER_DELTA = 20.0
CLASSIFICATION_THRESHOLD = 0.5

PRETRAINED = True

# -------------------------
# Classification labels
# -------------------------
# flank_wear = 1, adhesion 및 혼합 마모 = 0
TYPE_TO_LABEL = {
    "flank_wear": 1,
    "adhesion": 0,
    "flank_wear+adhesion": 0,
}


OUTPUT_DIR = OUTPUT_DIRS[TASK]
BEST_MODEL_NAME = (
    "best_resnet_regression.pth"
    if TASK == "regression"
    else "best_resnet_classification.pth"
)
BEST_PATH = OUTPUT_DIR / BEST_MODEL_NAME


# =========================================================
# 2. Data: Metadata 처리, Dataset, DataLoader
# =========================================================

def normalize_set_column(series: pd.Series) -> pd.Series:
    """'Set4', '4', 4 등의 Set 표기를 정수로 통일."""
    normalized = (
        series.astype(str)
        .str.strip()
        .str.replace(r"(?i)^set\s*", "", regex=True)
    )
    numeric = pd.to_numeric(normalized, errors="raise")

    if numeric.isna().any() or numeric.mod(1).ne(0).any():
        raise ValueError("Set 열에 정수로 변환할 수 없는 값이 있습니다.")

    return numeric.astype(int)


def attach_type_from_labels(
    metadata: pd.DataFrame,
    labels_path: Path,
) -> pd.DataFrame:
    """
    metadata.csv에 type 열이 없는 경우 labels.csv에서 type을 매칭.
    매칭 키: metadata의 (Set, original_filename)
              labels의 (Set, ImageName)
    """
    if "type" in metadata.columns:
        return metadata

    if not labels_path.is_file():
        raise FileNotFoundError(
            f"metadata.csv에 type 열이 없고 labels.csv도 없습니다: {labels_path}"
        )

    if "original_filename" not in metadata.columns:
        raise ValueError(
            "type 매칭을 위해 metadata.csv에 original_filename 열이 필요합니다."
        )

    labels = pd.read_csv(
        labels_path,
        dtype=str,
        keep_default_na=False,
    )

    required = {"Set", "ImageName", "type"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            f"labels.csv 필수 컬럼 누락: {sorted(missing)}"
        )

    labels["Set"] = normalize_set_column(labels["Set"])
    labels["ImageName"] = (
        labels["ImageName"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.rsplit("/", n=1)
        .str[-1]
        .str.strip()
    )
    labels["type"] = (
        labels["type"].astype(str).str.strip().str.lower()
    )

    if labels.duplicated(["Set", "ImageName"]).any():
        duplicated = labels.loc[
            labels.duplicated(["Set", "ImageName"], keep=False),
            ["Set", "ImageName"],
        ].head()
        raise ValueError(
            "labels.csv에 중복된 (Set, ImageName)이 있습니다.\n"
            f"{duplicated.to_string(index=False)}"
        )

    metadata = metadata.copy()
    metadata["original_filename"] = (
        metadata["original_filename"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.rsplit("/", n=1)
        .str[-1]
        .str.strip()
    )

    metadata = metadata.merge(
        labels[["Set", "ImageName", "type"]],
        left_on=["Set", "original_filename"],
        right_on=["Set", "ImageName"],
        how="left",
        validate="many_to_one",
    )
    metadata.drop(columns=["ImageName"], inplace=True)

    return metadata


def load_and_prepare_metadata(
    data_dir: Path,
    root_dir: Path,
    task: str,
) -> pd.DataFrame:
    """metadata 로드, 컬럼 검증, 라벨 처리, 이미지 경로 생성."""
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv를 찾을 수 없습니다: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    required_columns = {"Set", "split", "filename"}
    if task == "regression":
        required_columns.add("wear")

    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(
            f"metadata.csv 필수 컬럼 누락: {sorted(missing)}"
        )

    metadata = metadata.copy()
    metadata["Set"] = normalize_set_column(metadata["Set"])
    metadata["split"] = (
        metadata["split"].astype(str).str.strip().str.lower()
    )
    metadata["filename"] = (
        metadata["filename"].astype(str).str.strip()
    )

    allowed_splits = {"train", "valid", "test"}
    invalid_splits = set(metadata["split"].unique()) - allowed_splits
    if invalid_splits:
        raise ValueError(f"잘못된 split 값: {sorted(invalid_splits)}")

    if metadata["filename"].eq("").any():
        raise ValueError("metadata.csv의 filename에 빈 값이 있습니다.")

    if task == "regression":
        metadata["wear"] = pd.to_numeric(
            metadata["wear"], errors="raise"
        )
        if not np.isfinite(metadata["wear"].to_numpy(dtype=float)).all():
            raise ValueError("wear에 NaN 또는 무한대 값이 있습니다.")

    elif task == "classification":
        metadata = attach_type_from_labels(
            metadata,
            root_dir / "labels.csv",
        )

        metadata["type"] = (
            metadata["type"].astype(str).str.strip().str.lower()
        )

        invalid_type_mask = metadata["type"].isin(["", "nan", "none"])
        if invalid_type_mask.any():
            examples = metadata.loc[
                invalid_type_mask, ["Set", "filename"]
            ].head()
            raise ValueError(
                "type 라벨이 누락된 데이터가 있습니다.\n"
                f"{examples.to_string(index=False)}"
            )

        metadata["label"] = metadata["type"].map(TYPE_TO_LABEL).astype(int)

        print("\n[Classification Label Distribution]")
        print(
            metadata.groupby(["type", "label"])
            .size()
            .rename("count")
            .to_string()
        )

    metadata["image_path"] = [
        data_dir / split / filename
        for split, filename in zip(metadata["split"], metadata["filename"])
    ]

    for split in ("train", "valid", "test"):
        if not metadata["split"].eq(split).any():
            raise ValueError(f"{split} 데이터가 비어 있습니다.")

    return metadata


def build_image_transform(image_size: int) -> transforms.Compose:
    """ImageNet pretrained ResNet18에 맞춘 이미지 전처리."""
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

class ToolImageDataset(Dataset):
    """이미지와 task별 target을 반환하는 공통 Dataset."""

    def __init__(
        self,
        rows: pd.DataFrame,
        task: str,
        image_transform,
    ):
        self.rows = rows.reset_index(drop=True).copy()
        self.task = task
        self.image_transform = image_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]

        with Image.open(row["image_path"]) as source:
            image = self.image_transform(source.convert("RGB"))

        if self.task == "regression":
            target = float(row["wear"])
        else:
            target = float(row["label"])

        target = torch.tensor(target, dtype=torch.float32)
        return image, target


def build_dataloaders(
    metadata: pd.DataFrame,
    task: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
):
    """Train/Valid/Test DataLoader 생성."""
    image_transform = build_image_transform(image_size)
    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders = {}
    for split in ("train", "valid", "test"):
        rows = metadata.loc[metadata["split"].eq(split)].copy()
        dataset = ToolImageDataset(
            rows=rows,
            task=task,
            image_transform=image_transform,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            generator=generator if split == "train" else None,
            drop_last=False,
        )

    return loaders


# =========================================================
# 3. Model: ResNet18 및 Backbone 동결 설정
# =========================================================

class ResNet18ToolWearModel(nn.Module):
    """
    ResNet18 이미지 인코더 + task별 단일 출력 헤드.
    Regression: 연속형 wear 값
    Classification: flank_wear 양성 클래스의 logit
    """

    def __init__(
        self,
        task: str,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if task not in {"regression", "classification"}:
            raise ValueError(f"지원하지 않는 task: {task}")

        self.task = task
        self.image_encoder = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.image_encoder.fc = nn.Identity()

        # 기존 학습 전략: 앞단은 동결하고 layer3/layer4 및 head 학습
        self.frozen_blocks = (
            "conv1",
            "bn1",
            "layer1",
            "layer2",
        )

        for block_name in self.frozen_blocks:
            getattr(self.image_encoder, block_name).requires_grad_(False)

        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def train(self, mode: bool = True):
        """동결된 BatchNorm 층의 running statistics가 바뀌지 않도록 eval 유지."""
        super().train(mode)

        for block_name in self.frozen_blocks:
            getattr(self.image_encoder, block_name).eval()

        return self

    def forward(self, image):
        features = self.image_encoder(image)
        output = self.head(features)
        return output.squeeze(1)


def build_criterion(task: str):
    if task == "regression":
        return nn.HuberLoss(delta=HUBER_DELTA)
    if task == "classification":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"지원하지 않는 task: {task}")


def build_optimizer(
    model: ResNet18ToolWearModel,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
):
    """학습 가능한 ResNet layer3/layer4와 head에 서로 다른 LR 적용."""
    backbone_parameters = (
        list(model.image_encoder.layer3.parameters())
        + list(model.image_encoder.layer4.parameters())
    )

    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


# =========================================================
# 4. Metrics: 회귀 / 분류 평가 지표
# =========================================================

def regression_metrics(y_true, y_pred, huber_delta: float):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    abs_error = np.abs(error)
    squared_error = error ** 2

    huber_values = np.where(
        abs_error <= huber_delta,
        0.5 * squared_error,
        huber_delta * (abs_error - 0.5 * huber_delta),
    )

    total_variation = np.sum((y_true - y_true.mean()) ** 2)

    return {
        "huber": float(huber_values.mean()),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "r2": (
            float(1.0 - squared_error.sum() / total_variation)
            if total_variation > 0
            else float("nan")
        ),
        "bias": float(error.mean()),
        "p90": float(np.percentile(abs_error, 90)),
    }


def classification_metrics(
    y_true,
    probabilities,
    threshold: float = 0.5,
):
    """0=non-flank, 1=flank_wear 기준 이진 분류 지표."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    recalls = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.full(2, np.nan, dtype=float),
        where=cm.sum(axis=1) > 0,
    )

    both_classes = len(np.unique(y_true)) == 2

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "recall_nonflank": float(recalls[0]),
        "recall_flank": float(recalls[1]),
        "auc_flank": (
            float(roc_auc_score(y_true, probabilities))
            if both_classes else float("nan")
        ),
        "ap_nonflank": (
            float(average_precision_score(
                1 - y_true,
                1 - probabilities,
            ))
            if both_classes else float("nan")
        ),
        "nonflank_correct": int(cm[0, 0]),
        "nonflank_total": int(cm[0].sum()),
        "n": int(len(y_true)),
    }


# =========================================================
# 5. Engine: train_epoch, evaluate_model
# =========================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
):
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits_or_values = model(images)
        loss = criterion(logits_or_values, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

    return loss_sum / sample_count


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    task: str,
    classification_threshold: float,
):
    model.eval()

    loss_sum = 0.0
    sample_count = 0
    targets_all = []
    outputs_all = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

        targets_all.extend(targets.cpu().numpy().tolist())
        outputs_all.extend(outputs.cpu().numpy().tolist())


    y_true = np.asarray(targets_all)
    raw_outputs = np.asarray(outputs_all)

    mean_loss = loss_sum / sample_count

    records = loader.dataset.rows[["Set", "filename"]].copy()

    if task == "regression":
        records["wear"] = y_true
        records["prediction"] = raw_outputs
        metrics = regression_metrics(
            y_true,
            raw_outputs,
            huber_delta=HUBER_DELTA,
        )

    else:
        # 수치 안정성을 위해 sigmoid를 numpy에서 계산
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw_outputs, -80, 80)))
        predictions = (probabilities >= classification_threshold).astype(int)

        records["type"] = loader.dataset.rows["type"].to_numpy()
        records["true_label"] = y_true.astype(int)
        records["probability_flank"] = probabilities
        records["prediction_label"] = predictions

        metrics = classification_metrics(
            y_true,
            probabilities,
            threshold=classification_threshold,
        )

    return mean_loss, metrics, records


# =========================================================
# 6. Trainer: 학습 루프, Early Stopping, Checkpoint 저장
# =========================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    val_loss: float,
    val_metrics: dict,
    task: str,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")

    checkpoint = {
        "task": task,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "type_to_label": TYPE_TO_LABEL,
        "config": {
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "backbone_lr": BACKBONE_LR,
            "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "huber_delta": HUBER_DELTA,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
        },
    }

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def train_model(
    model: nn.Module,
    loaders: dict,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    task: str,
    output_dir: Path,
    best_path: Path,
):
    best_metric = (
    float("inf")
    if task == "regression"
    else float("-inf")
    )
    best_epoch = 0
    stale_epochs = 0
    history = []

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== Training Start ==========")
    print("Task:", task)
    print("Device:", device)
    print("Best Model Path:", best_path)

    for epoch in range(1, MAX_EPOCHS + 1):
        start_time = perf_counter()

        train_loss = train_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_metrics, _ = evaluate_model(
            model=model,
            loader=loaders["valid"],
            criterion=criterion,
            device=device,
            task=task,
            classification_threshold=CLASSIFICATION_THRESHOLD,
        )

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise ValueError(
                f"Epoch {epoch}: train/validation loss가 유효하지 않습니다."
            )

        # =====================================================
        # Task별 Best Checkpoint 및 Early Stopping 기준
        # =====================================================

        if task == "regression":
            current_metric = val_loss
            is_best = current_metric < best_metric

        else:
            current_metric = val_metrics["macro_f1"]
            is_best = current_metric > best_metric


        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            stale_epochs = 0

            save_checkpoint(
                path=best_path,
                model=model,
                epoch=epoch,
                val_loss=val_loss,
                val_metrics=val_metrics,
                task=task,
            )

        else:
            stale_epochs += 1

        elapsed = perf_counter() - start_time

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "seconds": elapsed,
        }
        history.append(history_row)
        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
        )

        if task == "regression":
            print(
                f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Train Huber {train_loss:.3f} | "
                f"Val Huber {val_loss:.3f} | "
                f"Val MAE {val_metrics['mae']:.3f} µm | "
                f"Val RMSE {val_metrics['rmse']:.3f} µm | "
                f"미개선 {stale_epochs}/{PATIENCE} | "
                f"{elapsed:.1f}초",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Train BCE {train_loss:.4f} | "
                f"Val BCE {val_loss:.4f} | "
                f"Val Acc {val_metrics['accuracy']:.4f} | "
                f"Val Macro F1 {val_metrics['macro_f1']:.4f} | "
                f"Val Recall Flank {val_metrics['recall_flank']:.4f} | "
                f"미개선 {stale_epochs}/{PATIENCE} | "
                f"{elapsed:.1f}초",
                flush=True,
            )

        scheduler.step()

        if stale_epochs >= PATIENCE:
            print("\nEarly Stopping")
            break

    if not best_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint가 생성되지 않았습니다: {best_path}"
        )

    print("\n========== Training Completed ==========")
    print("Best Epoch:", best_epoch)

    if task == "regression":
        print("Best Validation Loss:", best_metric)
    else:
        print("Best Validation Macro F1:", best_metric)

    print("Saved Model:", best_path)

    return history, best_epoch, best_metric


def load_best_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    expected_task: str,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint


def save_test_results(
    test_records: pd.DataFrame,
    task: str,
    output_dir: Path,
):
    """전체 Test 및 Set별 평가 결과 저장."""
    test_groups = {"Test Overall": test_records}

    for set_number, group in test_records.groupby("Set"):
        test_groups[f"Set{set_number}"] = group

    metric_rows = []

    for group_name, records in test_groups.items():
        if task == "regression":
            metrics = regression_metrics(
                records["wear"].to_numpy(),
                records["prediction"].to_numpy(),
                huber_delta=HUBER_DELTA,
            )
            row = {
                "범위": group_name,
                "기록 수": len(records),
                **metrics,
            }
        else:
            metrics = classification_metrics(
                records["true_label"].to_numpy(),
                records["probability_flank"].to_numpy(),
                threshold=CLASSIFICATION_THRESHOLD,
            )
            row = {
                "범위": group_name,
                **metrics,
            }

        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)

    test_records.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )
    metrics_df.to_csv(
        output_dir / "test_metrics.csv",
        index=False,
    )

    return metrics_df


# =========================================================
# 7. Utils: Seed 고정 및 공통 유틸리티
# =========================================================

def set_seed(seed: int):
    """Python, NumPy, PyTorch 난수 시드 고정."""
    #random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)

    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed(seed)
    #    torch.cuda.manual_seed_all(seed)

    # 재현성을 우선하는 설정
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 8. Main: 전체 학습 실행
# =========================================================

def main():
    set_seed(SEED)
    train_generator = torch.Generator().manual_seed(SEED)
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Metadata 준비
    metadata = load_and_prepare_metadata(
        data_dir=DATA_DIR,
        root_dir=ROOT_DIR,
        task=TASK,
    )

    # 2) DataLoader 생성
    loaders = build_dataloaders(
        metadata=metadata,
        task=TASK,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
        seed=SEED,
    )

    # 3) Model / Loss / Optimizer / Scheduler
    model = ResNet18ToolWearModel(
        task=TASK,
        pretrained=PRETRAINED,
    ).to(device)

    criterion = build_criterion(TASK)

    optimizer = build_optimizer(
        model=model,
        backbone_lr=BACKBONE_LR,
        head_lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
    )

    # 4) Training
    train_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task=TASK,
        output_dir=OUTPUT_DIR,
        best_path=BEST_PATH,
    )

    # 5) Best checkpoint 로드
    checkpoint = load_best_checkpoint(
        model=model,
        checkpoint_path=BEST_PATH,
        device=device,
        expected_task=TASK,
    )
    print("\nBest Checkpoint Loaded")
    print("Best Epoch:", checkpoint["epoch"])
    print("Best Validation Loss:", checkpoint["val_loss"])

    # 6) Test 평가
    test_loss, test_metrics, test_records = evaluate_model(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        task=TASK,
        classification_threshold=CLASSIFICATION_THRESHOLD,
    )

    print("\n========== Test Evaluation ==========")
    print("Task:", TASK)
    print("Test Loss:", test_loss)
    print(pd.DataFrame([test_metrics], index=["Test Overall"]).to_string())

    # 7) 전체 및 Set별 결과 저장
    test_metrics_df = save_test_results(
        test_records=test_records,
        task=TASK,
        output_dir=OUTPUT_DIR,
    )

    print("\n========== Test Metrics by Set ==========")
    print(test_metrics_df.to_string(index=False))

    print("\n========== Results Saved ==========")
    print("Output Directory:", OUTPUT_DIR)
    print("Best Model:", BEST_PATH)
    print("History:", OUTPUT_DIR / "history.csv")
    print("Test Predictions:", OUTPUT_DIR / "test_predictions.csv")
    print("Test Metrics:", OUTPUT_DIR / "test_metrics.csv")


if __name__ == "__main__":
    main()