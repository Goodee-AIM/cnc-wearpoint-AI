from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from time import perf_counter
import random

import numpy as np
import pandas as pd 

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# =========================================================
# 1. Configuration
# =========================================================

TASK = "regression"  # "regression" 또는 "classification"

ROOT_DIR = Path("/home/ubuntu/datasets")

# 두 번째 첨부 전처리 코드의 SENSOR_OUTPUT_DIR_NAME과 일치해야 함
SENSOR_DATA_DIR = (
    ROOT_DIR
    / "outputs"
    / "sensor_force_y_hampel21_sigma4_rms100"
)
MANIFEST_PATH = SENSOR_DATA_DIR / "manifest_sensor_preprocessed.csv"

OUTPUT_DIRS = {
    "classification": (
        ROOT_DIR
        / "classification_model"
        / "sensor_classification_model"
    ),
    "regression": (
        ROOT_DIR
        / "regression_model"
        / "sensor_regression_model"
    ),
}

SEED = 42
NUM_WORKERS = 0

WINDOW_SIZE = 512
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

HUBER_DELTA = 20.0
CLASSIFICATION_THRESHOLD = 0.5

OUTPUT_DIR = OUTPUT_DIRS[TASK]
BEST_MODEL_NAME = (
    "best_1dcnn_regression.pth"
    if TASK == "regression"
    else "best_1dcnn_classification.pth"
)
BEST_PATH = OUTPUT_DIR / BEST_MODEL_NAME


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

    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_data_path(path_value: str | Path) -> Path:
    """Manifest 경로가 절대경로/상대경로인 경우 모두 처리."""
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return ROOT_DIR / path

# =========================================================
# 3. Load processed sensor manifest
# =========================================================

def normalize_set_column(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.replace(r"(?i)^set\s*", "", regex=True)
    )

    return pd.to_numeric(normalized).astype(int)


def load_sensor_manifest(
    manifest_path: Path,
    task: str,
    window_size: int,
) -> pd.DataFrame:

    data = pd.read_csv(manifest_path)

    data = data.copy()

    data["Set"] = normalize_set_column(data["Set"])
    data["split"] = (
        data["split"].astype(str).str.strip().str.lower()
    )
    data["SensorName"] = data["SensorName"].astype(str).str.strip()
    data["type"] = (
        data["type"].astype(str).str.strip().str.lower()
    )

    data["wear"] = pd.to_numeric(data["wear"])
    data["target"] = pd.to_numeric(data["target"]).astype(int)
    data["n_windows"] = pd.to_numeric(data["n_windows"]).astype(int)

    # 전처리 성공 및 실제 윈도우가 존재하는 기록만 사용
    data = data.loc[
        data["cut_status"].eq("detected")
        & data["windows_path"].notna()
        & data["n_windows"].gt(0)
    ].copy()

    data["windows_path"] = data["windows_path"].map(resolve_data_path)

    data = data.reset_index(drop=True)

    print("\n[Sensor Manifest]")
    print(f"사용 기록 수: {len(data):,}")
    print(
        data.groupby("split")
        .agg(
            records=("SensorName", "size"),
            windows=("n_windows", "sum"),
        )
        .to_string()
    )

    return data

# =========================================================
# 4. Dataset and DataLoader
# =========================================================

class ForceYWindowDataset(Dataset):
    """
    하나의 센서 기록에 포함된 여러 윈도우를 개별 학습 샘플로 반환.
    반환 sensor shape: (1, WINDOW_SIZE)
    """

    def __init__(self, split_rows: pd.DataFrame, task: str):
        self.rows = split_rows.reset_index(drop=True).copy()
        self.task = task

        self.windows_by_record = []
        self.offsets = [0]

        for row in self.rows.itertuples(index=False):
            windows = np.load(
                row.windows_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            # mmap 배열을 Tensor로 안전하게 변환하기 위해 복사
            windows_tensor = torch.from_numpy(
                np.array(windows, dtype=np.float32, copy=True)
            )

            self.windows_by_record.append(windows_tensor)
            self.offsets.append(
                self.offsets[-1] + len(windows_tensor)
            )

        self.total_windows = self.offsets[-1]

    def __len__(self):
        return self.total_windows

    def __getitem__(self, index):
        if not 0 <= index < self.total_windows:
            raise IndexError(index)

        record_index = bisect_right(self.offsets, index) - 1
        window_index = index - self.offsets[record_index]

        row = self.rows.iloc[record_index]
        windows = self.windows_by_record[record_index]

        if self.task == "regression":
            target_value = float(row["wear"])
        else:
            target_value = float(row["target"])

        return {
            "sensor": windows[window_index],
            "target": torch.tensor(
                [target_value], dtype=torch.float32
            ),
            "record_index": record_index,
        }


def build_dataloaders(
    manifest: pd.DataFrame,
    task: str,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    loaders = {}
    datasets = {}

    generator = torch.Generator().manual_seed(seed)

    for split in ("train", "valid", "test"):
        split_rows = manifest.loc[
            manifest["split"].eq(split)
        ].reset_index(drop=True)

        dataset = ForceYWindowDataset(
            split_rows=split_rows,
            task=task,
        )
        datasets[split] = dataset

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if split == "train" else None,
            drop_last=False,
        )

    return datasets, loaders

# =========================================================
# 5. Model: Force Y 1D CNN
# =========================================================

class ForceY1DCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.sensor_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, sensor):
        features = self.sensor_encoder(sensor)
        output = self.head(features)
        return output


def build_criterion(task: str):
    if task == "regression":
        return nn.HuberLoss(delta=HUBER_DELTA)

    if task == "classification":
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"지원하지 않는 task: {task}")

# =========================================================
# 6. Metrics
# =========================================================

def regression_metrics(
    y_true,
    y_pred,
    huber_delta: float,
):
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

    total_variation = np.sum(
        (y_true - y_true.mean()) ** 2
    )

    return {
        "huber": float(huber_values.mean()),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "r2": (
            float(
                1.0 - squared_error.sum() / total_variation
            )
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
            if both_classes
            else float("nan")
        ),
        "ap_nonflank": (
            float(
                average_precision_score(
                    1 - y_true,
                    1 - probabilities,
                )
            )
            if both_classes
            else float("nan")
        ),
        "nonflank_correct": int(cm[0, 0]),
        "nonflank_total": int(cm[0].sum()),
        "flank_correct": int(cm[1, 1]),
        "flank_total": int(cm[1].sum()),
        "n": int(len(y_true)),
    }


# =========================================================
# 7. Train / Evaluation Engine
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

    for batch in loader:
        sensor = batch["sensor"].to(
            device, non_blocking=True
        )
        targets = batch["target"].to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(sensor)
        loss = criterion(outputs, targets)

        if not torch.isfinite(loss):
            raise ValueError("Training loss가 NaN 또는 Inf입니다.")

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("학습 DataLoader가 비어 있습니다.")

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

    window_loss_sum = 0.0
    window_count = 0

    record_outputs = [
        [] for _ in range(len(loader.dataset.rows))
    ]

    for batch in loader:
        sensor = batch["sensor"].to(
            device, non_blocking=True
        )
        targets = batch["target"].to(
            device, non_blocking=True
        )
        record_indices = batch["record_index"].tolist()

        outputs = model(sensor)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        window_loss_sum += loss.item() * batch_size
        window_count += batch_size

        output_values = outputs.squeeze(1).cpu().numpy()

        for record_index, output_value in zip(
            record_indices, output_values
        ):
            record_outputs[int(record_index)].append(
                float(output_value)
            )


    rows = loader.dataset.rows.copy()

    # 각 센서 기록의 윈도우 예측값 평균
    record_raw_outputs = np.asarray(
        [np.mean(values) for values in record_outputs],
        dtype=float,
    )

    window_mean_loss = window_loss_sum / window_count

    if task == "regression":
        y_true = rows["wear"].to_numpy(dtype=float)
        y_pred = record_raw_outputs

        metrics = regression_metrics(
            y_true,
            y_pred,
            huber_delta=HUBER_DELTA,
        )

        records = rows[
            ["Set", "SensorName", "wear", "type", "n_windows"]
        ].copy()
        records["prediction"] = y_pred
        records["error"] = records["prediction"] - records["wear"]
        records["abs_error"] = records["error"].abs()

        # 기록 단위 Huber를 validation loss로 사용
        selection_loss = metrics["huber"]

    else:
        y_true = rows["target"].to_numpy(dtype=int)

        probabilities = 1.0 / (
            1.0 + np.exp(
                -np.clip(record_raw_outputs, -80, 80)
            )
        )
        predictions = (
            probabilities >= classification_threshold
        ).astype(int)

        metrics = classification_metrics(
            y_true,
            probabilities,
            threshold=classification_threshold,
        )

        records = rows[
            [
                "Set",
                "SensorName",
                "type",
                "target",
                "n_windows",
            ]
        ].copy()
        records["true_label"] = y_true
        records["probability_flank"] = probabilities
        records["prediction_label"] = predictions

        selection_loss = window_mean_loss

    metrics["window_loss"] = float(window_mean_loss)
    metrics["n_records"] = int(len(rows))
    metrics["n_windows"] = int(window_count)

    return selection_loss, metrics, records

# =========================================================
# 8. Checkpoint
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
        "config": {
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "huber_delta": HUBER_DELTA,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "sensor_channel": "force_y",
            "input_channels": 1,
        },
    }

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


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

    if checkpoint.get("task") != expected_task:
        raise ValueError(
            f"Checkpoint task 불일치: "
            f"expected={expected_task}, "
            f"actual={checkpoint.get('task')}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint


# =========================================================
# 9. Trainer: Early Stopping
# =========================================================

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
    if task == "regression":
        best_metric = float("inf")
    else:
        best_metric = float("-inf")

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

        if (
            not np.isfinite(train_loss)
            or not np.isfinite(val_loss)
        ):
            raise ValueError(
                f"Epoch {epoch}: train/validation loss가 "
                "유효하지 않습니다."
            )

        # Regression: 기록 단위 Validation Huber 최소화
        # Classification: 기록 단위 Validation Macro F1 최대화
        if task == "regression":
            current_metric = val_metrics["huber"]
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
            "train_window_loss": train_loss,
            "val_selection_loss": val_loss,
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            },
            "learning_rate": optimizer.param_groups[0]["lr"],
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
                f"Train Huber(window) {train_loss:.3f} | "
                f"Val Huber(record) {val_metrics['huber']:.3f} | "
                f"Val MAE {val_metrics['mae']:.3f} µm | "
                f"Val RMSE {val_metrics['rmse']:.3f} µm | "
                f"미개선 {stale_epochs}/{PATIENCE} | "
                f"{elapsed:.1f}초",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Train BCE(window) {train_loss:.4f} | "
                f"Val Macro F1(record) "
                f"{val_metrics['macro_f1']:.4f} | "
                f"Val Acc {val_metrics['accuracy']:.4f} | "
                f"Val Recall Flank "
                f"{val_metrics['recall_flank']:.4f} | "
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
        print("Best Validation Huber:", best_metric)
    else:
        print("Best Validation Macro F1:", best_metric)

    print("Saved Model:", best_path)

    return history, best_epoch, best_metric

# =========================================================
# 10. Test results: Overall and Set-wise
# =========================================================

def save_test_results(
    test_records: pd.DataFrame,
    task: str,
    output_dir: Path,
):
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
        encoding="utf-8-sig",
    )
    metrics_df.to_csv(
        output_dir / "test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return metrics_df

# =========================================================
# 11. Main
# =========================================================

def main():
    set_seed(SEED)
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Manifest 로드 및 검증
    manifest = load_sensor_manifest(
        manifest_path=MANIFEST_PATH,
        task=TASK,
        window_size=WINDOW_SIZE,
    )

    # 2) Dataset / DataLoader
    datasets, loaders = build_dataloaders(
        manifest=manifest,
        task=TASK,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )

    # 3) Batch shape 확인
    check_loader = loaders["train"]
    check_batch = next(iter(check_loader))

    assert check_batch["sensor"].shape[1:] == (
        1,
        WINDOW_SIZE,
    )
    assert check_batch["target"].shape == (
        len(check_batch["sensor"]),
        1,
    )
    assert torch.isfinite(check_batch["sensor"]).all().item()
    assert torch.isfinite(check_batch["target"]).all().item()

    if TASK == "classification":
        assert (
            (check_batch["target"] == 0)
            | (check_batch["target"] == 1)
        ).all().item()


    # 4) Model / Loss / Optimizer / Scheduler
    model = ForceY1DCNN(dropout=DROPOUT).to(device)
    criterion = build_criterion(TASK)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
    )

    # 모델 출력 shape 사전 검증
    model.eval()
    with torch.inference_mode():
        sample_input = check_batch["sensor"][:2].to(device)
        sample_output = model(sample_input)

    assert sample_output.shape == (2, 1)

    # 5) Training
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

    # 6) Best checkpoint 로드
    checkpoint = load_best_checkpoint(
        model=model,
        checkpoint_path=BEST_PATH,
        device=device,
        expected_task=TASK,
    )

    print("\n========== Best Checkpoint Loaded ==========")
    print("Best Epoch:", checkpoint["epoch"])
    print("Best Validation Loss:", checkpoint["val_loss"])

    # 7) Test 평가
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
    print("Test Selection Loss:", test_loss)
    print(pd.DataFrame(
        [test_metrics],
        index=["Test Overall"],
    ).to_string())

    # 8) Overall 및 Set별 결과 저장
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
