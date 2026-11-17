from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d


# =========================================================
# 1. Configuration
# =========================================================

ROOT_DIR = Path("/home/ubuntu/datasets")

# Image preprocessing output
IMAGE_DIR_NAME = "image_inte_3"

# Intermediate output: cutting-segment waveforms and manifest
CUT_OUTPUT_DIR_NAME = "sensor_cut_type_split"

# Final output: Hampel + Moving RMS + windows + normalization
SENSOR_OUTPUT_DIR_NAME = "sensor_force_y_hampel21_sigma4_rms100"

TYPE_ORDER = [
    "flank_wear",
    "adhesion",
    "flank_wear+adhesion",
]

# Raw sensor CSV columns
SENSOR_COLUMNS = [
    "acc",
    "audio",
    "force_x",
    "force_y",
    "force_z",
    "timestamp",
]

# Channels saved in the cutting waveform NPY
WAVEFORM_COLUMNS = [
    "acc",
    "audio",
    "force_x",
    "force_y",
]

# Cutting segment extraction
CUT_WINDOW_MS = 50
CUT_MIN_DURATION_MS = 100

# Hampel filter
SENSOR_CHANNEL = "force_y"
SENSOR_CHANNEL_INDEX = 3
HAMPEL_WINDOW_SIZE = 21
HAMPEL_N_SIGMAS = 4

# Moving RMS
RMS_WINDOW_MS = 100

# Windowing
WINDOW_SIZE = 512
WINDOW_OVERLAP = 0

# Output filenames
CUT_MANIFEST_NAME = "manifest_multimodal_type.csv"
PROCESSED_MANIFEST_NAME = "manifest_sensor_preprocessed.csv"
NORMALIZATION_NAME = "normalization.json"


# =========================================================
# 2. Cutting segment extraction
# =========================================================

def extract_cutting_segment(
    df: pd.DataFrame,
    fs: float,
    window_ms: float = CUT_WINDOW_MS,
    min_duration_ms: float = CUT_MIN_DURATION_MS,
) -> tuple[pd.DataFrame, int, int]:
    """
    Extract the longest cutting segment using combined Fx/Fy moving RMS.

    Hysteresis thresholds:
        Entry: noise median + 10 * robust sigma
        Exit : noise median +  6 * robust sigma
    """
    if len(df) == 0:
        raise ValueError("센서 데이터가 비어 있습니다.")

    fx = df["force_x"].to_numpy(dtype=np.float64)
    fy = df["force_y"].to_numpy(dtype=np.float64)

    if not np.isfinite(fx).all() or not np.isfinite(fy).all():
        raise ValueError("Force X/Y에 NaN 또는 무한대 값이 포함되어 있습니다.")

    # 1. Estimate baseline from the first 10% of the signal.
    baseline_n = max(1, int(len(df) * 0.1))
    base_x = np.median(fx[:baseline_n])
    base_y = np.median(fy[:baseline_n])

    fx = fx - base_x
    fy = fy - base_y

    # 2. Calculate channel-wise moving RMS and combine them.
    window = max(1, int(round(fs * window_ms / 1000)))

    rms_x = np.sqrt(
        uniform_filter1d(fx ** 2, size=window, mode="reflect")
    )
    rms_y = np.sqrt(
        uniform_filter1d(fy ** 2, size=window, mode="reflect")
    )
    energy = np.sqrt(rms_x ** 2 + rms_y ** 2)

    # 3. Estimate noise level from the lowest 20% of energy values.
    baseline_energy = energy[
        energy <= np.percentile(energy, 20)
    ]

    noise_median = float(np.median(baseline_energy))
    mad = float(
        np.median(np.abs(baseline_energy - noise_median))
    )
    noise_std = 1.4826 * mad

    low_threshold = noise_median + 6 * noise_std
    high_threshold = noise_median + 10 * noise_std

    # 4. Hysteresis thresholding.
    active = False
    mask = np.zeros(len(df), dtype=bool)

    for i, value in enumerate(energy):
        if not active and value > high_threshold:
            active = True
        elif active and value < low_threshold:
            active = False

        mask[i] = active

    # 5. Find continuous active segments.
    changes = np.diff(
        np.r_[False, mask, False].astype(np.int8)
    )
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    min_samples = max(
        1,
        int(round(fs * min_duration_ms / 1000)),
    )

    valid_segments = [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= min_samples
    ]

    if not valid_segments:
        raise ValueError(
            "최소 절삭구간 길이를 만족하는 구간을 찾지 못했습니다."
        )

    start_idx, end_idx = max(
        valid_segments,
        key=lambda segment: segment[1] - segment[0],
    )

    cutting_df = df.iloc[start_idx:end_idx].copy()

    return cutting_df, start_idx, end_idx


def read_and_cut_sensor(
    csv_path: Path,
) -> tuple[pd.DataFrame, float, int, int]:
    """Read a raw sensor CSV and extract its cutting segment."""
    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(
            f"센서 CSV 열 개수가 올바르지 않습니다. "
            f"expected={len(SENSOR_COLUMNS)}, actual={df.shape[1]}, "
            f"path={csv_path}"
        )

    df.columns = SENSOR_COLUMNS

    numeric_columns = SENSOR_COLUMNS[:-1]
    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if df[numeric_columns].isna().any().any():
        raise ValueError("센서 수치 데이터에 NaN이 포함되어 있습니다.")

    timestamp = pd.to_datetime(
        df["timestamp"],
        format="mixed",
        errors="coerce",
    )

    intervals_sec = (
        timestamp.diff()
        .dt.total_seconds()
        .iloc[1:]
        .to_numpy(dtype=np.float64)
    )
    intervals_sec = intervals_sec[
        np.isfinite(intervals_sec) & (intervals_sec > 0)
    ]

    if intervals_sec.size == 0:
        raise ValueError("유효한 timestamp 간격을 계산할 수 없습니다.")

    median_dt_sec = float(np.median(intervals_sec))
    fs_hz = 1.0 / median_dt_sec

    cutting_df, start_idx, end_idx = extract_cutting_segment(
        df,
        fs=fs_hz,
    )

    return cutting_df, fs_hz, start_idx, end_idx


# =========================================================
# 3. Image-label-sensor matching
# =========================================================

def build_paired_metadata(
    root_dir: Path,
    image_dir: Path,
) -> pd.DataFrame:

    labels_path = root_dir / "labels.csv"
    metadata_path = image_dir / "metadata.csv"

    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv를 찾을 수 없습니다: {labels_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"이미지 metadata.csv를 찾을 수 없습니다: {metadata_path}. "
            "이미지 전처리를 먼저 실행해야 합니다."
        )

    labels = pd.read_csv(
        labels_path,
        keep_default_na=False,
    )
    metadata = pd.read_csv(metadata_path)

    required_label_columns = {
        "Set", "ImageName", "wear", "type",
        "SensorName", "SensorFile",
    }
    required_metadata_columns = {
        "Set", "split", "filename", "original_filename",
    }

    missing_labels = required_label_columns - set(labels.columns)
    missing_metadata = required_metadata_columns - set(metadata.columns)

    if missing_labels:
        raise ValueError(
            f"labels.csv 필수 컬럼 누락: {sorted(missing_labels)}"
        )
    if missing_metadata:
        raise ValueError(
            f"image metadata 필수 컬럼 누락: {sorted(missing_metadata)}"
        )

    labels["type"] = (
        labels["type"].astype(str).str.replace(r"\s+", "", regex=True)
    )
    labels["Set"] = pd.to_numeric(labels["Set"], errors="coerce")
    labels["wear"] = pd.to_numeric(labels["wear"], errors="coerce")

    labels["original_filename"] = (
        labels["ImageName"]
        .astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.rsplit("/", n=1)
        .str[-1]
    )

    # Match image metadata with wear type.
    type_labels = (
        labels.loc[
            labels["type"].isin(TYPE_ORDER),
            ["Set", "original_filename", "type"],
        ]
        .drop_duplicates()
    )

    data = metadata.drop(
    columns=["type"],
    errors="ignore",
    ).copy()
    data["Set"] = pd.to_numeric(data["Set"], errors="coerce")

    data = data.merge(
        type_labels,
        on=["Set", "original_filename"],
        how="left",
        validate="one_to_one",
    )

    # Binary classification target retained from the original notebook:
    # flank_wear = 1, adhesion and mixed type = 0.
    data["target"] = data["type"].eq("flank_wear").astype("int64")

    # Match sensor records to image records.
    for column in ["SensorName", "SensorFile"]:
        labels[column] = labels[column].astype(str).str.strip()

    sensor_basename = (
        labels["SensorFile"]
        .str.replace("\\", "/", regex=False)
        .str.rsplit("/", n=1)
        .str[-1]
    )

    sensor_labels = labels.loc[
        labels["SensorName"].ne("")
        & labels["SensorName"].eq(sensor_basename)
        & np.isfinite(labels["wear"]),
        [
            "Set", "original_filename", "type", "wear",
            "SensorName", "SensorFile",
        ],
    ].drop_duplicates()

    keys = data[["Set", "original_filename"]].drop_duplicates()

    sensor_labels = sensor_labels.merge(
        keys,
        on=["Set", "original_filename"],
        how="inner",
        validate="many_to_one",
    )

    sensor_labels = sensor_labels.rename(
        columns={
            "wear": "label_wear",
            "type": "label_type",
        }
    )

    paired_data = data.merge(
        sensor_labels,
        on=["Set", "original_filename"],
        how="left",
        validate="one_to_one",
    )

    paired_data["image_path"] = [
        image_dir / str(split) / str(filename)
        for split, filename in zip(
            paired_data["split"],
            paired_data["filename"],
        )
    ]

    paired_data["sensor_path"] = [
        root_dir
        / "set"
        / f"Set{int(set_no)}"
        / "sensordata"
        / str(sensor_name)

        if pd.notna(set_no) and str(sensor_name).strip()
        else Path("__missing_sensor_file__")
        for set_no, sensor_name in zip(
            paired_data["Set"],
            paired_data["SensorName"],
        )
    ]

    paired_data["label_ok"] = (
        paired_data["type"].eq(paired_data["label_type"])
        & np.isclose(
            pd.to_numeric(paired_data["wear"], errors="coerce"),
            paired_data["label_wear"],
            rtol=0,
            atol=1e-9,
            equal_nan=False,
        )
    )

    paired_data["image_ok"] = paired_data["image_path"].map(
        lambda path: Path(path).is_file()
    )
    paired_data["sensor_ok"] = paired_data["sensor_path"].map(
        lambda path: Path(path).is_file()
        and Path(path).stat().st_size > 0
    )

    return paired_data


# =========================================================
# 4. Generate cutting waveform NPY files and manifest
# =========================================================

def generate_cutting_waveforms(
    paired_data: pd.DataFrame,
    root_dir: Path,
    output_dir: Path,
) -> Path:
    """Extract cutting segments from raw sensor CSV files and save NPY."""
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for number, row in enumerate(
        paired_data.itertuples(index=False),
        start=1,
    ):
        result: dict[str, Any] = {
            "Set": int(row.Set) if pd.notna(row.Set) else np.nan,
            "split": row.split,
            "original_filename": row.original_filename,
            "SensorName": row.SensorName,
            "image_path": str(row.image_path),
            "sensor_path": str(row.sensor_path),
            "wear": row.wear,
            "type": row.type,
            "target": int(row.target),
            "cut_method": "auto_fx_fy_rms",
            "waveform_path": "",
            "window_count": 0,
            "cut_error": "",
        }

        try:
            if not row.sensor_ok:
                raise FileNotFoundError(
                    f"센서 파일이 없거나 비어 있습니다: {row.sensor_path}"
                )

            cutting_df, fs_hz, start_idx, end_idx = read_and_cut_sensor(
                Path(row.sensor_path)
            )

            waveform = cutting_df[WAVEFORM_COLUMNS].to_numpy(
                dtype=np.float64
            )

            set_dir = (
                output_dir
                / "waveforms"
                / f"Set{int(row.Set)}"
            )
            set_dir.mkdir(parents=True, exist_ok=True)

            waveform_path = (
                set_dir / f"{Path(str(row.SensorName)).stem}.npy"
            )
            temp_path = waveform_path.with_suffix(".tmp.npy")

            np.save(
                temp_path,
                waveform,
                allow_pickle=False,
            )
            temp_path.replace(waveform_path)

            result.update({
                "fs_hz": fs_hz,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "selected_samples": len(waveform),
                "duration_sec": len(waveform) / fs_hz,
                "window_count": len(waveform) // WINDOW_SIZE,
                "waveform_path": str(waveform_path),
                "cut_status": "detected",
                "cut_source": "generated",
            })

        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update({
                "cut_status": "failed",
                "cut_source": "failed",
                "cut_error": str(error),
            })

        records.append(result)

        if number % 100 == 0 or number == len(paired_data):
            print(
                f"[절삭구간 추출] {number:,}/{len(paired_data):,}",
                flush=True,
            )

    manifest = pd.DataFrame(records)

    manifest_path = output_dir / CUT_MANIFEST_NAME
    temp_manifest_path = output_dir / (
        Path(CUT_MANIFEST_NAME).stem + ".tmp.csv"
    )

    manifest.to_csv(
        temp_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )
    temp_manifest_path.replace(manifest_path)

    print("\n[절삭구간 추출 결과]")
    if not manifest.empty:
        print(
            pd.crosstab(
                manifest["split"],
                manifest["cut_source"],
                margins=True,
            )
        )

        failed = manifest.loc[
            manifest["cut_status"].ne("detected")
            | manifest["window_count"].eq(0)
        ]
        print(f"절삭 실패 또는 512 미만 샘플 파일 수: {len(failed):,}")

    print(f"절삭구간 Manifest: {manifest_path}")

    return manifest_path


# =========================================================
# 5. Hampel filter and moving RMS
# =========================================================

def hampel_filter(
    signal: np.ndarray,
    window_size: int = HAMPEL_WINDOW_SIZE,
    n_sigmas: float = HAMPEL_N_SIGMAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a local median/MAD-based Hampel filter."""
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Hampel 입력은 1차원 배열이어야 합니다.")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("Hampel window_size는 양의 홀수여야 합니다.")
    if not np.isfinite(signal).all():
        raise ValueError("Hampel 입력에 NaN 또는 무한대가 포함되어 있습니다.")

    half_window = window_size // 2
    padded = np.pad(
        signal,
        (half_window, half_window),
        mode="reflect",
    )
    windows = sliding_window_view(padded, window_size)

    local_median = np.median(windows, axis=1)
    deviations = np.abs(windows - local_median[:, None])
    local_mad = np.median(deviations, axis=1)

    robust_sigma = 1.4826 * local_mad
    threshold = n_sigmas * robust_sigma

    outlier_mask = (
        np.abs(signal - local_median) > threshold
    )

    filtered = signal.copy()
    filtered[outlier_mask] = local_median[outlier_mask]

    return filtered, outlier_mask


def preprocess_sensor_record(
    npy_path: Path,
    fs_hz: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Force Y extraction -> Hampel filtering -> moving RMS."""
    waveform = np.load(
        npy_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if waveform.ndim != 2 or waveform.shape[1] <= SENSOR_CHANNEL_INDEX:
        raise ValueError(
            f"Waveform shape이 올바르지 않습니다: "
            f"{waveform.shape}, path={npy_path}"
        )

    raw = np.asarray(
        waveform[:, SENSOR_CHANNEL_INDEX],
        dtype=np.float64,
    )

    filtered, outlier_mask = hampel_filter(
        raw,
        window_size=HAMPEL_WINDOW_SIZE,
        n_sigmas=HAMPEL_N_SIGMAS,
    )

    rms_samples = max(
        1,
        int(round(fs_hz * RMS_WINDOW_MS / 1000)),
    )

    mean_square = uniform_filter1d(
        filtered ** 2,
        size=rms_samples,
        mode="reflect",
    )
    rms = np.sqrt(np.maximum(mean_square, 0.0))

    return rms, outlier_mask, rms_samples


# =========================================================
# 6. Running statistics for train-only normalization
# =========================================================

def update_running_stats(
    count: int,
    mean: float,
    m2: float,
    values: np.ndarray,
) -> tuple[int, float, float]:
    """Merge batch statistics into running count, mean, and M2."""
    values = np.asarray(values, dtype=np.float64)
    batch_count = values.size

    if batch_count == 0:
        return count, mean, m2

    batch_mean = float(values.mean())
    batch_m2 = float(np.sum((values - batch_mean) ** 2))

    if count == 0:
        return batch_count, batch_mean, batch_m2

    total_count = count + batch_count
    delta = batch_mean - mean

    total_mean = mean + delta * batch_count / total_count
    total_m2 = (
        m2
        + batch_m2
        + delta ** 2 * count * batch_count / total_count
    )

    return total_count, total_mean, total_m2


# =========================================================
# 7. Windowing, normalization, and output
# =========================================================

def create_windows(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    overlap: int = WINDOW_OVERLAP,
) -> np.ndarray:
    """Return windows with shape (n_windows, 1, window_size)."""
    if window_size <= 0:
        raise ValueError("WINDOW_SIZE는 양수여야 합니다.")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("WINDOW_OVERLAP은 0 이상 WINDOW_SIZE 미만이어야 합니다.")

    step = window_size - overlap

    if len(signal) < window_size:
        raise ValueError(
            f"신호 길이({len(signal)})가 WINDOW_SIZE({window_size})보다 짧습니다."
        )

    windows_2d = sliding_window_view(
        np.asarray(signal),
        window_size,
    )[::step]

    return windows_2d[:, None, :]


def preprocess_all_sensor_data(
    manifest_path: Path,
    output_dir: Path,
) -> Path:
    """Run Hampel, moving RMS, windowing, and train-only Z-score."""
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"절삭구간 Manifest를 찾을 수 없습니다: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)
    manifest = manifest.loc[
        manifest["cut_status"].eq("detected")
    ].copy()

    if manifest.empty:
        raise ValueError("전처리할 절삭 성공 데이터가 없습니다.")

    manifest["fs_hz"] = pd.to_numeric(
        manifest["fs_hz"],
        errors="raise",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "_temporary"
    temp_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    train_mean = 0.0
    train_m2 = 0.0

    records: list[dict[str, Any]] = []

    try:
        # -------------------------------------------------
        # Pass 1: Hampel + Moving RMS + windowing.
        # Fit normalization statistics using train only.
        # -------------------------------------------------
        for position, row in enumerate(
            manifest.itertuples(index=False),
            start=1,
        ):
            waveform_path = Path(str(row.waveform_path))
            if not waveform_path.is_absolute():
                waveform_path = ROOT_DIR / waveform_path

            fs_hz = float(row.fs_hz)

            rms, outlier_mask, rms_samples = preprocess_sensor_record(
                waveform_path,
                fs_hz,
            )

            windows = create_windows(
                rms,
                window_size=WINDOW_SIZE,
                overlap=WINDOW_OVERLAP,
            )

            usable_length = (
                WINDOW_SIZE
                + (len(windows) - 1)
                * (WINDOW_SIZE - WINDOW_OVERLAP)
            )
            usable = rms[:usable_length]

            temp_path = temp_dir / f"{position:05d}.npy"
            np.save(
                temp_path,
                windows.astype(np.float64),
                allow_pickle=False,
            )

            if row.split == "train":
                (
                    train_count,
                    train_mean,
                    train_m2,
                ) = update_running_stats(
                    train_count,
                    train_mean,
                    train_m2,
                    usable,
                )

            records.append({
                "source_index": position - 1,
                "temp_path": str(temp_path),
                "n_windows": len(windows),
                "cut_samples": len(rms),
                "hampel_replaced": int(outlier_mask.sum()),
                "rms_window_samples": rms_samples,
            })

            if position % 100 == 0 or position == len(manifest):
                print(
                    f"[센서 신호 처리] {position:,}/{len(manifest):,}",
                    flush=True,
                )

        if train_count == 0:
            raise ValueError(
                "Train 데이터가 없어 정규화 통계를 계산할 수 없습니다."
            )

        sensor_mean = float(train_mean)
        sensor_std = float(np.sqrt(train_m2 / train_count))

        if sensor_std <= 0 or not np.isfinite(sensor_std):
            raise ValueError(
                f"Train 표준편차가 유효하지 않습니다: {sensor_std}"
            )

        # -------------------------------------------------
        # Pass 2: Normalize all windows using train stats.
        # -------------------------------------------------
        window_paths: list[str] = []

        for record_info in records:
            source_row = manifest.iloc[
                record_info["source_index"]
            ]

            temp_path = Path(record_info["temp_path"])
            windows = np.load(
                temp_path,
                allow_pickle=False,
            )

            normalized = (
                (windows - sensor_mean) / sensor_std
            ).astype(np.float32)

            split_dir = (
                output_dir
                / "windows"
                / str(source_row["split"])
                / f"Set{int(source_row['Set'])}"
            )
            split_dir.mkdir(parents=True, exist_ok=True)

            output_name = (
                f"{Path(str(source_row['SensorName'])).stem}.npy"
            )
            output_path = split_dir / output_name

            np.save(
                output_path,
                normalized,
                allow_pickle=False,
            )
            window_paths.append(str(output_path))

        # -------------------------------------------------
        # Save processed manifest.
        # -------------------------------------------------
        processed_manifest = manifest.reset_index(drop=True).copy()

        processed_manifest["windows_path"] = window_paths
        processed_manifest["n_windows"] = [
            item["n_windows"] for item in records
        ]
        processed_manifest["cut_samples"] = [
            item["cut_samples"] for item in records
        ]
        processed_manifest["hampel_replaced"] = [
            item["hampel_replaced"] for item in records
        ]
        processed_manifest["rms_window_samples"] = [
            item["rms_window_samples"] for item in records
        ]

        manifest_output_path = (
            output_dir / PROCESSED_MANIFEST_NAME
        )
        processed_manifest.to_csv(
            manifest_output_path,
            index=False,
            encoding="utf-8-sig",
        )

        # -------------------------------------------------
        # Save normalization configuration and statistics.
        # -------------------------------------------------
        normalization = {
            "sensor_channel": SENSOR_CHANNEL,
            "sensor_channel_index": SENSOR_CHANNEL_INDEX,
            "hampel_window_size": HAMPEL_WINDOW_SIZE,
            "hampel_n_sigmas": HAMPEL_N_SIGMAS,
            "rms_window_ms": RMS_WINDOW_MS,
            "window_size": WINDOW_SIZE,
            "window_overlap": WINDOW_OVERLAP,
            "normalization_method": "z_score",
            "normalization_fit_split": "train",
            "sensor_mean": sensor_mean,
            "sensor_std": sensor_std,
            "train_sample_count": train_count,
        }

        normalization_path = output_dir / NORMALIZATION_NAME
        with normalization_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                normalization,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print("\n[센서 전처리 완료]")
        print(f"Window 저장 폴더: {output_dir / 'windows'}")
        print(f"Manifest: {manifest_output_path}")
        print(f"Normalization: {normalization_path}")
        print(f"Train 평균: {sensor_mean:.8f}")
        print(f"Train 표준편차: {sensor_std:.8f}")
        print(f"Train sample 수: {train_count:,}")

        return manifest_output_path

    finally:
        # Remove temporary window files even if a later step fails.
        if temp_dir.exists():
            for temp_path in temp_dir.glob("*.npy"):
                temp_path.unlink()
            temp_dir.rmdir()


# =========================================================
# 8. Main entry point
# =========================================================

def run_sensor_preprocessing(
    root_dir: str | Path = ROOT_DIR,
    image_dir_name: str = IMAGE_DIR_NAME,
) -> Path:
    root_dir = Path(root_dir)
    image_dir = root_dir / image_dir_name

    cut_output_dir = (
        root_dir / "outputs" / CUT_OUTPUT_DIR_NAME
    )
    sensor_output_dir = (
        root_dir / "outputs" / SENSOR_OUTPUT_DIR_NAME
    )

    print("=" * 60)
    print("Sensor preprocessing started")
    print(f"Dataset root : {root_dir}")
    print(f"Image dir    : {image_dir}")
    print(f"Cut output   : {cut_output_dir}")
    print(f"Sensor output: {sensor_output_dir}")
    print("=" * 60)

    # 1. Match images, labels, and raw sensor files.
    paired_data = build_paired_metadata(
        root_dir=root_dir,
        image_dir=image_dir,
    )

    print("\n[데이터 매칭 결과]")
    print(f"이미지 metadata 수: {len(paired_data):,}")
    print(f"이미지 파일 존재: {int(paired_data['image_ok'].sum()):,}")
    print(f"센서 파일 존재: {int(paired_data['sensor_ok'].sum()):,}")
    print(f"이미지-라벨 일치: {int(paired_data['label_ok'].sum()):,}")

    # 2. Extract cutting segments and save waveform NPY files.
    cut_manifest_path = generate_cutting_waveforms(
        paired_data=paired_data,
        root_dir=root_dir,
        output_dir=cut_output_dir,
    )

    # 3. Hampel + moving RMS + windowing + normalization.
    processed_manifest_path = preprocess_all_sensor_data(
        manifest_path=cut_manifest_path,
        output_dir=sensor_output_dir,
    )

    print("\nSensor preprocessing finished successfully.")

    return processed_manifest_path


if __name__ == "__main__":
    run_sensor_preprocessing()
