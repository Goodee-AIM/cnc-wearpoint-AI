from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# =========================================================
# 1. Configuration
# =========================================================

ROOT = Path("/home/ubuntu/datasets")  # 팀원 환경에서는 이 경로만 변경
RAW = ROOT / "set"
OUT = ROOT / "image_inte_3"

SPLITS = {
    "train": (1, 5, 7, 8, 10, 12),
    "valid": (6, 13, 15),
    "test": (4, 9, 11, 16),
}

# Set number -> split name
SPLIT_BY_SET = {
    n: split
    for split, numbers in SPLITS.items()
    for n in numbers
}

# Existing manually excluded image
MANUAL_EXCLUSIONS = {
    (1, "File_name_2022-09-12T13_00_26.578508.jpg"): "black_frame",
}

METADATA_COLUMNS = [
    "index",
    "filename",
    "original_filename",
    "Set",
    "wear",
    "type",
    "split",
]

AUDIT_COLUMNS = [
    "Set",
    "ImageName",
    "ExpectedFolder",
    "Reason",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# =========================================================
# 2. Helper functions
# =========================================================

def basename(values: pd.Series) -> pd.Series:
    """Return the basename of paths that may contain either slash style."""
    return (
        values.astype(str)
        .str.replace("\\", "/", regex=False)
        .str.rsplit("/", n=1)
        .str[-1]
    )


def load_crop_boxes(root: Path) -> dict[int, tuple[int, int, int, int]]:
    """Load per-Set crop coordinates from sets.csv."""
    sets_path = root / "sets.csv"

    if not sets_path.is_file():
        raise FileNotFoundError(sets_path)

    sets = pd.read_csv(sets_path, index_col=0)

    crop_boxes = {}

    for n in SPLIT_BY_SET:
        row_name = f"Set {n}"

        if row_name not in sets.index:
            raise KeyError(
                f"sets.csv에 '{row_name}' 행이 없습니다."
            )

        crop_value = sets.loc[row_name, "crop"]

        if pd.isna(crop_value):
            raise ValueError(
                f"sets.csv의 '{row_name}' crop 값이 비어 있습니다."
            )

        values = tuple(
            int(value.strip())
            for value in str(crop_value).split(",")
        )

        if len(values) != 4:
            raise ValueError(
                f"'{row_name}' crop은 x,y,w,h 또는 "
                f"(left, top, right, bottom) 4개 값이어야 합니다: "
                f"{crop_value}"
            )

        crop_boxes[n] = values

    return crop_boxes


def load_and_validate_labels(root: Path) -> pd.DataFrame:
    """Load labels.csv and validate image/sensor filename pairs."""
    labels_path = root / "labels.csv"

    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)

    labels = pd.read_csv(
        labels_path,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "Set",
        "ImageName",
        "ImageFile",
        "SensorName",
        "SensorFile",
        "wear",
        "type",
    }

    missing_columns = required_columns - set(labels.columns)

    if missing_columns:
        raise ValueError(
            f"labels.csv 필수 컬럼 누락: {sorted(missing_columns)}"
        )

    labels["Set"] = pd.to_numeric(
        labels["Set"]
        .str.strip()
        .str.replace(r"(?i)^set\s*", "", regex=True),
        errors="coerce",
    )

    labels = labels.loc[
        labels["Set"].isin(SPLIT_BY_SET)
    ].copy()

    labels["Set"] = labels["Set"].astype(int)

    fields = [
        "ImageName",
        "ImageFile",
        "SensorName",
        "SensorFile",
        "type",
    ]

    for field in fields:
        labels[field] = labels[field].str.strip()

    labels["wear"] = pd.to_numeric(
        labels["wear"],
        errors="coerce",
    )

    valid = (
        labels[fields].ne("").all(axis=1)
        & np.isfinite(labels["wear"])
        & labels["ImageName"].eq(basename(labels["ImageFile"]))
        & labels["SensorName"].eq(basename(labels["SensorFile"]))
    )

    pairs = labels.loc[
        valid,
        ["Set", "ImageName", "SensorName", "wear", "type"],
    ].drop_duplicates()

    if (
        pairs.duplicated(["Set", "ImageName"]).any()
        or pairs.duplicated(["Set", "SensorName"]).any()
    ):
        raise ValueError(
            "같은 세트 안에 이미지 또는 센서가 중복 매칭됐습니다."
        )

    return labels, pairs


# =========================================================
# 3. Main preprocessing
# =========================================================

def run_image_preprocessing(
    root: str | Path = ROOT,
    output_dir_name: str = "image_inte_3",
) -> Path:
    
    root = Path(root)
    raw_dir = root / "set"
    output_dir = root / output_dir_name

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw Set 폴더를 찾을 수 없습니다: {raw_dir}"
        )

    crop_boxes = load_crop_boxes(root)

    # Validate expected raw folders.
    for n in SPLIT_BY_SET:
        for folder in ("images", "sensordata"):
            path = raw_dir / f"Set{n}" / folder

            if not path.is_dir():
                raise FileNotFoundError(path)

    labels, pairs = load_and_validate_labels(root)

    pairs_by_image = {
        (row.Set, row.ImageName): row
        for row in pairs.itertuples(index=False)
    }

    # Avoid silently mixing results from previous runs.
    if output_dir.exists():
        raise FileExistsError(
            f"출력 폴더가 이미 존재합니다: {output_dir}\n"
            "기존 결과를 보존하기 위해 자동 삭제하지 않았습니다. "
            "폴더명을 변경하거나 기존 폴더를 직접 정리한 뒤 실행하세요."
        )

    output_dir.mkdir(parents=True)

    for split in SPLITS:
        (output_dir / split).mkdir()

    metadata_rows = []
    excluded_rows = []

    print("=" * 60)
    print("Image preprocessing started")
    print(f"Raw directory   : {raw_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    for n in sorted(SPLIT_BY_SET):
        split = SPLIT_BY_SET[n]
        set_dir = raw_dir / f"Set{n}"
        image_dir = set_dir / "images"

        left, top, right, bottom = crop_boxes[n]

        original_names = {
            path.name
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        label_names = (
            set(
                labels.loc[
                    labels["Set"].eq(n),
                    "ImageName",
                ]
            )
            - {""}
        )

        for name in sorted(original_names | label_names):
            key = (n, name)
            source = image_dir / name
            pair = pairs_by_image.get(key)
            reason = MANUAL_EXCLUSIONS.get(key)

            if reason is None:
                if name not in original_names:
                    reason = "missing_image"

                elif pair is None:
                    reason = "no_valid_sensor_pair"

                else:
                    sensor_path = (
                        set_dir / "sensordata" / pair.SensorName
                    )

                    if (
                        not sensor_path.is_file()
                        or sensor_path.stat().st_size == 0
                    ):
                        reason = "missing_or_empty_sensor"

            if reason is None:
                try:
                    with Image.open(source) as image:
                        if not (
                            0 <= left < right <= image.width
                            and 0 <= top < bottom <= image.height
                        ):
                            reason = "crop_outside_image"

                        else:
                            crop = image.crop(
                                crop_boxes[n]
                            ).convert("RGB")

                            if all(
                                low == high
                                for low, high in crop.getextrema()
                            ):
                                reason = "uniform_image"

                except OSError:
                    reason = "unreadable_image"

            if reason is not None:
                excluded_rows.append(
                    (n, name, str(image_dir), reason)
                )
                continue

            index = len(metadata_rows) + 1
            filename = f"{index:05d}_set{n}.png"

            crop.save(
                output_dir / split / filename
            )

            metadata_rows.append(
                (
                    index,
                    filename,
                    name,
                    n,
                    pair.wear,
                    pair.type,
                    split,
                )
            )

        print(
            f"[Set{n}] 완료 | "
            f"누적 저장 이미지: {len(metadata_rows):,}"
        )

    # Save metadata and exclusion audit.
    metadata = pd.DataFrame(
        metadata_rows,
        columns=METADATA_COLUMNS,
    )

    metadata.to_csv(
        output_dir / "metadata.csv",
        index=False,
    )

    audit = pd.DataFrame(
        excluded_rows,
        columns=AUDIT_COLUMNS,
    )

    audit.to_csv(
        output_dir / "missing_images.csv",
        index=False,
    )

    print("\n[Image preprocessing completed]")
    print(f"Output directory: {output_dir}")
    print(f"Saved images    : {len(metadata):,}")
    print(f"Excluded images : {len(audit):,}")

    if not metadata.empty:
        print("\n[Split counts]")
        print(metadata["split"].value_counts().reindex(SPLITS.keys(), fill_value=0))

    if not audit.empty:
        print("\n[Exclusion reasons]")
        print(audit["Reason"].value_counts())

    return output_dir


# =========================================================
# 4. Script entry point
# =========================================================

if __name__ == "__main__":
    run_image_preprocessing()