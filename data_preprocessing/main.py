from pathlib import Path
import sys
import traceback

from image_preprocessing import run_image_preprocessing
from sensor_preprocessing import run_sensor_preprocessing


# =========================================================
# 1. Configuration
# =========================================================

# 팀원 환경에서는 이 경로만 변경
ROOT_DIR = Path("/home/ubuntu/datasets")

# image_preprocessing.py output folder name
IMAGE_OUTPUT_DIR_NAME = "image_inte_3"


# =========================================================
# 2. Main pipeline
# =========================================================

def main() -> None:
    print("=" * 70)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Dataset root: {ROOT_DIR}")
    print()

    # -----------------------------------------------------
    # Preflight checks
    # -----------------------------------------------------
    required_paths = [
        ROOT_DIR / "set",
        ROOT_DIR / "sets.csv",
        ROOT_DIR / "labels.csv",
    ]

    missing_paths = [
        path for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        message = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "필수 데이터 경로를 찾을 수 없습니다:\n" + message
        )

    # -----------------------------------------------------
    # Step 1. Image preprocessing
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("[STEP 1/2] IMAGE PREPROCESSING")
    print("=" * 70)
    image_output_dir = run_image_preprocessing(
        root=ROOT_DIR,
        output_dir_name=IMAGE_OUTPUT_DIR_NAME,
    )

    image_metadata_path = image_output_dir / "metadata.csv"

    if not image_metadata_path.is_file():
        raise FileNotFoundError(
            "이미지 전처리 결과 metadata.csv를 찾을 수 없습니다: "
            f"{image_metadata_path}"
        )

    print("\n[STEP 1 COMPLETED]")
    print(f"Image output: {image_output_dir}")
    print(f"Image metadata: {image_metadata_path}")

    # -----------------------------------------------------
    # Step 2. Sensor preprocessing
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("[STEP 2/2] SENSOR PREPROCESSING")
    print("=" * 70)

    sensor_manifest_path = run_sensor_preprocessing(
        root_dir=ROOT_DIR,
        image_dir_name=image_output_dir.name,
    )

    if not sensor_manifest_path.is_file():
        raise FileNotFoundError(
            "센서 전처리 결과 Manifest를 찾을 수 없습니다: "
            f"{sensor_manifest_path}"
        )

    print("\n[STEP 2 COMPLETED]")
    print(f"Sensor manifest: {sensor_manifest_path}")

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("[PREPROCESSING FAILED]")
        print(f"{type(error).__name__}: {error}")
        print("=" * 70)
        traceback.print_exc()
        sys.exit(1)
