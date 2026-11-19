"""Run existing training scripts with optional MLflow, or import saved runs."""

import argparse
import csv
import importlib
import json
import math
import re
import os
import subprocess
import functools
from contextlib import contextmanager
from getpass import getpass
from tempfile import TemporaryDirectory
from pathlib import Path

# ============================================================
# 1. 공통 설정 (Configuration)
# ============================================================

# mlflow의 도메인 (--tracking-uri로 변경 가능)
DEFAULT_TRACKING_URI = "https://aimw.duckdns.org/mlflow/"

MODELS = {
    "sensor": ("sensor_1DCNN_regression_classification", "1dcnn"),
    "image": ("image_resnet_regression_classification", "resnet"),
    "multimodal": ("multimodal_multitask", "mm"),
}

# MLflow Parameters에 기록할 설정 목록
PARAMS = (
    "SEED", "NUM_WORKERS", "WINDOW_SIZE", "IMAGE_SIZE", "BATCH_SIZE",
    "MAX_EPOCHS", "PATIENCE", "LEARNING_RATE", "BACKBONE_LR", "HEAD_LR",
    "WEIGHT_DECAY", "DROPOUT", "HUBER_DELTA", "CLASSIFICATION_THRESHOLD",
    "PRETRAINED", "INIT_MODE", "ENCODER_MODE", "FUSION_EMBED_DIM",
    "MM_IMAGE_LR", "MM_SENSOR_LR", "MM_CLASSIFIER_LR", "CLS_LOSS_WEIGHT",
    "CLS_THRESHOLD", "SENSOR_CHANNEL", "SENSOR_CHANNEL_INDEX", "USE_HAMPEL",
    "HAMPEL_WINDOW_SIZE", "HAMPEL_N_SIGMAS", "RMS_WINDOW_MS",
)

# ============================================================
# 2. 결과 파일 및 평가 지표 처리 (Result Utilities)
# ============================================================

def checkpoint_name(model, task):
    return "best_mm.pth" if model == "multimodal" else f"best_{MODELS[model][1]}_{task}.pth"


def files_for(directory, model, task):
    names = ["history.csv", "test_metrics.csv", "test_predictions.csv", checkpoint_name(model, task)]
    if model == "multimodal":
        names.append("pareto_candidates.csv")
    return [directory / name for name in names]


def numeric_metrics(row, excluded=()):
    result = {}
    for key, value in row.items():
        if key in excluded or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[key] = number
    return result


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

# ============================================================
# 3. 실험 결과 업로드 (Metrics & Artifacts)
# ============================================================

def upload_results(mlflow, directory, model, task, *, upload_checkpoint=True):
    # 3-1. 결과 파일 누락 여부 확인
    paths = files_for(directory, model, task)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing run outputs: " + ", ".join(missing))
    
    # 3-2. history.csv의 학습 지표를 epoch 단위로 기록
    history = rows(paths[0])
    if not history:
        raise ValueError("history.csv is empty")
    for row in history:
        mlflow.log_metrics(numeric_metrics(row, ("epoch",)), step=int(row["epoch"]))
    mlflow.log_metric("epochs_completed", int(history[-1]["epoch"]))

    # 3-3. 전체 Test 및 Set별 평가 지표 기록
    for row in rows(paths[1]):
        scope = row.get("scope") or row.get("범위")
        if not scope:
            raise ValueError("Missing test metric scope")
        prefix = "test" if scope.lower() in {"overall", "test overall"} else "test_" + re.sub(r"[^a-z0-9_]+", "_", scope.lower())
        metrics = numeric_metrics(row, ("scope", "범위"))
        mlflow.log_metrics({f"{prefix}_{'records' if key == '기록 수' else key}": value for key, value in metrics.items()})

    # 3-4. 기존 결과 CSV와 체크포인트 파일을 Artifacts에 저장
    for path in paths:
        if path.suffix != ".pth" or upload_checkpoint:
            mlflow.log_artifact(str(path), artifact_path="results")

# ============================================================
# 4. 기존 학습 모듈 연결 (Model Configuration)
# ============================================================

def configure_module(model, task):
    module = importlib.import_module(MODELS[model][0])
    if model != "multimodal":
        module.TASK = task
        module.OUTPUT_DIR = module.OUTPUT_DIRS[task]
        module.BEST_MODEL_NAME = checkpoint_name(model, task)
        module.BEST_PATH = module.OUTPUT_DIR / module.BEST_MODEL_NAME
    return module

# ============================================================
# 5. 데이터셋 기록 (Dataset Tracking)
# ============================================================

def log_dataset(mlflow, frame, model, task):
    """Log the actual filtered/paired records returned by the training loader."""
    frame = frame.copy()

    # 5-1. Dataset 해시 계산을 위해 Path 객체를 문자열로 변환
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].map(lambda value: str(value) if isinstance(value, Path) else value)
    if "split" not in frame:
        raise ValueError("Dataset manifest needs a split column")
    
    # 5-2. 태스크별 정답 컬럼 검증
    if task == "regression":
        target = "wear"
    elif task == "classification":
        target = "target" if model == "sensor" else "label"
    else:
        target = None  # 멀티태스크는 wear와 type_target을 manifest에 모두 유지
    required = {"wear", "type_target"} if task == "multitask" else {target}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest missing target columns: {required - set(frame.columns)}")
    
    # 5-3. Train / Valid / Test manifest를 분리해 각각 기록
    with TemporaryDirectory() as tmp:
        for split, context in (("train", "training"), ("valid", "validation"), ("test", "testing")):
            subset = frame.loc[frame["split"].astype(str).str.strip().str.lower().eq(split)].reset_index(drop=True)
            if subset.empty:
                raise ValueError(f"Empty dataset split: {split}")
            path = Path(tmp) / f"{split}_manifest.csv"
            subset.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="datasets")
            source = f"{mlflow.get_artifact_uri('datasets')}/{path.name}"

            # 5-4. Dataset 메타데이터 등록.
            options = {"name": f"MATWI-{model}-{split}", "targets": target}
            try:
                dataset = mlflow.data.from_pandas(subset, source=source, **options)
            except mlflow.exceptions.MlflowException:
                dataset = mlflow.data.from_pandas(subset, **options)
            mlflow.log_input(dataset, context=context, tags={"manifest_uri": source})
            mlflow.log_param(f"{split}_records", len(subset))
            if "n_windows" in subset:
                mlflow.log_param(f"{split}_windows", int(subset["n_windows"].sum()))

# ============================================================
# 6. 학습 단계 추적 (Trace)
# ============================================================

@contextmanager
def instrument_training(mlflow, module, model, task):
    """Temporarily instrument existing module functions without editing training files."""
    originals = {}
    in_training = False
    loader = {"sensor": "load_sensor_manifest", "image": "load_and_prepare_metadata", "multimodal": "load_paired_metadata"}[model]

    def wrap(name, stage):
        original = getattr(module, name)
        originals[name] = original

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            nonlocal in_training
            # evaluate_model is also called every epoch; history stores those metrics.
            if name == "evaluate_model" and in_training:
                return original(*args, **kwargs)
            with mlflow.start_span(stage):
                previous = in_training
                if name == "train_model":
                    in_training = True
                try:
                    result = original(*args, **kwargs)
                    if name == loader:
                        log_dataset(mlflow, result, model, task)
                    return result
                finally:
                    in_training = previous
        setattr(module, name, wrapped)

    # Dataset 로딩 → DataLoader 생성 → 학습/검증 → Test 평가 단계 추적
    try:
        wrap(loader, "dataset_loading")
        wrap("build_dataloaders", "dataloader_build")
        wrap("train_model", "training_and_validation")
        wrap("evaluate_test" if model == "multimodal" else "evaluate_model", "test_evaluation")
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)

# ============================================================
# 7. 소스코드 및 버전 기록 (Source Code & Git)
# ============================================================

def log_source(mlflow, source_dir, model, historical=False):
    """Only upload explicit Python sources, never .env or credential files."""
    script = source_dir / f"{MODELS[model][0]}.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    mlflow.log_artifact(str(script), artifact_path="source")
    mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source/import_tool" if historical else "source")
    if not historical:
        try:
            commit = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            mlflow.set_tag("git_commit", commit)
            dirty = subprocess.check_output(["git", "-C", str(source_dir), "status", "--porcelain"], text=True).strip()
            mlflow.set_tag("git_dirty", bool(dirty))
        except (OSError, subprocess.CalledProcessError):
            pass
    return script

# ============================================================
# 8. 학습 모델 등록 (MLflow Model & Model Registry)
# ============================================================

def register_checkpoint(mlflow, checkpoint_path, source_file, model, task, registry_name):
    """Register a portable PyFunc with named tensor inputs, not just a .pth artifact."""
    import numpy as np
    import pandas as pd
    import torch
    from mlflow.models import infer_signature

    # 8-1. 최적 체크포인트를 CPU에서 읽고 태스크 및 입력 크기 확인
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    if model != "multimodal" and checkpoint.get("task") != task:
        raise ValueError("Checkpoint task does not match the requested task")
    window = int(checkpoint.get("window_size", config.get("window_size", 512)))
    size = checkpoint.get("image_size", config.get("image_size", 224))
    if isinstance(size, (tuple, list)):
        height, width = map(int, size)
    else:
        height = width = int(size)
    module_name = source_file.stem

    # 8-2. 세 가지 모델 유형을 PyFunc로 구성
    class WearPredictor(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import torch
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, context.artifacts["source"])
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            ckpt = torch.load(context.artifacts["checkpoint"], map_location="cpu", weights_only=True)
            cfg = ckpt.get("config", {})
            if model == "sensor":
                self.net = mod.ForceY1DCNN(dropout=float(cfg.get("dropout", 0.3)))
            elif model == "image":
                self.net = mod.ResNet18ToolWearModel(task=task, pretrained=False)
            else:
                mod.FUSION_EMBED_DIM = int(ckpt["fusion_embed_dim"])
                self.net = mod.ImageForceYFusionModel(init_mode="random_init", encoder_mode=ckpt["encoder_mode"])
            self.net.load_state_dict(ckpt["model_state_dict"], strict=True)
            self.net.eval()
            self.threshold = float(ckpt.get("cls_threshold", cfg.get("classification_threshold", 0.5)))

        # 전처리된 입력 텐서로 회귀값 및/또는 분류 확률과 결과를 반환
        def predict(self, context, model_input, params=None):
            import torch
            import pandas as pd
            def tensor(name):
                return torch.as_tensor(model_input[name], dtype=torch.float32)
            with torch.inference_mode():
                if model == "multimodal":
                    wear, logit = self.net(tensor("image"), tensor("sensor"))
                elif model == "sensor":
                    out = self.net(tensor("sensor"))
                else:
                    out = self.net(tensor("image"))
                if model != "multimodal":
                    wear, logit = (out, None) if task == "regression" else (None, out)
                data = {}
                if wear is not None:
                    data["wear_um"] = wear.reshape(-1).cpu().numpy()
                if logit is not None:
                    prob = torch.sigmoid(logit).reshape(-1).cpu().numpy()
                    data["flank_probability"] = prob
                    data["flank_prediction"] = (prob >= self.threshold).astype("int64")
                return pd.DataFrame(data)

    # 8-3. 실제 원본 데이터 대신 가상 입력으로 Signature 생성
    # 센서: (N, 1, window), 이미지: (N, 3, height, width)
    example = {}
    if model in ("sensor", "multimodal"):
        example["sensor"] = np.zeros((1, 1, window), dtype=np.float32)
    if model in ("image", "multimodal"):
        example["image"] = np.zeros((1, 3, height, width), dtype=np.float32)
    # Use the exact supplied training source for reconstruction, also for historical imports.
    import types
    from importlib.metadata import version
    artifacts = {"checkpoint": str(checkpoint_path), "source": str(source_file)}
    predictor = WearPredictor()
    predictor.load_context(types.SimpleNamespace(artifacts=artifacts))
    signature = infer_signature(example, predictor.predict(None, example))
    dependencies = ["mlflow", "torch", "numpy", "pandas", "cloudpickle", "scikit-learn"]
    if model != "sensor":
        dependencies.extend(["torchvision", "pillow"])

    # 8-4. 모델 가중치, 소스코드, Signature, 패키지 버전을 함께 저장
    info = mlflow.pyfunc.log_model(
        name="model", python_model=WearPredictor(),
        artifacts=artifacts,
        input_example=example, signature=signature,
        registered_model_name=registry_name,
        pip_requirements=[
            f"{dep}=={mlflow.__version__}"
            if dep == "mlflow"
            else f"{dep}=={version(dep).split('+')[0]}"
            for dep in dependencies
        ],
    )

    # 8-5. 등록 모델 이름 및 Best Epoch 기록
    if "epoch" in checkpoint:
        mlflow.log_metric("best_epoch", int(checkpoint["epoch"]))
    mlflow.set_tag("registered_model_name", registry_name)
    print(f"Registered model: {registry_name} ({info.model_uri})")

# ============================================================
# 9. 실행 옵션
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "import"])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--task", choices=["regression", "classification", "multitask"])
    parser.add_argument("--mlflow", action="store_true", help="Enable tracking for new training")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI), help="Defaults to team MLflow domain")
    parser.add_argument("--experiment", help="Defaults to sensor, image, or multimodal")
    parser.add_argument("--run-name")
    parser.add_argument("--results-dir", type=Path, help="Required for importing a completed run")
    parser.add_argument("--params-json", type=Path, help="Original parameters for importing a completed run (optional)")
    parser.add_argument("--no-checkpoint", action="store_true", help="Do not upload the .pth artifact")
    parser.add_argument("--no-registry", action="store_true", help="Skip model registry (also avoids packaging the checkpoint into an MLflow model)")
    parser.add_argument("--registered-model-name", help="Override the default registered model name")
    parser.add_argument("--manifest", type=Path, help="Original filtered/paired manifest for a historical import")
    parser.add_argument("--source-dir", type=Path, help="Original model source directory for a historical import")
    args = parser.parse_args()
    if args.task is None:
        args.task = "multitask" if args.model == "multimodal" else "regression"
    if (args.model == "multimodal") != (args.task == "multitask"):
        parser.error("multimodal requires multitask; sensor/image require regression or classification")
    if args.mode == "import" and args.results_dir is None:
        parser.error("import requires --results-dir")
    if args.mode == "train" and (args.results_dir or args.params_json or args.manifest or args.source_dir):
        parser.error("--results-dir, --params-json, --manifest and --source-dir are for import only")
    return args

# ============================================================
# 10. Main
# ============================================================

def main():
    args = parse_args()
    module = configure_module(args.model, args.task) if args.mode == "train" else None
    if args.mode == "train" and not args.mlflow:
        module.main()
        return

    # MLflow is an optional dependency; ordinary training never imports it.
    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit("Install MLflow in the current environment: python -m pip install mlflow") from exc
    if args.tracking_uri == "https://aimw.duckdns.org/mlflow":
        raise SystemExit("Set DEFAULT_TRACKING_URI in runner.py or pass --tracking-uri once.")
    if not args.tracking_uri.startswith("https://"):
        raise SystemExit("Use an HTTPS tracking URI for Basic Auth.")
    if not hasattr(mlflow, "start_span") or int(mlflow.__version__.split(".")[0]) < 3:
        raise SystemExit("This runner requires MLflow 3.x with tracing support; check client/server compatibility.")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = input("MLflow Username: ")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass("MLflow Password: ")
    mlflow.set_experiment(args.experiment or args.model)
    directory = Path(module.OUTPUT_DIR) if module else args.results_dir
    params = {"modality": args.model, "task": args.task}
    if module:
        params.update({key.lower(): getattr(module, key) for key in PARAMS if hasattr(module, key)})
    elif args.params_json:
        original = json.loads(args.params_json.read_text(encoding="utf-8"))
        if not isinstance(original, dict):
            raise ValueError("--params-json must contain a JSON object")
        params = {**original, **params}

    with mlflow.start_run(run_name=args.run_name or f"{args.model}-{args.task}") as run:
        mlflow.set_tag("source", "training" if module else "historical_import")
        mlflow.log_params(params)
        with mlflow.start_span(f"{args.mode}_{args.model}_{args.task}", attributes={"run_id": run.info.run_id, "task": args.task, "model": args.model}):
            mlflow.update_current_trace(tags={"run_id": run.info.run_id, "mode": args.mode})
            source = None
            if module or args.source_dir:
                with mlflow.start_span("source_logging"):
                    source = log_source(mlflow, Path(module.__file__).resolve().parent if module else args.source_dir.resolve(), args.model, historical=not bool(module))
            else:
                mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source/import_tool")
                print("Historical source omitted: use --source-dir to provide the original code.")
            if module:
                paths = files_for(directory, args.model, args.task)
                before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.exists()}
                with instrument_training(mlflow, module, args.model, args.task):
                    module.main()
                stale = [str(p) for p in paths if not p.is_file() or before.get(p) == (p.stat().st_mtime_ns, p.stat().st_size)]
                if stale:
                    raise RuntimeError("Training did not produce fresh outputs: " + ", ".join(stale))
                norm = getattr(module, "NORMALIZATION_PATH", None)
                if norm is None and args.model == "sensor":
                    norm = module.MANIFEST_PATH.parent / "normalization.json"
                if norm and Path(norm).is_file():
                    mlflow.log_artifact(str(norm), artifact_path="datasets")
            elif args.manifest:
                import pandas as pd
                with mlflow.start_span("historical_manifest_import"):
                    log_dataset(mlflow, pd.read_csv(args.manifest), args.model, args.task)
            else:
                print("Historical dataset omitted: use --manifest to provide the original filtered/paired manifest.")
            with mlflow.start_span("results_upload"):
                upload_results(mlflow, directory, args.model, args.task, upload_checkpoint=not args.no_checkpoint)
            if not args.no_registry and source:
                if args.no_checkpoint:
                    print("--no-checkpoint skips only the standalone artifact; Registry still packages model weights.")
                with mlflow.start_span("model_registration"):
                    register_checkpoint(mlflow, directory / checkpoint_name(args.model, args.task), source, args.model, args.task, args.registered_model_name or f"matwi-{args.model}-{args.task}")
            elif not args.no_registry:
                print("Registry skipped: historical imports require --source-dir with matching original model code.")
        print(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
