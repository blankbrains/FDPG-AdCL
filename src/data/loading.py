"""Unified, fail-fast loaders for the nine supported AdCL-FD datasets."""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_DATASETS = ("smd", "msl", "smap", "swat", "wadi", "psm", "kpi", "yahoo", "ucr")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a comma-separated file without dropping a numeric first row."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        first_row = next(csv.reader(handle), None)
    if not first_row:
        raise ValueError(f"Dataset file is empty: {path}")
    has_header = False
    for value in first_row:
        try:
            float(value)
        except ValueError:
            has_header = True
            break
    return pd.read_csv(path, header=0 if has_header else None)


def _numeric_2d(data: np.ndarray, source: str) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{source} must be a non-empty 2D time-series array, got {array.shape}")
    if np.isinf(array).any():
        raise ValueError(f"{source} contains Inf values")
    return array


def _binary_labels(values: np.ndarray | pd.Series, source: str) -> np.ndarray:
    raw = np.asarray(values).ravel()
    if raw.size == 0:
        raise ValueError(f"{source} labels are empty")
    converted = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)
    if np.isfinite(converted).all():
        unique = set(np.unique(converted).tolist())
        if unique.issubset({0.0, 1.0}):
            return converted.astype(np.int32)
        if unique.issubset({-1.0, 1.0}):
            return (converted < 0).astype(np.int32)
    normalized = np.char.lower(np.char.strip(raw.astype(str)))
    normal = np.isin(normalized, ["0", "normal", "false"])
    anomalous = np.isin(normalized, ["1", "attack", "anomaly", "true"])
    if np.all(normal | anomalous):
        return anomalous.astype(np.int32)
    raise ValueError(f"{source} labels must use 0/1, -1/1, or named normal/attack values")


def _numeric_sensor_frame(
    frame: pd.DataFrame,
    *,
    skip_columns: set[str],
    source: str,
) -> pd.DataFrame:
    normalized = [str(column).strip() for column in frame.columns]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{source} contains duplicate columns")
    skip = {name.strip().lower() for name in skip_columns}
    selected = [
        column
        for column, name in zip(frame.columns, normalized)
        if name.lower() not in skip
    ]
    if not selected:
        raise ValueError(f"No sensor columns found in {source}")
    numeric = frame.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    numeric.columns = [str(column).strip() for column in selected]
    if np.isinf(numeric.to_numpy(dtype=np.float64)).any():
        raise ValueError(f"{source} contains Inf sensor values")
    return numeric


def _align_sensor_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    expected: int | None,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_columns = list(train.columns)
    test_columns = list(test.columns)
    if set(train_columns) != set(test_columns):
        missing = sorted(set(train_columns) - set(test_columns))
        extra = sorted(set(test_columns) - set(train_columns))
        raise ValueError(
            f"{source} sensor schema mismatch: missing={missing}, extra={extra}"
        )
    if expected is not None and len(train_columns) != expected:
        raise ValueError(
            f"{source} has {len(train_columns)} channels, expected {expected}"
        )
    return train, test.loc[:, train_columns]


def _drop_unusable_training_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if list(train.columns) != list(test.columns):
        raise ValueError(f"{source} frames must be schema-aligned before column filtering")
    dropped = [
        str(column)
        for column in train.columns
        if not train[column].notna().any()
    ]
    for column in dropped:
        if test[column].notna().any():
            raise ValueError(
                f"{source} training-all-missing column {column!r} contains test values"
            )
    if len(dropped) == len(train.columns):
        raise ValueError(f"{source} has no usable training sensor columns")
    return (
        train.drop(columns=dropped),
        test.drop(columns=dropped),
        dropped,
    )


def _read_numeric_frame(path: Path, *, skip_columns: set[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return _numeric_sensor_frame(
        _read_csv(path),
        skip_columns=skip_columns or set(),
        source=str(path),
    )


def _validate_channels(data: np.ndarray, expected: int | None, source: str) -> np.ndarray:
    array = _numeric_2d(data, source)
    if expected is not None and array.shape[1] != expected:
        raise ValueError(f"{source} has {array.shape[1]} channels, expected {expected}")
    return array


def _entity_file(folder: Path, entity: str | None, suffixes: tuple[str, ...]) -> Path:
    if not entity:
        raise ValueError(f"An entity name is required for dataset directory {folder}")
    direct = folder / entity
    candidates = [direct] + [folder / f"{entity}{suffix}" for suffix in suffixes if not entity.endswith(suffix)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No entity file found for {entity!r} under {folder}")


def _load_smd(root: Path, entity: str | None, expected: int | None):
    train_path = _entity_file(root / "train", entity, (".txt", ".csv"))
    test_path = _entity_file(root / "test", entity, (".txt", ".csv"))
    label_path = _entity_file(root / "test_label", entity, (".txt", ".csv"))
    train = np.loadtxt(train_path, delimiter=",")
    test = np.loadtxt(test_path, delimiter=",")
    labels = _binary_labels(np.loadtxt(label_path, delimiter=","), str(label_path))
    train = _validate_channels(train, expected, str(train_path))
    test = _validate_channels(test, expected, str(test_path))
    if len(test) != len(labels):
        raise ValueError("SMD test data and labels have different lengths")
    return train, test, np.zeros(len(train), dtype=np.int32), labels


def _load_nasa(root: Path, entity: str | None, expected: int | None):
    train_path = _entity_file(root / "train", entity, (".npy",))
    test_path = _entity_file(root / "test", entity, (".npy",))
    manifest_path = root / "labeled_anomalies.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"NASA anomaly manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if "chan_id" not in manifest or "anomaly_sequences" not in manifest:
        raise ValueError("NASA manifest must contain chan_id and anomaly_sequences columns")
    key = Path(train_path).stem
    rows = manifest[manifest["chan_id"].astype(str) == key]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one anomaly manifest row for entity {key!r}, got {len(rows)}")
    train = _validate_channels(np.load(train_path), expected, str(train_path))
    test = _validate_channels(np.load(test_path), expected, str(test_path))
    labels = np.zeros(len(test), dtype=np.int32)
    sequences = ast.literal_eval(str(rows.iloc[0]["anomaly_sequences"]))
    for start, end in sequences:
        start_i, end_i = int(start), int(end)
        if start_i < 0 or end_i < start_i or end_i >= len(labels):
            raise ValueError(f"Invalid anomaly interval {(start_i, end_i)} for entity {key}")
        labels[start_i:end_i + 1] = 1
    return train, test, np.zeros(len(train), dtype=np.int32), labels


def _find_first(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of the expected files exist under {root}: {', '.join(names)}")


def _load_swat(root: Path, expected: int | None, entity: str | None = None):
    train_path = _find_first(root, ("swat_train.csv", "SWaT_Dataset_Normal_v1_train.csv"))
    test_path = (
        _entity_file(root, entity, ())
        if entity
        else _find_first(
            root,
            (
                "swat_test.csv",
                "SWaT_Dataset_Attack_v0_test.csv",
                "attack_raw.csv",
            ),
        )
    )

    def read(path: Path):
        frame = pd.read_csv(path)
        label_column = next((c for c in frame.columns if str(c).strip().lower() in {"attack", "label", "normal/attack"}), None)
        if label_column is None:
            raise ValueError(f"SWaT file lacks an attack/label column: {path}")
        labels = _binary_labels(frame[label_column], str(path))
        features = _numeric_sensor_frame(
            frame,
            skip_columns={str(label_column), "timestamp", "date", "time"},
            source=str(path),
        )
        return features, labels

    train_frame, train_labels = read(train_path)
    test_frame, test_labels = read(test_path)
    train_frame, test_frame = _align_sensor_frames(
        train_frame, test_frame, expected=expected, source="SWaT"
    )
    return (
        train_frame.to_numpy(dtype=np.float32),
        test_frame.to_numpy(dtype=np.float32),
        train_labels,
        test_labels,
    )


def _load_wadi(root: Path, expected: int | None):
    train_path = _find_first(root, ("wadi_14days_new.csv", "WADI_14days_new.csv"))
    test_path = _find_first(root, ("wadi_attackdataLABLE.csv", "WADI_attackdataLABLE.csv"))

    def read_features(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        if frame.shape[1] < 4:
            raise ValueError(f"WADI file has too few columns: {path}")
        metadata = {str(column) for column in frame.columns[:3]}
        label_names = {"attack", "label", "anomaly", "class"}
        return _numeric_sensor_frame(
            frame,
            skip_columns=metadata | label_names,
            source=str(path),
        )

    train_raw = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    attack_column = next((c for c in test_frame.columns if str(c).strip().lower() == "attack"), None)
    if attack_column is None:
        raise ValueError(f"WADI test file lacks Attack column: {test_path}")
    train_frame = read_features(train_raw, train_path)
    test_features = read_features(test_frame, test_path)
    train_frame, test_features = _align_sensor_frames(
        train_frame, test_features, expected=None, source="WADI"
    )
    train_frame, test_features, _ = _drop_unusable_training_columns(
        train_frame,
        test_features,
        source="WADI",
    )
    train_frame, test_features = _align_sensor_frames(
        train_frame, test_features, expected=expected, source="WADI"
    )
    labels = _binary_labels(test_frame[attack_column], str(test_path))
    if len(test_features) != len(labels):
        raise ValueError("WADI test data and labels have different lengths")
    train = train_frame.to_numpy(dtype=np.float32)
    test = test_features.to_numpy(dtype=np.float32)
    return train, test, np.zeros(len(train), dtype=np.int32), labels


def _load_psm(root: Path, expected: int | None):
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    label_path = root / "test_label.csv"
    timestamp_names = {"timestamp", "timestamp_(min)", "time", "date", "index"}
    train_frame = _numeric_sensor_frame(
        _read_csv(train_path),
        skip_columns=timestamp_names,
        source=str(train_path),
    )
    test_frame = _numeric_sensor_frame(
        _read_csv(test_path),
        skip_columns=timestamp_names,
        source=str(test_path),
    )
    train_frame, test_frame = _align_sensor_frames(
        train_frame, test_frame, expected=expected, source="PSM"
    )
    label_frame = _read_numeric_frame(label_path)
    labels = _binary_labels(label_frame.iloc[:, -1], str(label_path))
    train = _validate_channels(train_frame.to_numpy(), expected, str(train_path))
    test = _validate_channels(test_frame.to_numpy(), expected, str(test_path))
    if len(test) != len(labels):
        raise ValueError("PSM test data and labels have different lengths")
    return train, test, np.zeros(len(train), dtype=np.int32), labels


def _read_value_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    frame = _read_csv(path)
    lower = {str(c).strip().lower(): c for c in frame.columns}
    value_col = lower.get("value")
    label_col = lower.get("label") or lower.get("anomaly") or lower.get("attack")
    if value_col is None or label_col is None:
        if frame.shape[1] < 2:
            raise ValueError(f"Expected value and label columns in {path}")
        value_col, label_col = frame.columns[-2], frame.columns[-1]
    values = pd.to_numeric(frame[value_col], errors="coerce")
    if np.isinf(values.to_numpy(dtype=np.float64)).any():
        raise ValueError(f"Inf values in {path}")
    raw_labels = frame[label_col]
    normalized_labels = raw_labels.astype(str).str.strip().str.lower()
    missing_labels = raw_labels.isna() | normalized_labels.isin(
        {"", "nan", "none", "null"}
    )
    if (missing_labels & values.notna()).any():
        count = int((missing_labels & values.notna()).sum())
        raise ValueError(
            f"{path} has {count} missing labels for observed sensor values"
        )
    label_values = raw_labels.mask(missing_labels, 0)
    labels = _binary_labels(label_values, str(path))
    return values.to_numpy(dtype=np.float32)[:, None], labels


def _read_value_only(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    frame = _read_csv(path)
    lower = {str(column).strip().lower(): column for column in frame.columns}
    value_column = lower.get("value")
    if value_column is None:
        if frame.shape[1] == 1:
            value_column = frame.columns[0]
        elif frame.shape[1] >= 2:
            value_column = frame.columns[-2]
        else:
            raise ValueError(f"No value column found in {path}")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if np.isinf(values.to_numpy(dtype=np.float64)).any():
        raise ValueError(f"Inf values in {path}")
    return values.to_numpy(dtype=np.float32)[:, None]


def _load_kpi(root: Path, entity: str | None, expected: int | None):
    train_path = _entity_file(root / "train", entity, (".csv",))
    test_path = _entity_file(root / "test", entity, (".csv",))
    train = _read_value_only(train_path)
    test, test_labels = _read_value_label(test_path)
    return (
        _validate_channels(train, expected, str(train_path)),
        _validate_channels(test, expected, str(test_path)),
        np.zeros(len(train), dtype=np.int32),
        test_labels,
    )


def _load_yahoo(root: Path, entity: str | None, expected: int | None):
    path = _entity_file(root, entity, (".csv",))
    data, labels = _read_value_label(path)
    split = len(data) // 2
    if split < 2 or len(data) - split < 2:
        raise ValueError(f"Yahoo entity {path.name} is too short for a train/test split")
    data = _validate_channels(data, expected, str(path))
    return data[:split], data[split:], labels[:split], labels[split:]


def _load_ucr(root: Path, entity: str | None, expected: int | None):
    path = _entity_file(root, entity, (".csv", ".txt"))
    try:
        data, labels = _read_value_label(path)
    except ValueError:
        raw = np.loadtxt(path, delimiter=",")
        if raw.ndim == 2 and raw.shape[1] > 1:
            raw = raw[:, -1]
        data = np.asarray(raw, dtype=np.float32).reshape(-1, 1)
        labels = np.zeros(len(data), dtype=np.int32)
        match = re.search(r"_(\d+)_(\d+)_(\d+)\.(?:csv|txt)$", path.name, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot infer UCR train/anomaly boundaries from {path.name}")
        _, anomaly_start, anomaly_end = map(int, match.groups())
        labels[anomaly_start:anomaly_end] = 1
    match = re.search(r"_(\d+)_(\d+)_(\d+)\.(?:csv|txt)$", path.name, re.IGNORECASE)
    if match is not None:
        split = int(match.group(1))
    else:
        anomaly_indices = np.flatnonzero(labels)
        if anomaly_indices.size == 0:
            raise ValueError(f"UCR entity {path.name} has no split metadata or anomaly labels")
        split = int(anomaly_indices[0])
    if split < 2 or split >= len(data) - 1:
        raise ValueError(f"Invalid UCR training prefix length {split} for {path.name}")
    data = _validate_channels(data, expected, str(path))
    return data[:split], data[split:], labels[:split], labels[split:]


def load_feature_pair(
    dataset: str,
    data_root: str | Path,
    *,
    entity: str | None = None,
    expected_channels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load chronological train/test features without reading anomaly labels."""
    name = dataset.strip().lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if expected_channels is not None and expected_channels < 1:
        raise ValueError("expected_channels must be positive")

    if name == "smd":
        train_path = _entity_file(root / "train", entity, (".txt", ".csv"))
        test_path = _entity_file(root / "test", entity, (".txt", ".csv"))
        train = np.loadtxt(train_path, delimiter=",")
        test = np.loadtxt(test_path, delimiter=",")
        return (
            _validate_channels(train, expected_channels, str(train_path)),
            _validate_channels(test, expected_channels, str(test_path)),
        )

    if name in {"msl", "smap"}:
        train_path = _entity_file(root / "train", entity, (".npy",))
        test_path = _entity_file(root / "test", entity, (".npy",))
        return (
            _validate_channels(np.load(train_path), expected_channels, str(train_path)),
            _validate_channels(np.load(test_path), expected_channels, str(test_path)),
        )

    if name == "swat":
        train_path = _find_first(
            root,
            ("swat_train.csv", "SWaT_Dataset_Normal_v1_train.csv"),
        )
        test_path = (
            _entity_file(root, entity, ())
            if entity
            else _find_first(
                root,
                (
                    "swat_test.csv",
                    "SWaT_Dataset_Attack_v0_test.csv",
                    "attack_raw.csv",
                ),
            )
        )

        def read_features(path: Path) -> pd.DataFrame:
            frame = pd.read_csv(path)
            return _numeric_sensor_frame(
                frame,
                skip_columns={
                    "attack", "label", "normal/attack",
                    "timestamp", "date", "time",
                },
                source=str(path),
            )

        train_frame, test_frame = _align_sensor_frames(
            read_features(train_path),
            read_features(test_path),
            expected=expected_channels,
            source="SWaT",
        )
        return (
            _validate_channels(train_frame.to_numpy(), expected_channels, str(train_path)),
            _validate_channels(test_frame.to_numpy(), expected_channels, str(test_path)),
        )

    if name == "wadi":
        train_path = _find_first(root, ("wadi_14days_new.csv", "WADI_14days_new.csv"))
        test_path = _find_first(root, ("wadi_attackdataLABLE.csv", "WADI_attackdataLABLE.csv"))

        def read_features(path: Path) -> pd.DataFrame:
            frame = pd.read_csv(path)
            if frame.shape[1] < 4:
                raise ValueError(f"WADI file has too few columns: {path}")
            metadata = {str(column) for column in frame.columns[:3]}
            return _numeric_sensor_frame(
                frame,
                skip_columns=metadata | {"attack", "label", "anomaly", "class"},
                source=str(path),
            )

        train_frame = read_features(train_path)
        test_frame = read_features(test_path)
        train_frame, test_frame = _align_sensor_frames(
            train_frame,
            test_frame,
            expected=None,
            source="WADI",
        )
        train_frame, test_frame, _ = _drop_unusable_training_columns(
            train_frame,
            test_frame,
            source="WADI",
        )
        train_frame, test_frame = _align_sensor_frames(
            train_frame,
            test_frame,
            expected=expected_channels,
            source="WADI",
        )
        return (
            train_frame.to_numpy(dtype=np.float32),
            test_frame.to_numpy(dtype=np.float32),
        )

    if name == "psm":
        train_path = root / "train.csv"
        test_path = root / "test.csv"
        timestamp_names = {"timestamp", "timestamp_(min)", "time", "date", "index"}
        train_frame = _numeric_sensor_frame(
            _read_csv(train_path),
            skip_columns=timestamp_names,
            source=str(train_path),
        )
        test_frame = _numeric_sensor_frame(
            _read_csv(test_path),
            skip_columns=timestamp_names,
            source=str(test_path),
        )
        train_frame, test_frame = _align_sensor_frames(
            train_frame,
            test_frame,
            expected=expected_channels,
            source="PSM",
        )
        return (
            train_frame.to_numpy(dtype=np.float32),
            test_frame.to_numpy(dtype=np.float32),
        )

    if name == "kpi":
        train_path = _entity_file(root / "train", entity, (".csv",))
        test_path = _entity_file(root / "test", entity, (".csv",))
        train = _read_value_only(train_path)
        test = _read_value_only(test_path)
        return (
            _validate_channels(train, expected_channels, str(train_path)),
            _validate_channels(test, expected_channels, str(test_path)),
        )

    if name == "yahoo":
        path = _entity_file(root, entity, (".csv",))
        data = _read_value_only(path)
        split = len(data) // 2
        if split < 2 or len(data) - split < 2:
            raise ValueError(f"Yahoo entity {path.name} is too short for a train/test split")
        data = _validate_channels(data, expected_channels, str(path))
        return data[:split], data[split:]

    path = _entity_file(root, entity, (".csv", ".txt"))
    match = re.search(r"_(\d+)_(\d+)_(\d+)\.(?:csv|txt)$", path.name, re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"UCR entity {path.name} lacks filename split metadata required for unlabeled loading"
        )
    split = int(match.group(1))
    data = _read_value_only(path)
    if split < 2 or split >= len(data) - 1:
        raise ValueError(f"Invalid UCR training prefix length {split} for {path.name}")
    data = _validate_channels(data, expected_channels, str(path))
    return data[:split], data[split:]


def load_dataset(
    dataset: str,
    data_root: str | Path,
    *,
    entity: str | None = None,
    expected_channels: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return contiguous ``train, test, train_labels, test_labels`` arrays."""
    name = dataset.strip().lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if expected_channels is not None and expected_channels < 1:
        raise ValueError("expected_channels must be positive")
    if name == "smd":
        result = _load_smd(root, entity, expected_channels)
    elif name in {"msl", "smap"}:
        result = _load_nasa(root, entity, expected_channels)
    elif name == "swat":
        result = _load_swat(root, expected_channels, entity)
    elif name == "wadi":
        result = _load_wadi(root, expected_channels)
    elif name == "psm":
        result = _load_psm(root, expected_channels)
    elif name == "kpi":
        result = _load_kpi(root, entity, expected_channels)
    elif name == "yahoo":
        result = _load_yahoo(root, entity, expected_channels)
    else:
        result = _load_ucr(root, entity, expected_channels)
    train, test, train_labels, test_labels = result
    if len(train) != len(train_labels) or len(test) != len(test_labels):
        raise RuntimeError("Loader returned data/label length mismatch")
    return train, test, train_labels.astype(np.int32), test_labels.astype(np.int32)


def load_training_data(
    dataset: str,
    data_root: str | Path,
    *,
    entity: str | None = None,
    expected_channels: int | None = None,
) -> np.ndarray:
    """Load training features without opening test data or parsing labels."""
    name = dataset.strip().lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if expected_channels is not None and expected_channels < 1:
        raise ValueError("expected_channels must be positive")

    if name == "smd":
        path = _entity_file(root / "train", entity, (".txt", ".csv"))
        train = np.loadtxt(path, delimiter=",")
    elif name in {"msl", "smap"}:
        path = _entity_file(root / "train", entity, (".npy",))
        train = np.load(path)
    elif name == "swat":
        path = _find_first(
            root,
            ("swat_train.csv", "SWaT_Dataset_Normal_v1_train.csv"),
        )
        frame = pd.read_csv(path)
        features = _numeric_sensor_frame(
            frame,
            skip_columns={
                "attack", "label", "normal/attack",
                "timestamp", "date", "time",
            },
            source=str(path),
        )
        train = features.to_numpy(dtype=np.float32)
    elif name == "wadi":
        path = _find_first(
            root,
            ("wadi_14days_new.csv", "WADI_14days_new.csv"),
        )
        frame = pd.read_csv(path)
        if frame.shape[1] < 4:
            raise ValueError(f"WADI file has too few columns: {path}")
        metadata = {str(column) for column in frame.columns[:3]}
        features = _numeric_sensor_frame(
            frame,
            skip_columns=metadata | {"attack", "label", "anomaly", "class"},
            source=str(path),
        )
        features, _, _ = _drop_unusable_training_columns(
            features,
            features.copy(),
            source="WADI",
        )
        train = features.to_numpy(dtype=np.float32)
    elif name == "psm":
        path = root / "train.csv"
        features = _numeric_sensor_frame(
            _read_csv(path),
            skip_columns={
                "timestamp", "timestamp_(min)", "time", "date", "index",
            },
            source=str(path),
        )
        train = features.to_numpy(dtype=np.float32)
    elif name == "kpi":
        path = _entity_file(root / "train", entity, (".csv",))
        train = _read_value_only(path)
    elif name == "yahoo":
        path = _entity_file(root, entity, (".csv",))
        values = _read_value_only(path)
        split = len(values) // 2
        if split < 2 or len(values) - split < 2:
            raise ValueError(f"Yahoo entity {path.name} is too short for a train/test split")
        train = values[:split]
    else:
        path = _entity_file(root, entity, (".csv", ".txt"))
        match = re.search(
            r"_(\d+)_(\d+)_(\d+)\.(?:csv|txt)$",
            path.name,
            re.IGNORECASE,
        )
        if match is None:
            raise ValueError(
                f"UCR entity {path.name} lacks filename split metadata required for unlabeled training"
            )
        split = int(match.group(1))
        values = _read_value_only(path)
        if split < 2 or split >= len(values) - 1:
            raise ValueError(f"Invalid UCR training prefix length {split} for {path.name}")
        train = values[:split]
    return _validate_channels(train, expected_channels, f"{name} training data")


def load_validation_data(
    dataset: str,
    data_root: str | Path,
    *,
    entity: str | None = None,
    expected_channels: int | None = None,
) -> np.ndarray:
    """Load explicit validation features without reading anomaly labels."""
    name = dataset.strip().lower()
    if name != "swat":
        raise ValueError("Explicit validation data is currently supported only for SWaT")
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    path = (
        _entity_file(root, entity, ())
        if entity
        else _find_first(root, ("swat_validation.csv",))
    )
    frame = pd.read_csv(path)
    features = _numeric_sensor_frame(
        frame,
        skip_columns={"attack", "label", "normal/attack", "timestamp", "date", "time"},
        source=str(path),
    )
    return _validate_channels(
        features.to_numpy(dtype=np.float32),
        expected_channels,
        f"{name} validation data",
    )
