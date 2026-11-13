import csv
import io
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get(
    "HAR_MODEL_PATH",
    APP_DIR / "outputs" / "mlp_har.joblib",
))
LR_MODEL_PATH = Path(os.environ.get(
    "HAR_LR_MODEL_PATH",
    APP_DIR / "outputs" / "lr_har.joblib",
))
INDEX_PATH = APP_DIR / "index.html"
SAMPLE_PATH = APP_DIR / "samples" / "har_test_sample.csv"
FEATURE_COUNT = 561
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_ROWS = 10_000
MAX_COLUMNS = 1_000
PREVIEW_ROWS = 20
PREVIEW_COLUMNS = 20
LABEL_COLUMN_NAMES = {
    "activity", "activity_label", "activity_id", "label", "target",
    "class", "y", "ground_truth", "true_label",
}

app = FastAPI(title="UCI HAR Activity Predictor")


def _number_or_none(value):
    value = value.strip()
    if not value or value.casefold() in {"na", "n/a", "nan", "null", "none"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _looks_like_header(row):
    non_empty = [value.strip() for value in row if value.strip()]
    return bool(non_empty) and any(_number_or_none(value) is None for value in non_empty)


def _column_statistics(name, values):
    missing_count = sum(
        not value.strip() or value.strip().casefold() in {"na", "n/a", "nan", "null", "none"}
        for value in values
    )
    parsed = [_number_or_none(value) for value in values]
    numeric_values = [value for value in parsed if value is not None]
    non_missing_count = len(values) - missing_count
    non_numeric_count = non_missing_count - len(numeric_values)
    if non_missing_count == 0:
        column_type = "empty"
    elif non_numeric_count == 0:
        column_type = "numeric"
    elif not numeric_values:
        column_type = "text"
    else:
        column_type = "mixed"

    stats = {
        "name": name,
        "type": column_type,
        "missing": missing_count,
        "non_numeric": non_numeric_count,
        "unique": len({value.strip() for value in values if value.strip()}),
    }

    if numeric_values:
        q1, median, q3 = np.percentile(numeric_values, [25, 50, 75])
        iqr = q3 - q1
        outlier_count = sum(
            value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr
            for value in numeric_values
        )
        stats.update({
            "min": round(float(min(numeric_values)), 6),
            "max": round(float(max(numeric_values)), 6),
            "mean": round(float(np.mean(numeric_values)), 6),
            "median": round(float(median), 6),
            "std": round(float(np.std(numeric_values)), 6),
            "q1": round(float(q1), 6),
            "q3": round(float(q3), 6),
            "outliers": int(outlier_count),
        })
    if non_numeric_count:
        counts = Counter(value.strip() for value in values if value.strip())
        stats["most_common"] = [
            {"value": value, "count": count}
            for value, count in counts.most_common(3)
        ]
    return stats


def parse_csv_data(contents):
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV phải được mã hóa UTF-8.") from error

    try:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(io.StringIO(text), strict=True)
            if any(cell.strip() for cell in row)
        ]
    except csv.Error as error:
        raise ValueError("Không đọc được cấu trúc CSV.") from error

    if not rows:
        raise ValueError("CSV không có dữ liệu.")

    has_header = _looks_like_header(rows[0])
    headers = rows[0] if has_header else []
    data_rows = rows[1:] if has_header else rows
    if not data_rows:
        raise ValueError("CSV chỉ có tiêu đề, chưa có dòng dữ liệu.")
    if len(data_rows) > MAX_ROWS:
        raise ValueError(f"Tối đa {MAX_ROWS:,} dòng dữ liệu mỗi lần tải lên.")

    column_count = max(len(row) for row in [headers] + data_rows)
    if column_count > MAX_COLUMNS:
        raise ValueError(f"Tối đa {MAX_COLUMNS:,} cột mỗi lần tải lên.")

    if has_header:
        headers = [
            header if header else f"Cột {index + 1}"
            for index, header in enumerate(headers)
        ]
        headers.extend(f"Cột {index + 1}" for index in range(len(headers), column_count))
    else:
        headers = [f"Cột {index + 1}" for index in range(column_count)]

    normalized_rows = [
        row + [""] * (column_count - len(row))
        for row in data_rows
    ]
    inconsistent_rows = sum(len(row) != column_count for row in data_rows)
    column_values = [
        [row[index] for row in normalized_rows]
        for index in range(column_count)
    ]
    all_numeric = all(
        _number_or_none(value) is not None
        for row in normalized_rows
        for value in row
    )
    statistics = [
        _column_statistics(headers[index], values)
        for index, values in enumerate(column_values)
    ]
    missing_cells = sum(stat["missing"] for stat in statistics)
    non_numeric_cells = sum(stat["non_numeric"] for stat in statistics)
    duplicate_rows = len(normalized_rows) - len({
        tuple(row) for row in normalized_rows
    })
    constant_columns = [
        stat["name"] for stat in statistics
        if stat["unique"] == 1 and stat["missing"] == 0
    ]
    numeric_columns = sum(stat["type"] == "numeric" for stat in statistics)
    text_columns = sum(stat["type"] == "text" for stat in statistics)
    mixed_columns = sum(stat["type"] == "mixed" for stat in statistics)

    total_cells = len(normalized_rows) * column_count
    insights = []
    if missing_cells:
        insights.append(
            f"Có {missing_cells:,} ô trống hoặc giá trị thiếu "
            f"({missing_cells / total_cells:.1%} tổng số ô)."
        )
    if non_numeric_cells:
        insights.append(
            f"Có {non_numeric_cells:,} ô chứa dữ liệu không phải số "
            "(có thể là dữ liệu phân loại hoặc cần làm sạch)."
        )
    if duplicate_rows:
        insights.append(f"Có {duplicate_rows:,} dòng trùng hoàn toàn.")
    if constant_columns:
        names = ", ".join(constant_columns[:5])
        suffix = "…" if len(constant_columns) > 5 else ""
        insights.append(
            f"Có {len(constant_columns)} cột chỉ có một giá trị: {names}{suffix}."
        )
    if inconsistent_rows:
        insights.append(
            f"Có {inconsistent_rows:,} dòng có số cột lệch so với bảng; "
            "các ô thiếu đã được tính vào thống kê."
        )
    if not insights:
        insights.append(
            "Không phát hiện ô thiếu, dòng trùng hoặc cột không biến thiên."
        )

    return {
        "rows": normalized_rows,
        "headers": headers,
        "columns": column_count,
        "has_header": has_header,
        "all_numeric": all_numeric,
        "statistics": statistics,
        "summary": {
            "missing_cells": missing_cells,
            "non_numeric_cells": non_numeric_cells,
            "duplicate_rows": duplicate_rows,
            "constant_columns": len(constant_columns),
            "numeric_columns": numeric_columns,
            "text_columns": text_columns,
            "mixed_columns": mixed_columns,
            "inconsistent_rows": inconsistent_rows,
            "missing_percent": round(missing_cells / total_cells * 100, 2),
        },
        "insights": insights,
    }


def _normalized_name(value):
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _labeled_har_data(dataset, class_names):
    if not dataset["has_header"]:
        return None

    label_indexes = [
        index for index, name in enumerate(dataset["headers"])
        if _normalized_name(name) in LABEL_COLUMN_NAMES
    ]
    if len(label_indexes) != 1:
        return None

    label_index = label_indexes[0]
    feature_indexes = [
        index for index in range(dataset["columns"])
        if index != label_index
    ]
    if len(feature_indexes) != FEATURE_COUNT:
        return None

    label_to_index = {
        _normalized_name(name): index
        for index, name in enumerate(class_names)
    }
    labels = []
    features = []
    for row in dataset["rows"]:
        label_value = row[label_index].strip()
        normalized_label = _normalized_name(label_value)
        if normalized_label in label_to_index:
            label = label_to_index[normalized_label]
        else:
            try:
                numeric_label = float(label_value)
            except ValueError:
                return None
            if not numeric_label.is_integer():
                return None
            activity_id = int(numeric_label)
            if not 1 <= activity_id <= len(class_names):
                return None
            label = activity_id - 1

        values = [_number_or_none(row[index]) for index in feature_indexes]
        if any(value is None for value in values):
            return None
        labels.append(label)
        features.append(values)

    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=int)


def _classification_metrics(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(np.mean(precision)), 6),
        "macro_recall": round(float(np.mean(recall)), 6),
        "macro_f1": round(float(f1_score(
            y_true, y_pred, labels=np.arange(len(class_names)),
            average="macro", zero_division=0,
        )), 6),
        "per_class": [
            {
                "label": class_names[index],
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index in range(len(class_names))
        ],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(len(class_names))
        ).tolist(),
    }


def _feature_histograms(dataset, excluded_indexes):
    candidates = [
        (index, stat["std"])
        for index, stat in enumerate(dataset["statistics"])
        if index not in excluded_indexes and stat["type"] == "numeric" and "std" in stat
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)

    histograms = []
    for index, _ in candidates[:3]:
        values = [
            float(row[index])
            for row in dataset["rows"]
            if _number_or_none(row[index]) is not None
        ]
        if not values:
            continue
        counts, edges = np.histogram(values, bins=min(8, max(1, len(set(values)))))
        histograms.append({
            "name": dataset["headers"][index],
            "bins": [
                {
                    "range": f"{edges[bin_index]:.3g}–{edges[bin_index + 1]:.3g}",
                    "count": int(counts[bin_index]),
                }
                for bin_index in range(len(counts))
            ],
        })
    return histograms


def _class_counts(values, class_names):
    counts = np.bincount(values, minlength=len(class_names))
    return [
        {"label": class_names[index], "count": int(counts[index])}
        for index in range(len(class_names))
    ]


@lru_cache(maxsize=2)
def get_model_bundle(name="mlp"):
    model_path = {"mlp": MODEL_PATH, "lr": LR_MODEL_PATH}.get(name)
    if model_path is None:
        raise ValueError(f"Unknown HAR model: {name}")
    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Không tìm thấy mô hình {name.upper()}: {model_path}",
        )

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "class_names" not in bundle:
        raise RuntimeError("Tệp mô hình không đúng định dạng HAR.")
    if not hasattr(bundle["model"], "predict_proba"):
        raise RuntimeError("Mô hình HAR không hỗ trợ dự đoán xác suất.")
    return bundle


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_PATH, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    bundle = get_model_bundle()
    return {
        "status": "ok",
        "model": "MLP",
        "models": ["MLP", "Logistic Regression"],
        "classes": bundle["class_names"],
    }


@app.get("/api/sample", include_in_schema=False)
def sample_csv():
    if not SAMPLE_PATH.is_file():
        raise HTTPException(status_code=404, detail="File CSV mẫu chưa được cài đặt.")
    return FileResponse(
        SAMPLE_PATH,
        media_type="text/csv",
        filename="har_test_sample.csv",
    )


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.casefold() != ".csv":
        raise HTTPException(status_code=400, detail="Vui lòng tải lên tệp .csv.")

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Tệp vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        dataset = parse_csv_data(contents)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await file.close()

    response = {
        "mode": "analysis",
        "rows": len(dataset["rows"]),
        "columns": dataset["columns"],
        "summary": dataset["summary"],
        "insights": dataset["insights"],
        "column_names": dataset["headers"],
        "preview": [
            row[:PREVIEW_COLUMNS]
            for row in dataset["rows"][:PREVIEW_ROWS]
        ],
        "preview_rows": min(len(dataset["rows"]), PREVIEW_ROWS),
        "preview_columns": min(dataset["columns"], PREVIEW_COLUMNS),
        "statistics": dataset["statistics"],
        "results": [],
    }

    label_index = next((
        index for index, name in enumerate(dataset["headers"])
        if dataset["has_header"] and _normalized_name(name) in LABEL_COLUMN_NAMES
    ), None)
    excluded_indexes = {label_index} if label_index is not None else set()
    response["histograms"] = _feature_histograms(dataset, excluded_indexes)
    is_unlabeled_har = dataset["columns"] == FEATURE_COUNT and dataset["all_numeric"]
    is_labeled_har_shape = (
        dataset["columns"] == FEATURE_COUNT + 1 and label_index is not None
    )

    if not is_unlabeled_har and not is_labeled_har_shape:
        response["message"] = (
            f"Đã phân tích CSV gồm {dataset['columns']} cột. "
            "Để đánh giá mô hình, thêm cột activity/label cùng đúng 561 đặc trưng số."
        )
        return response

    mlp_bundle = get_model_bundle()
    class_names = mlp_bundle["class_names"]
    labeled_data = _labeled_har_data(dataset, class_names)
    if is_labeled_har_shape and labeled_data is None:
        response["message"] = (
            "Đã phân tích dữ liệu nhưng chưa đánh giá được: cột activity/label "
            "phải chứa tên một trong sáu hoạt động HAR hoặc mã từ 1 đến 6; "
            "561 đặc trưng còn lại phải là số và không được thiếu."
        )
        return response

    if labeled_data is None:
        features = np.asarray(dataset["rows"], dtype=np.float64)
        y_true = None
    else:
        features, y_true = labeled_data

    model_outputs = {}
    predictions_by_model = {}
    for model_name, display_name in (("mlp", "MLP"), ("lr", "Logistic Regression")):
        bundle = mlp_bundle if model_name == "mlp" else get_model_bundle(model_name)
        if bundle["class_names"] != class_names:
            raise RuntimeError("Các mô hình HAR không dùng chung thứ tự nhãn.")

        model = bundle["model"]
        predictions = model.predict(features).astype(int)
        probabilities = model.predict_proba(features)
        predictions_by_model[model_name] = predictions
        model_outputs[model_name] = {
            "name": display_name,
            "predicted_counts": _class_counts(predictions, class_names),
            "metrics": (
                _classification_metrics(y_true, predictions, class_names)
                if y_true is not None else None
            ),
            "predictions": predictions,
            "confidence": probabilities.max(axis=1),
            "probabilities": probabilities,
        }

    agreement = float(np.mean(
        predictions_by_model["mlp"] == predictions_by_model["lr"]
    ))
    response["mode"] = "evaluation" if y_true is not None else "prediction"
    response["model"] = "MLP"
    response["models"] = ["MLP", "Logistic Regression"]
    response["message"] = (
        "Đã so sánh hai mô hình với nhãn thật trong CSV."
        if y_true is not None
        else "Đã so sánh dự đoán của MLP và Logistic Regression."
    )
    response["class_names"] = class_names
    response["class_distribution"] = (
        _class_counts(y_true, class_names)
        if y_true is not None else None
    )
    response["model_agreement"] = round(agreement, 6)
    response["model_metrics"] = {
        name: output["metrics"] for name, output in model_outputs.items()
    }
    response["predicted_distributions"] = {
        name: output["predicted_counts"] for name, output in model_outputs.items()
    }
    response["predictions"] = [
        {
            "row": index + 1,
            "actual": class_names[int(y_true[index])] if y_true is not None else None,
            "mlp": class_names[int(predictions_by_model["mlp"][index])],
            "mlp_confidence": round(float(model_outputs["mlp"]["confidence"][index]), 6),
            "logistic_regression": class_names[int(predictions_by_model["lr"][index])],
            "lr_confidence": round(float(model_outputs["lr"]["confidence"][index]), 6),
        }
        for index in range(len(features))
    ]
    mlp_model = model_outputs["mlp"]
    response["results"] = [
        {
            "row": index + 1,
            "prediction": class_names[int(mlp_model["predictions"][index])],
            "confidence": round(float(mlp_model["confidence"][index]), 6),
            "probabilities": {
                class_names[int(class_id)]: round(float(mlp_model["probabilities"][index, column]), 6)
                for column, class_id in enumerate(mlp_bundle["model"].classes_)
            },
        }
        for index in range(len(features))
    ]
    return response
