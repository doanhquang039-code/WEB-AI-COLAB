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

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get(
    "HAR_MODEL_PATH",
    APP_DIR / "outputs" / "mlp_har.joblib",
))
INDEX_PATH = APP_DIR / "index.html"
FEATURE_COUNT = 561
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_ROWS = 10_000
MAX_COLUMNS = 1_000
PREVIEW_ROWS = 20
PREVIEW_COLUMNS = 20

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
    if non_numeric_count == 0:
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


@lru_cache(maxsize=1)
def get_model_bundle():
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Không tìm thấy mô hình đã huấn luyện: {MODEL_PATH}",
        )

    bundle = joblib.load(MODEL_PATH)
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
    return {"status": "ok", "model": "MLP", "classes": bundle["class_names"]}


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

    if dataset["columns"] != FEATURE_COUNT or not dataset["all_numeric"]:
        response["message"] = (
            f"Đã phân tích CSV gồm {dataset['columns']} cột. "
            "Mô hình HAR chỉ dự đoán khi dữ liệu có đúng 561 đặc trưng số."
        )
        return response

    features = np.asarray(dataset["rows"], dtype=np.float64)
    bundle = get_model_bundle()
    model = bundle["model"]
    class_names = bundle["class_names"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    model_classes = model.classes_

    results = []
    for index, prediction in enumerate(predictions):
        predicted_class = int(prediction)
        if predicted_class < 0 or predicted_class >= len(class_names):
            raise RuntimeError("Mô hình trả về mã lớp không hợp lệ.")

        probability_by_class = {
            class_names[int(model_class)]: round(float(probabilities[index, column]), 6)
            for column, model_class in enumerate(model_classes)
        }
        results.append({
            "row": index + 1,
            "prediction": class_names[predicted_class],
            "confidence": round(float(probabilities[index].max()), 6),
            "probabilities": probability_by_class,
        })

    response["mode"] = "prediction"
    response["model"] = "MLP"
    response["message"] = "CSV có 561 đặc trưng số; đã chạy dự đoán HAR."
    response["results"] = results
    return response
