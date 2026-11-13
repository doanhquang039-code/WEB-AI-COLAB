import csv
import io
import unittest

import numpy as np
from fastapi.testclient import TestClient

from web_app import app, get_model_bundle


def make_csv(rows, header=False):
    output = io.StringIO()
    writer = csv.writer(output)
    if header:
        writer.writerow([f"feature_{index}" for index in range(len(rows[0]))])
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


class WebAppTests(unittest.TestCase):
    def test_prediction_accepts_header_and_headerless_561_feature_csv(self):
        samples = [[float(index) / 1000 for index in range(561)]]

        with TestClient(app) as client:
            with_header = client.post(
                "/api/predict",
                files={"file": ("sample.csv", make_csv(samples, header=True), "text/csv")},
            )
            without_header = client.post(
                "/api/predict",
                files={"file": ("sample.csv", make_csv(samples), "text/csv")},
            )

        self.assertEqual(with_header.status_code, 200)
        self.assertEqual(without_header.status_code, 200)
        self.assertEqual(with_header.json()["mode"], "prediction")
        self.assertEqual(without_header.json()["mode"], "prediction")

    def test_100_column_csv_returns_preview_and_statistics_without_prediction(self):
        samples = [
            [float(row * 100 + column) for column in range(100)]
            for row in range(5)
        ]

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                files={"file": ("sample.csv", make_csv(samples), "text/csv")},
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["mode"], "analysis")
        self.assertEqual(result["rows"], 5)
        self.assertEqual(result["columns"], 100)
        self.assertEqual(len(result["preview"]), 5)
        self.assertEqual(result["statistics"][0]["min"], 0)
        self.assertEqual(result["statistics"][0]["max"], 400)
        self.assertEqual(result["statistics"][0]["median"], 200)
        self.assertEqual(result["summary"]["numeric_columns"], 100)
        self.assertEqual(result["summary"]["missing_cells"], 0)
        self.assertEqual(result["results"], [])

    def test_analysis_reports_missing_and_text_values(self):
        contents = b"name,value\nalpha,1\nbeta,\ngamma,abc\n"

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                files={"file": ("sample.csv", contents, "text/csv")},
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["mode"], "analysis")
        self.assertEqual(result["statistics"][1]["missing"], 1)
        self.assertEqual(result["statistics"][0]["type"], "text")
        self.assertEqual(result["statistics"][1]["type"], "mixed")
        self.assertEqual(result["summary"]["non_numeric_cells"], 4)

    def test_analysis_detects_duplicate_rows_and_numeric_outliers(self):
        contents = b"value,label\n1,a\n2,b\n3,c\n100,d\n1,a\n"

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                files={"file": ("sample.csv", contents, "text/csv")},
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        value_stats = result["statistics"][0]
        self.assertEqual(result["summary"]["duplicate_rows"], 1)
        self.assertEqual(value_stats["median"], 2)
        self.assertEqual(value_stats["outliers"], 1)
        self.assertTrue(any("dòng trùng" in insight for insight in result["insights"]))

    def test_empty_and_header_only_csv_are_rejected(self):
        with TestClient(app) as client:
            empty = client.post(
                "/api/predict",
                files={"file": ("empty.csv", b"", "text/csv")},
            )
            header_only = client.post(
                "/api/predict",
                files={"file": ("header.csv", b"name,value\n", "text/csv")},
            )

        self.assertEqual(empty.status_code, 422)
        self.assertEqual(header_only.status_code, 422)

    def test_prediction_matches_saved_mlp(self):
        features = np.zeros((1, 561), dtype=np.float64)
        bundle = get_model_bundle()
        model = bundle["model"]
        expected_prediction = int(model.predict(features)[0])
        expected_probabilities = model.predict_proba(features)[0]

        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                files={"file": ("sample.csv", make_csv(features.tolist()), "text/csv")},
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()["results"][0]
        self.assertEqual(result["prediction"], bundle["class_names"][expected_prediction])
        self.assertEqual(result["confidence"], round(float(expected_probabilities.max()), 6))


if __name__ == "__main__":
    unittest.main()
