"""API contract checks; inference checks use the real trained artifacts."""
import unittest
from fastapi.testclient import TestClient
from app.main import app


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_empty_review_is_rejected(self):
        for text in ["", "   ", "x" * 20001]:
            self.assertEqual(self.client.post("/api/predict", json={"text": text}).status_code, 422)

    def test_paired_predictions_are_real_probabilities(self):
        for text, expected in [("A wonderful, moving film with brilliant acting. I loved every minute.", "positive"),
                               ("An awful, boring film. Terrible acting and a complete waste of time.", "negative")]:
            response = self.client.post("/api/predict", json={"text": text})
            self.assertEqual(response.status_code, 200, response.text)
            rows = response.json()["predictions"]
            self.assertEqual({row["id"] for row in rows}, {"baseline", "transformer"})
            for row in rows:
                self.assertEqual(row["label"], expected)
                self.assertAlmostEqual(sum(row["probabilities"].values()), 1, places=5)
                self.assertGreaterEqual(row["confidence"], .5)
                self.assertLessEqual(row["confidence"], 1)

    def test_report_counts_match_test_split(self):
        response = self.client.get("/api/report")
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertEqual(len(report["models"]), 2)
        for model in report["models"]:
            cm = model["confusion_matrix"]
            count = sum(sum(row) for row in cm)
            self.assertEqual(count, report["dataset"]["test_count"])
            self.assertAlmostEqual((cm[0][0] + cm[1][1]) / count, model["accuracy"])


if __name__ == "__main__":
    unittest.main()
