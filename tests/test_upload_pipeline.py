"""
Unit tests for src/upload_pipeline.py - CSV upload -> forecast + reorder alerts.

Mirrors tests/test_module6.py's pattern: builds a temporary mock predictions
CSV and points reorder_logic.PREDICTIONS_PATH at it, so tests don't depend on
or mutate the real data/processed/hybrid_predictions.csv.
"""

import shutil
import tempfile
import unittest
from io import StringIO
from pathlib import Path

import pandas as pd

import src.reorder_logic as reorder_logic
import src.upload_pipeline as upload_pipeline


class TestUploadPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_predictions_path = Path(cls.temp_dir) / "test_predictions.csv"

        # Store 4/Dept 92 and Store 4/Dept 95 exist in the "trained catalog";
        # Store 9/Dept 999 does not, to exercise the unmatched-pairs path.
        mock_data = [
            {"store_id": 4, "dept_id": 92, "date": "2012-08-24", "weekly_sales": 20000.0, "prediction": 22140.5, "is_holiday": False},
            {"store_id": 4, "dept_id": 92, "date": "2012-08-31", "weekly_sales": 21000.0, "prediction": 23000.0, "is_holiday": False},
            {"store_id": 4, "dept_id": 95, "date": "2012-08-24", "weekly_sales": 10000.0, "prediction": 12000.0, "is_holiday": False},
        ]
        pd.DataFrame(mock_data).to_csv(cls.temp_predictions_path, index=False)

        cls.orig_predictions_path = reorder_logic.PREDICTIONS_PATH
        reorder_logic.PREDICTIONS_PATH = cls.temp_predictions_path

    @classmethod
    def tearDownClass(cls):
        reorder_logic.PREDICTIONS_PATH = cls.orig_predictions_path
        shutil.rmtree(cls.temp_dir)

    def _sample_upload_csv(self) -> pd.DataFrame:
        csv_text = (
            "Store,Dept,Date,Weekly_Sales\n"
            "4,92,2012-07-27,19000\n"
            "4,92,2012-08-03,19500\n"
            "4,92,2012-08-10,20500\n"
            "4,92,2012-08-17,20800\n"
            "9,999,2012-08-17,500\n"
        )
        return pd.read_csv(StringIO(csv_text))

    # --- validate_and_prepare ---

    def test_validate_missing_required_columns(self):
        df = pd.DataFrame({"Store": [4], "Dept": [92]})
        with self.assertRaises(ValueError) as ctx:
            upload_pipeline.validate_and_prepare(df)
        self.assertIn("Missing required column(s)", str(ctx.exception))
        self.assertIn("Date", str(ctx.exception))
        self.assertIn("Weekly_Sales", str(ctx.exception))

    def test_validate_zero_valid_rows(self):
        df = pd.DataFrame({
            "Store": ["not_a_number"],
            "Dept": [92],
            "Date": ["2012-08-24"],
            "Weekly_Sales": [1000.0],
        })
        with self.assertRaises(ValueError) as ctx:
            upload_pipeline.validate_and_prepare(df)
        self.assertIn("no valid rows", str(ctx.exception))

    def test_validate_drops_bad_rows_keeps_good_ones(self):
        df = pd.DataFrame({
            "Store": [4, "bad"],
            "Dept": [92, 92],
            "Date": ["2012-08-24", "2012-08-31"],
            "Weekly_Sales": [1000.0, 2000.0],
        })
        prepared = upload_pipeline.validate_and_prepare(df)
        self.assertEqual(len(prepared), 1)

    def test_validate_defaults_missing_optional_columns(self):
        df = self._sample_upload_csv()
        prepared = upload_pipeline.validate_and_prepare(df)
        self.assertIn("IsHoliday", prepared.columns)
        self.assertFalse(prepared["IsHoliday"].any())
        for i in range(1, 6):
            self.assertIn(f"MarkDown{i}", prepared.columns)

    # --- engineer_for_display ---

    def test_engineer_for_display_computes_lag_and_calendar_features(self):
        df = self._sample_upload_csv()
        prepared = upload_pipeline.validate_and_prepare(df)
        engineered = upload_pipeline.engineer_for_display(prepared)
        self.assertIn("weekly_sales_roll_mean_4", engineered.columns)
        self.assertIn("month", engineered.columns)
        self.assertIn("is_negative_sales", engineered.columns)
        # Store 4/Dept 92's rows are 19000, 19500, 20500, 20800 (sorted by date) --
        # the last row's lag_1 should be the previous row's own weekly_sales (20500).
        dept_92_rows = engineered[(engineered["store_id"] == 4) & (engineered["dept_id"] == 92)]
        self.assertEqual(len(dept_92_rows), 4)
        self.assertEqual(dept_92_rows["weekly_sales_lag_1"].iloc[-1], 20500.0)

    # --- match_pairs ---

    def test_match_pairs_splits_matched_and_unmatched(self):
        pairs = [(4, 92), (4, 95), (9, 999)]
        matched_df, unmatched = upload_pipeline.match_pairs(pairs, as_of_date="2026-08-21")
        self.assertEqual(len(matched_df), 2)
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched[0]["store_id"], 9)
        self.assertEqual(unmatched[0]["dept_id"], 999)
        self.assertIn("reason", unmatched[0])

    # --- process_upload end to end ---

    def test_process_upload_end_to_end(self):
        df = self._sample_upload_csv()
        result = upload_pipeline.process_upload(df, as_of_date="2026-08-21")

        self.assertEqual(result["rows_received"], 5)
        self.assertEqual(result["matched_pairs"], 1)  # only (4, 92) is in the mock catalog
        self.assertEqual(len(result["unmatched_pairs"]), 1)
        self.assertEqual(result["unmatched_pairs"][0]["store_id"], 9)

        self.assertEqual(len(result["forecasts"]), 1)
        forecast = result["forecasts"][0]
        self.assertEqual(forecast["store_id"], "4")
        self.assertEqual(forecast["dept_id"], "92")
        self.assertEqual(forecast["week_ending_date"], "2026-08-21")
        self.assertEqual(forecast["predicted_weekly_sales"], 22140.5)

        self.assertEqual(len(result["reorder_alerts"]), 1)
        self.assertEqual(result["reorder_alerts"][0]["store_id"], "4")
        self.assertEqual(result["reorder_alerts"][0]["dept_id"], "92")
        self.assertIn(result["reorder_alerts"][0]["urgency"], ("red", "amber", "green"))

        self.assertIn("total_depts_at_risk", result["summary"])
        self.assertIn("total_capital_freed_estimate", result["summary"])

        self.assertEqual(result["upload_summary"]["unique_store_dept_pairs"], 2)
        self.assertEqual(result["upload_summary"]["rows_with_negative_sales"], 0)

    def test_process_upload_raises_when_nothing_matches(self):
        df = pd.DataFrame({
            "Store": [9],
            "Dept": [999],
            "Date": ["2012-08-24"],
            "Weekly_Sales": [500.0],
        })
        with self.assertRaises(ValueError) as ctx:
            upload_pipeline.process_upload(df)
        self.assertIn("unmatched", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
