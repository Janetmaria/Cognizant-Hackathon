"""
Unit Tests for Module 6 - Inventory Intelligence & Reorder Alert Engine.

Tests date mapping, simulated price/stock formulas, reorder calculations,
urgency boundaries, and markdown what-if lift simulations, using a mock temporary predictions CSV.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

import src.reorder_logic as reorder_logic

class TestModule6(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Create a temporary predictions file to test our calculations against
        without mutating the store predictions on disk.
        """
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_predictions_path = Path(cls.temp_dir) / "test_predictions.csv"
        
        # Build mock dataframe: 1 store (4), 3 departments (92, 95, 10) representing Red, Amber, Green states.
        # Holdout date range: 2012-08-24 to 2012-09-07.
        mock_data = [
            # Store 4, Dept 92 (High Demand, Red State simulation target)
            {"store_id": 4, "dept_id": 92, "date": "2012-08-24", "weekly_sales": 20000.0, "prediction": 22140.5, "stat_prediction": 21000.0, "is_holiday": False},
            {"store_id": 4, "dept_id": 92, "date": "2012-08-31", "weekly_sales": 21000.0, "prediction": 23000.0, "stat_prediction": 22500.0, "is_holiday": False},
            {"store_id": 4, "dept_id": 92, "date": "2012-09-07", "weekly_sales": 22000.0, "prediction": 24000.0, "stat_prediction": 23000.0, "is_holiday": True},
            
            # Store 4, Dept 95 (Medium Demand, Amber/Green State simulation target)
            {"store_id": 4, "dept_id": 95, "date": "2012-08-24", "weekly_sales": 10000.0, "prediction": 12000.0, "stat_prediction": 11000.0, "is_holiday": False},
            
            # Store 4, Dept 10 (Zero Demand placeholder)
            {"store_id": 4, "dept_id": 10, "date": "2012-08-24", "weekly_sales": 0.0, "prediction": 0.0, "stat_prediction": 0.0, "is_holiday": False}
        ]
        
        df = pd.DataFrame(mock_data)
        df.to_csv(cls.temp_predictions_path, index=False)
        
        # Override the logic predictions path to point to our mock file
        cls.orig_predictions_path = reorder_logic.PREDICTIONS_PATH
        reorder_logic.PREDICTIONS_PATH = cls.temp_predictions_path
        
    @classmethod
    def tearDownClass(cls):
        """
        Restore original paths and clean up the temp folder.
        """
        reorder_logic.PREDICTIONS_PATH = cls.orig_predictions_path
        shutil.rmtree(cls.temp_dir)
        
    def test_date_mapping_backward(self):
        """
        Checks that present-day query dates (2026) map to historical holdout Fridays (2012).
        For example: Friday 2026-08-21 -> Friday 2012-08-24.
        """
        mapped = reorder_logic.map_date_backward("2026-08-21")
        self.assertEqual(mapped, "2012-08-24")
        
    def test_date_mapping_forward(self):
        """
        Checks that historical dates (2012) map back to 2026 dates for visual reporting.
        """
        mapped = reorder_logic.map_date_forward("2012-08-24")
        self.assertEqual(mapped, "2026-08-21")
        
    def test_simulated_unit_price(self):
        """
        Assures that price assignment is deterministic and returns values inside correct bounds.
        """
        price_92 = reorder_logic.get_simulated_unit_price("92")
        price_95 = reorder_logic.get_simulated_unit_price("95")
        
        # Verify pricing ranges
        self.assertTrue(5.50 <= price_92 <= 49.50)
        self.assertTrue(5.50 <= price_95 <= 49.50)
        
        # Assert determinism - same department yields identical price every call
        self.assertEqual(price_92, reorder_logic.get_simulated_unit_price("92"))
        
    def test_simulated_stock_units(self):
        """
        Verifies simulated stock level sizing relative to weekly unit demand.
        """
        demand = 100.0
        stock_units = reorder_logic.get_simulated_stock_units("4", "92", demand)
        
        # Stock should be scaled by the deterministic multiplier (multiplier * demand)
        # Store 4, Dept 92 multiplier = (4 * 7 + 92 * 13) % 100 / 100 * 2.5
        # = (28 + 1196) % 100 / 100 * 2.5 = 1224 % 100 / 100 * 2.5 = 24 / 100 * 2.5 = 0.24 * 2.5 = 0.6
        # Expected stock = 100.0 * 0.6 = 60 units
        self.assertEqual(stock_units, 60)
        
        # Zero demand fallback stock check
        self.assertEqual(reorder_logic.get_simulated_stock_units("4", "92", 0.0), 10)
        
    def test_reorder_alerts_math(self):
        """
        Performs end-to-end mathematical audit on reorder alert outputs.
        """
        results = reorder_logic.get_reorder_alerts(store_id="4", as_of_date="2026-08-21")
        
        self.assertEqual(results["store_id"], "4")
        self.assertEqual(results["generated_at"], "2026-08-21")
        
        # Locate department 92 details:
        # Forecast = 22140.5 (from setUpClass)
        # Price = (92 * 31 % 45) + 5.5 = (2852 % 45) + 5.5 = 17 + 5.5 = 22.50
        # Weekly Demand Units = 22140.5 / 22.50 = 984.022 units
        # Stock level = int(984.022 * 0.6) = 590 units
        # Weeks to stockout = 590 / 984.022 = 0.599 (rounding check: 0.6 weeks)
        # Urgency should be Red (<= 1.0 week)
        # Target Safety Units (2 weeks demand) = ceil(2 * 984.022) = 1969 units
        # Recommended Reorder = 1969 - 590 = 1379 units
        # Optimal Max (4 weeks demand) = ceil(4 * 984.022) = 3937 units
        # Capital freed estimate = 0.0 (since stock 590 < optimal max 3937)
        
        dept_92_alerts = [a for a in results["alerts"] if a["dept_id"] == "92"]
        self.assertEqual(len(dept_92_alerts), 1)
        alert = dept_92_alerts[0]
        
        self.assertEqual(alert["urgency"], "red")
        self.assertEqual(alert["recommended_reorder_units"], 1969 - 590) # 1379
        self.assertEqual(alert["capital_freed_estimate"], 0.0)
        
    def test_whatif_promo_saturation_and_holiday_logic(self):
        """
        Validates `/whatif` promo lift calculation and holiday override multiplier logic.
        """
        # Test markdown lift for $5000:
        # Lift % = 30 * (5000 / 7000) = 21.43%
        # Date range covers 2026-08-21 (historical 2012-08-24) to 2026-09-04 (historical 2012-09-07)
        # Days = 3 weeks.
        
        # 1. No overrides, no markdown
        whatif_zero = reorder_logic.simulate_whatif(
            store_id="4", dept_id="92",
            start_date="2026-08-21", end_date="2026-09-04",
            markdown_amount=0.0, is_holiday_override=None
        )
        self.assertEqual(whatif_zero["projected_lift_pct"], 0.0)
        
        # 2. Markdown amount = 5000.0, which should generate ~21.43% lift
        whatif_promo = reorder_logic.simulate_whatif(
            store_id="4", dept_id="92",
            start_date="2026-08-21", end_date="2026-09-04",
            markdown_amount=5000.0, is_holiday_override=None
        )
        # Since week 3 (2012-09-07) is historically a holiday (holiday = True),
        # and we passed is_holiday_override=None, the holiday status remains True.
        # Thus, lift is purely from the markdown change (21.43%)
        self.assertAlmostEqual(whatif_promo["projected_lift_pct"], 21.43, places=1)
        
        # 3. Holiday Override flag = False on a week that was originally a holiday (week 3: 2026-09-04)
        # Should remove the 7.4% holiday lift on that week
        # Let's verify it decreases the overall lift compared to markdown-only
        whatif_override = reorder_logic.simulate_whatif(
            store_id="4", dept_id="92",
            start_date="2026-09-04", end_date="2026-09-04", # only week 3 (holiday week)
            markdown_amount=0.0, is_holiday_override=False
        )
        # Baseline sales (originally holiday) = 24000.0
        # Adjusted sales (override to non-holiday) = 24000 / 1.074 = 22346.37
        # Lift pct = (22346.37 - 24000) / 24000 * 100 = -6.89%
        self.assertAlmostEqual(whatif_override["projected_lift_pct"], -6.89, places=1)
        
    def test_whatif_promo_baseline_hybrid(self):
        """
        Validates `/whatif` baseline model queries the `stat_prediction` column.
        """
        whatif_baseline = reorder_logic.simulate_whatif(
            store_id="4", dept_id="92",
            start_date="2026-08-21", end_date="2026-08-21",
            markdown_amount=0.0, is_holiday_override=None,
            model="baseline"
        )
        self.assertEqual(len(whatif_baseline["baseline_predictions"]), 1)
        # Expected baseline sales for store 4, dept 92 on 2012-08-24 is 21000.0 (from stat_prediction)
        self.assertEqual(whatif_baseline["baseline_predictions"][0]["predicted_weekly_sales"], 21000.0)

if __name__ == "__main__":
    unittest.main()
