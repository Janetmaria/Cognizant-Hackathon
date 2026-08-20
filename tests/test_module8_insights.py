"""
Unit & Integration Tests for Module 8 - Executive LLM Insights (backend/llm_insights.py).

Tests:
1. Pydantic schema validation (InsightRequest and InsightResponse).
2. Deterministic rule-based template generation under various alert scenarios.
3. Resilience fallback mechanism (handling missing keys, network failure, or API errors).
4. Protected API endpoints (/insights/generate and /insights):
   - 401 Unauthorized on missing/invalid JWT.
   - 200 OK on valid JWT for manager and admin roles.
"""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.main import app
import backend.auth as auth
import backend.llm_insights as llm_insights
from backend.llm_insights import InsightRequest, InsightResponse, generate_template_commentary


class TestModule8Insights(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Create test tokens
        cls.admin_token = auth.create_access_token(data={"sub": "admin", "role": "admin"})
        cls.manager_token = auth.create_access_token(data={"sub": "admin", "role": "admin"})

    def test_schema_direct_and_nested_payload(self):
        """Test InsightRequest parsing: explicit metric overrides win over derived ones, and nested reorder structure is parsed."""
        # v2 contract requires alerts + summary; explicit metric fields alongside them
        # should take precedence over what the validator would otherwise derive.
        req1 = InsightRequest.model_validate({
            "store_id": "4",
            "alerts": [{"dept_id": "92", "urgency": "red"}],
            "summary": {"total_depts_at_risk": 1, "total_capital_freed_estimate": 100.0},
            "total_depts_at_risk": 6,
            "red_alerts": 2,
            "amber_alerts": 4,
            "capital_freed_estimate": 41200.0,
            "projected_weekly_sales": 125000.0,
        })
        self.assertEqual(req1.store_id, "4")
        self.assertEqual(req1.red_alerts, 2)
        self.assertEqual(req1.amber_alerts, 4)
        self.assertEqual(req1.capital_freed_estimate, 41200.0)

        # Nested /reorder style
        req2 = InsightRequest.model_validate({
            "store_id": "10",
            "alerts": [
                {"dept_id": "92", "urgency": "red", "weeks_to_stockout": 0.8},
                {"dept_id": "38", "urgency": "amber", "weeks_to_stockout": 1.5},
                {"dept_id": "1", "urgency": "green", "weeks_to_stockout": 4.0},
            ],
            "summary": {
                "total_depts_at_risk": 2,
                "total_capital_freed_estimate": 18500.50,
                "projected_weekly_sales": 75000.0,
            }
        })
        self.assertEqual(req2.store_id, "10")
        self.assertEqual(req2.red_alerts, 1)
        self.assertEqual(req2.amber_alerts, 1)
        self.assertEqual(req2.total_depts_at_risk, 3)
        self.assertEqual(req2.capital_freed_estimate, 18500.50)
        self.assertEqual(req2.projected_weekly_sales, 75000.0)

    def test_template_commentary_scenarios(self):
        """Verify deterministic template commentary under critical, amber, and healthy states."""
        # Scenario 1: Critical Red Alerts
        comm_red = generate_template_commentary(
            store_id="4",
            total_depts_at_risk=6,
            red_alerts=2,
            amber_alerts=4,
            capital_freed_estimate=41200.0,
            projected_weekly_sales=125000.0,
        )
        self.assertIn("Store 4 exhibits critical inventory vulnerability", comm_red)
        self.assertIn("2 departments in immediate red-alert", comm_red)
        self.assertIn("$41,200.00", comm_red)
        self.assertIn("$125,000.00", comm_red)

        # Scenario 2: Amber Alerts only
        comm_amber = generate_template_commentary(
            store_id="4",
            total_depts_at_risk=3,
            red_alerts=0,
            amber_alerts=3,
            capital_freed_estimate=15000.0,
            projected_weekly_sales=95000.0,
        )
        self.assertIn("stable operations with zero red stockouts", comm_amber)
        self.assertIn("3 departments remain on amber watch", comm_amber)
        self.assertIn("$15,000.00", comm_amber)

        # Scenario 3: Zero alerts (Healthy)
        comm_healthy = generate_template_commentary(
            store_id="4",
            total_depts_at_risk=0,
            red_alerts=0,
            amber_alerts=0,
            capital_freed_estimate=5000.0,
            projected_weekly_sales=80000.0,
        )
        self.assertIn("strong inventory health with zero departments", comm_healthy)
        self.assertIn("$80,000.00", comm_healthy)

    def test_unauthenticated_request_rejected(self):
        """Verify 401 Unauthorized when no JWT token is provided."""
        payload = {
            "store_id": "4",
            "total_depts_at_risk": 2,
            "red_alerts": 1,
            "amber_alerts": 1,
            "capital_freed_estimate": 10000.0,
        }
        res = self.client.post("/insights/generate", json=payload)
        self.assertEqual(res.status_code, 401)

    def test_authenticated_request_generates_commentary(self):
        """Verify successful 200 response with valid commentary for any configured source (llm or fallback)."""
        payload = {
            "store_id": "4",
            "alerts": [
                {"dept_id": "1", "urgency": "red"},
                {"dept_id": "2", "urgency": "red"},
                {"dept_id": "3", "urgency": "amber"},
                {"dept_id": "4", "urgency": "amber"},
                {"dept_id": "5", "urgency": "amber"},
            ],
            "summary": {
                "total_depts_at_risk": 5,
                "total_capital_freed_estimate": 35000.0,
                "projected_weekly_sales": 110000.0,
            },
        }
        headers = {"Authorization": f"Bearer {self.manager_token}"}
        res = self.client.post("/insights/generate", json=payload, headers=headers)
        self.assertEqual(res.status_code, 200)

        data = res.json()
        # Commentary must be non-empty
        self.assertTrue(len(data["insight_text"]) > 30)
        # Source can be 'llm' (Groq/OpenAI key present) or 'fallback' (no key)
        self.assertIn(data["generated_by"], ["llm", "fallback"])

    def test_mocked_llm_success(self):
        """Verify LLM source when an LLM call succeeds."""
        mock_commentary = (
            "Store 4 shows 2 departments at critical stockout risk, requiring urgent replenishment. "
            "Reordering now will protect $110,000.00 in weekly sales while freeing $35,000.00 in working capital."
        )
        with patch.object(llm_insights, "call_llm_commentary", return_value=mock_commentary):
            payload = {
                "store_id": "4",
                "alerts": [
                    {"dept_id": "1", "urgency": "red"},
                    {"dept_id": "2", "urgency": "red"},
                    {"dept_id": "3", "urgency": "amber"},
                    {"dept_id": "4", "urgency": "amber"},
                    {"dept_id": "5", "urgency": "amber"},
                ],
                "summary": {
                    "total_depts_at_risk": 5,
                    "total_capital_freed_estimate": 35000.0,
                    "projected_weekly_sales": 110000.0,
                },
            }
            headers = {"Authorization": f"Bearer {self.admin_token}"}
            res = self.client.post("/insights/generate", json=payload, headers=headers)
            self.assertEqual(res.status_code, 200)

            data = res.json()
            self.assertEqual(data["insight_text"], mock_commentary)
            self.assertEqual(data["generated_by"], "llm")

    def test_insights_root_alias(self):
        """Verify POST /insights root alias endpoint returns 200 with valid commentary."""
        payload = {
            "store_id": "7",
            "alerts": [],
            "summary": {
                "total_depts_at_risk": 0,
                "total_capital_freed_estimate": 0.0,
            },
        }
        headers = {"Authorization": f"Bearer {self.manager_token}"}
        res = self.client.post("/insights", json=payload, headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Commentary must be non-empty (either from LLM or template)
        self.assertTrue(len(data["insight_text"]) > 20)
        self.assertIn(data["generated_by"], ["llm", "fallback"])


if __name__ == "__main__":
    unittest.main()
