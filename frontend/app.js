/**
 * Module 7 - Frontend Logic (app.js)
 * Visual & Functional Bridge for Retail Sales Forecasting & Inventory Platform
 */

// Backend Base URL configuration
const API_BASE_URL = "http://localhost:8000";

// Global reference for Chart.js instance
let salesChartInstance = null;

// =========================================================
// 1. PRIMARY FUNCTION: Load Forecast Data & Render Chart
// =========================================================

/**
 * Triggered by the "Load Chart" button.
 * Fetches forecast predictions from GET /forecast/{store_id},
 * handles the 104-week validation error via alert(), and renders Chart.js line graph.
 */
async function loadForecastChart() {
  const storeId = document.getElementById("storeId").value.trim();
  const deptId = document.getElementById("deptId").value.trim();
  const startDate = document.getElementById("startDate").value.trim();
  const endDate = document.getElementById("endDate").value.trim();
  const model = document.getElementById("modelSelect") ? document.getElementById("modelSelect").value : "lightgbm";
  const loadBtn = document.getElementById("loadChartBtn");

  // Validate form completeness before sending
  if (!storeId || !deptId || !startDate || !endDate) {
    alert("Please fill in Store ID, Dept ID, Start Date, and End Date.");
    return;
  }

  // Update button UI state to show loading
  if (loadBtn) {
    loadBtn.disabled = true;
    loadBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin" style="animation: spin 1s linear infinite;">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
        <path d="M12 2a10 10 0 0 1 10 10"></path>
      </svg> Loading...
    `;
  }

  // Construct target query URL
  const queryParams = new URLSearchParams({
    dept_id: deptId,
    start_date: startDate,
    end_date: endDate,
    model: model
  });
  const endpoint = `${API_BASE_URL}/forecast/${encodeURIComponent(storeId)}?${queryParams.toString()}`;

  try {
    const response = await fetch(endpoint, {
      method: "GET",
      headers: {
        "Accept": "application/json"
      }
    });

    // Handle 400 Bad Request (including 104-week limit error) and other HTTP errors
    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.json();
      } catch (e) {
        errorData = { message: `HTTP error ${response.status}: ${response.statusText}` };
      }

      // Show graceful browser alert with exact message from server
      const alertMsg = errorData.message || "An error occurred while fetching the forecast.";
      alert(alertMsg);
      return;
    }

    // Parse successful JSON payload
    const data = await response.json();

    if (!data.predictions || data.predictions.length === 0) {
      alert("No forecast predictions returned for the selected parameters.");
      return;
    }

    // Map predictions array into Chart.js series
    const labels = data.predictions.map(item => item.week_ending_date);
    const predictedSales = data.predictions.map(item => item.predicted_weekly_sales);
    const actualSales = data.predictions.map(item => item.actual_weekly_sales !== undefined ? item.actual_weekly_sales : null);

    // Update KPI badges on page
    const kpiStoreDept = document.getElementById("kpiStoreDept");
    const kpiModel = document.getElementById("kpiModel");
    const chartRangeMeta = document.getElementById("chartRangeMeta");
    
    if (kpiStoreDept) kpiStoreDept.textContent = `Store ${data.store_id} / Dept ${data.dept_id}`;
    if (kpiModel) kpiModel.textContent = (data.model_used || model).toUpperCase();
    if (chartRangeMeta) {
      chartRangeMeta.textContent = `Displaying ${labels.length} weekly forecast intervals (${startDate} to ${endDate})`;
    }

    // Render the Chart.js Line Chart
    renderSalesChart(labels, predictedSales, actualSales, data.model_used || model);

  } catch (err) {
    console.error("Network or parsing error in loadForecastChart:", err);
    alert(`Could not connect to API at ${API_BASE_URL}. Ensure FastAPI is running on port 8000.\n\nError: ${err.message}`);
  } finally {
    // Reset button state
    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
        Load Chart
      `;
    }
  }
}


// =========================================================
// 2. CHART RENDERING ENGINE: Chart.js Line Graph
// =========================================================

/**
 * Renders or updates the Chart.js line graph on <canvas id="salesChart">.
 * Destroys existing instance before rendering a new one.
 */
function renderSalesChart(labels, predictedSales, actualSales, modelName) {
  const canvas = document.getElementById("salesChart");
  if (!canvas) {
    console.error("Canvas element #salesChart not found in DOM");
    return;
  }

  const ctx = canvas.getContext("2d");

  // Destroy existing chart instance to prevent canvas overlap glitches
  if (salesChartInstance) {
    salesChartInstance.destroy();
    salesChartInstance = null;
  }

  // Create subtle gradient for predicted sales area fill
  const gradientFill = ctx.createLinearGradient(0, 0, 0, 350);
  gradientFill.addColorStop(0, "rgba(99, 102, 241, 0.35)");
  gradientFill.addColorStop(1, "rgba(99, 102, 241, 0.0)");

  salesChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: `Predicted Weekly Sales (${modelName.toUpperCase()})`,
          data: predictedSales,
          borderColor: "#6366f1",
          backgroundColor: gradientFill,
          borderWidth: 3,
          tension: 0.35,
          fill: true,
          pointBackgroundColor: "#818cf8",
          pointBorderColor: "#ffffff",
          pointBorderWidth: 1.5,
          pointRadius: 4,
          pointHoverRadius: 7,
          pointHoverBackgroundColor: "#6366f1",
          pointHoverBorderColor: "#ffffff",
          pointHoverBorderWidth: 2
        },
        {
          label: "Actual Historical Sales",
          data: actualSales,
          borderColor: "#10b981",
          backgroundColor: "transparent",
          borderWidth: 2.5,
          borderDash: [5, 5],
          tension: 0.35,
          fill: false,
          pointBackgroundColor: "#10b981",
          pointBorderColor: "#ffffff",
          pointBorderWidth: 1.5,
          pointRadius: 4,
          pointHoverRadius: 7,
          pointHoverBackgroundColor: "#10b981",
          pointHoverBorderColor: "#ffffff",
          pointHoverBorderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          position: "top",
          align: "end",
          labels: {
            color: "#9ca3af",
            font: {
              family: "'Inter', sans-serif",
              size: 12,
              weight: 500
            },
            usePointStyle: true,
            boxWidth: 8,
            padding: 20
          }
        },
        tooltip: {
          backgroundColor: "rgba(17, 24, 39, 0.95)",
          titleColor: "#f9fafb",
          bodyColor: "#e5e7eb",
          borderColor: "rgba(255, 255, 255, 0.12)",
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
          callbacks: {
            label: function (context) {
              let label = context.dataset.label || "";
              if (label) {
                label += ": ";
              }
              if (context.parsed.y !== null) {
                label += new Intl.NumberFormat("en-US", {
                  style: "currency",
                  currency: "USD",
                  minimumFractionDigits: 2
                }).format(context.parsed.y);
              }
              return label;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            color: "rgba(255, 255, 255, 0.05)",
            drawBorder: false
          },
          ticks: {
            color: "#9ca3af",
            font: {
              family: "'Inter', sans-serif",
              size: 11
            },
            maxRotation: 45
          }
        },
        y: {
          grid: {
            color: "rgba(255, 255, 255, 0.06)",
            drawBorder: false
          },
          ticks: {
            color: "#9ca3af",
            font: {
              family: "'JetBrains Mono', monospace",
              size: 11
            },
            callback: function (value) {
              return "$" + (value >= 1000 ? (value / 1000).toFixed(0) + "k" : value);
            }
          }
        }
      }
    }
  });
}


// =========================================================
// 3. SECONDARY FUNCTION: Render Reorder Intelligence Table
// =========================================================

/**
 * Renders inventory dummy data into #reorderTableBody.
 * Color-codes the "Urgency" column:
 *   - Red   for <= 1 week
 *   - Amber for 1 - 2 weeks
 *   - Green for > 2 weeks
 */
function renderReorderTable() {
  const tbody = document.getElementById("reorderTableBody");
  if (!tbody) {
    console.error("Table body #reorderTableBody not found in DOM");
    return;
  }

  // Realistic retail department dummy data
  const dummyReorderData = [
    {
      dept: "92 (Groceries / Consumables)",
      currentStock: 320,
      demand: 410,
      weeksToStockout: 0.78,
      reorderUnits: 500,
      capitalFreed: 0.0
    },
    {
      dept: "95 (Household / Paper)",
      currentStock: 850,
      demand: 620,
      weeksToStockout: 1.37,
      reorderUnits: 390,
      capitalFreed: 0.0
    },
    {
      dept: "38 (Pharmacy / Wellness)",
      currentStock: 120,
      demand: 280,
      weeksToStockout: 0.43,
      reorderUnits: 440,
      capitalFreed: 0.0
    },
    {
      dept: "72 (Electronics / Audio)",
      currentStock: 1450,
      demand: 800,
      weeksToStockout: 1.81,
      reorderUnits: 150,
      capitalFreed: 0.0
    },
    {
      dept: "40 (Pet Supplies / Food)",
      currentStock: 3100,
      demand: 950,
      weeksToStockout: 3.26,
      reorderUnits: 0,
      capitalFreed: 1850.0
    },
    {
      dept: "90 (Dairy & Fresh Foods)",
      currentStock: 2400,
      demand: 710,
      weeksToStockout: 3.38,
      reorderUnits: 0,
      capitalFreed: 2340.0
    },
    {
      dept: "01 (Candy & Seasonal)",
      currentStock: 90,
      demand: 210,
      weeksToStockout: 0.43,
      reorderUnits: 330,
      capitalFreed: 0.0
    }
  ];

  let urgentCount = 0;
  let totalCapitalFreed = 0;

  // Build table rows
  const rowsHtml = dummyReorderData.map(row => {
    let urgencyBadge = "";
    const weeks = row.weeksToStockout;

    // Strict color-coding rules:
    // Red for <= 1 week
    // Amber for 1-2 weeks
    // Green for > 2 weeks
    if (weeks <= 1.0) {
      urgencyBadge = `<span class="badge-urgency badge-red"><span class="badge-dot"></span>Red (≤ 1 wk)</span>`;
      urgentCount++;
    } else if (weeks <= 2.0) {
      urgencyBadge = `<span class="badge-urgency badge-amber"><span class="badge-dot"></span>Amber (1-2 wks)</span>`;
    } else {
      urgencyBadge = `<span class="badge-urgency badge-green"><span class="badge-dot"></span>Green (> 2 wks)</span>`;
    }

    totalCapitalFreed += row.capitalFreed;

    const formattedCapital = row.capitalFreed > 0 
      ? `<span style="color: var(--accent-emerald); font-weight: 600;">$${row.capitalFreed.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>`
      : `<span style="color: var(--text-muted);">$0.00</span>`;

    const formattedReorder = row.reorderUnits > 0
      ? `<strong style="color: var(--accent-rose); font-family: 'JetBrains Mono', monospace;">+${row.reorderUnits.toLocaleString()} units</strong>`
      : `<span style="color: var(--accent-emerald); font-weight: 500;">Sufficient</span>`;

    return `
      <tr>
        <td style="font-weight: 600;">${row.dept}</td>
        <td class="num-highlight">${row.currentStock.toLocaleString()} units</td>
        <td class="num-highlight">${row.demand.toLocaleString()} units/wk</td>
        <td><strong class="num-highlight" style="color: ${weeks <= 1 ? 'var(--accent-rose)' : (weeks <= 2 ? 'var(--accent-amber)' : 'var(--accent-emerald)')};">${weeks.toFixed(2)} wks</strong></td>
        <td>${urgencyBadge}</td>
        <td>${formattedReorder}</td>
        <td>${formattedCapital}</td>
      </tr>
    `;
  }).join("");

  tbody.innerHTML = rowsHtml;

  // Update header KPI values if elements exist
  const kpiUrgent = document.getElementById("kpiUrgent");
  const kpiCapital = document.getElementById("kpiCapital");
  if (kpiUrgent) kpiUrgent.textContent = `${urgentCount} Depts`;
  if (kpiCapital) kpiCapital.textContent = `$${totalCapitalFreed.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
}


// =========================================================
// 4. BONUS FEATURE: Promotional What-If Simulation Runner
// =========================================================

/**
 * Simulates promotional markdowns via POST /whatif and updates UI.
 */
async function runWhatIfSimulation() {
  const storeId = document.getElementById("storeId").value.trim();
  const deptId = document.getElementById("deptId").value.trim();
  const startDate = document.getElementById("startDate").value.trim();
  const endDate = document.getElementById("endDate").value.trim();
  const markdownInput = document.getElementById("whatifMarkdown");
  const markdownAmount = parseFloat(markdownInput ? markdownInput.value : 5000) || 0.0;
  
  const holidaySelect = document.getElementById("whatifHoliday");
  let holidayOverride = null;
  if (holidaySelect && holidaySelect.value === "true") holidayOverride = true;
  if (holidaySelect && holidaySelect.value === "false") holidayOverride = false;

  const resultBox = document.getElementById("whatifResultBox");
  const liftDisplay = document.getElementById("whatifLiftDisplay");
  const summaryText = document.getElementById("whatifSummaryText");

  try {
    const payload = {
      store_id: storeId,
      dept_id: deptId,
      start_date: startDate,
      end_date: endDate,
      markdown_amount: markdownAmount,
      is_holiday_override: holidayOverride
    };

    const res = await fetch(`${API_BASE_URL}/whatif`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      alert(err.message || "Failed to execute What-If simulation.");
      return;
    }

    const data = await res.json();

    if (resultBox && liftDisplay && summaryText) {
      resultBox.style.display = "flex";
      liftDisplay.textContent = `+${data.projected_lift_pct}% Lift`;
      summaryText.textContent = `With $${markdownAmount.toLocaleString()} markdown on Store ${data.store_id} Dept ${data.dept_id} (${startDate} to ${endDate}).`;
    }

  } catch (err) {
    console.error("What-If simulation error:", err);
    alert(`Could not complete What-If simulation: ${err.message}`);
  }
}


// =========================================================
// 5. INITIALIZATION & EVENT LISTENERS
// =========================================================

document.addEventListener("DOMContentLoaded", () => {
  // Bind "Load Chart" click event
  const loadBtn = document.getElementById("loadChartBtn");
  if (loadBtn) {
    loadBtn.addEventListener("click", loadForecastChart);
  }

  // Bind "Run Simulation" click event
  const whatIfBtn = document.getElementById("runWhatIfBtn");
  if (whatIfBtn) {
    whatIfBtn.addEventListener("click", runWhatIfSimulation);
  }

  // Initial table render
  renderReorderTable();

  // Trigger initial chart load automatically
  loadForecastChart();
});
