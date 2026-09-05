// ============================================================
// CHART SETUP
// ============================================================
let chartType = "temp";
let liveChart = null;
const MAX_POINTS = 30;
const chartData = { temp: [], vib: [], curr: [], pres: [], labels: [] };
const colors = {
  temp: "#378ADD",
  vib: "#D85A30",
  curr: "#1D9E75",
  pres: "#7F77DD",
};
const chartLabels = {
  temp: "Supply temp (°C)",
  vib: "Vibration magnitude",
  curr: "Current (A)",
  pres: "Low pressure (PSI)",
};

function initChart() {
  const ctx = document.getElementById("liveChart").getContext("2d");
  liveChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: chartData.labels,
      datasets: [
        {
          label: chartLabels[chartType],
          data: chartData[chartType],
          borderColor: colors[chartType],
          backgroundColor: colors[chartType] + "15",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: {
            color: "#888780",
            font: { size: 10 },
            maxTicksLimit: 6,
          },
          grid: { color: "rgba(136,135,128,0.1)" },
        },
        y: {
          ticks: { color: "#888780", font: { size: 10 } },
          grid: { color: "rgba(136,135,128,0.1)" },
        },
      },
    },
  });
}

function setTab(type, btn) {
  chartType = type;
  document
    .querySelectorAll(".tab")
    .forEach((t) => t.classList.remove("active"));
  btn.classList.add("active");
  liveChart.data.datasets[0].label = chartLabels[type];
  liveChart.data.datasets[0].borderColor = colors[type];
  liveChart.data.datasets[0].backgroundColor = colors[type] + "15";
  liveChart.data.datasets[0].data = chartData[type];
  liveChart.update("none");
}

function pushChartData(d) {
  const time = new Date().toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  chartData.labels.push(time);
  chartData.temp.push(d.supply_temp);
  chartData.vib.push(d.vib_mag);
  chartData.curr.push(d.current);
  chartData.pres.push(d.low_psi);
  if (chartData.labels.length > MAX_POINTS) {
    chartData.labels.shift();
    ["temp", "vib", "curr", "pres"].forEach((k) => chartData[k].shift());
  }
  liveChart.data.labels = chartData.labels;
  liveChart.data.datasets[0].data = chartData[chartType];
  liveChart.update("none");
}

// ============================================================
// UI UPDATE
// ============================================================
const ALERT_CONF = {
  green: { icon: "❄️", title: "System Healthy", badgeClass: "badge-ok" },
  yellow: {
    icon: "🌡️",
    title: "Low Refrigerant Detected",
    badgeClass: "badge-warn",
  },
  orange: {
    icon: "🔥",
    title: "Fault Detected — Action Needed",
    badgeClass: "badge-warn",
  },
  red: {
    icon: "⚠️",
    title: "Critical Fault Detected",
    badgeClass: "badge-crit",
  },
};

function updateUI(data) {
  const level = data.alert_level || "green";
  const conf = ALERT_CONF[level];

  // Status banner
  const banner = document.getElementById("statusBanner");
  banner.className = "status-banner " + level;
  document.getElementById("statusIcon").textContent = conf.icon;
  document.getElementById("statusTitle").textContent =
    data.final_prediction === "HEALTHY" ? "System Healthy" : data.alert_message;
  document.getElementById("statusRec").textContent = data.recommendation;

  // Metrics
  const f = data.features;
  document.getElementById("mSupply").textContent = f.supply_temp.toFixed(1);
  document.getElementById("mRoom").textContent = f.room_temp.toFixed(1);
  document.getElementById("mVib").textContent = Math.round(
    f.vib_mag,
  ).toLocaleString();
  document.getElementById("mCurr").textContent = f.current.toFixed(1);

  const supplyOk = f.supply_temp < 17;
  document.getElementById("mSupplyStatus").className =
    "metric-status " + (supplyOk ? "s-ok" : "s-warn");
  document.getElementById("mSupplyStatus").textContent = supplyOk
    ? "Normal range"
    : "Above normal";

  const vibOk = f.vib_mag < 12000;
  document.getElementById("mVibStatus").className =
    "metric-status " + (vibOk ? "s-ok" : "s-crit");
  document.getElementById("mVibStatus").textContent = vibOk
    ? "Normal"
    : "Elevated!";

  const currOk = f.current < 12;
  document.getElementById("mCurrStatus").className =
    "metric-status " + (currOk ? "s-ok" : "s-warn");
  document.getElementById("mCurrStatus").textContent = currOk
    ? "Normal draw"
    : "High draw";

  // Models
  document.getElementById("rfFault").textContent = data.rf_prediction;
  document.getElementById("rfConf").textContent = data.rf_confidence + "%";
  document.getElementById("rfBar").style.width = data.rf_confidence + "%";

  if (data.lstm_prediction !== "UNAVAILABLE") {
    document.getElementById("lstmFault").textContent = data.lstm_prediction;
    document.getElementById("lstmConf").textContent =
      data.lstm_confidence + "%";
    document.getElementById("lstmBar").style.width = data.lstm_confidence + "%";
    const agree = data.rf_prediction === data.lstm_prediction;
    document.getElementById("agreeNote").textContent = agree
      ? "✓ Both models agree — high reliability"
      : "⚠ Models disagree — monitoring...";
  } else {
    document.getElementById("lstmFault").textContent =
      "Buffering... (" + data.buffer_size + "/60)";
  }

  // Pressures
  document.getElementById("gLo").textContent = f.low_psi.toFixed(0);
  document.getElementById("gHi").textContent = f.high_psi.toFixed(0);
  const loFrac = Math.min(f.low_psi / 120, 1) * 100;
  const hiFrac = Math.min(f.high_psi / 500, 1) * 100;
  const loColor =
    f.low_psi >= 60 && f.low_psi <= 90 ? "var(--green)" : "var(--red)";
  const hiColor =
    f.high_psi >= 250 && f.high_psi <= 350 ? "var(--green)" : "var(--red)";
  document.getElementById("gLoBar").style.width = loFrac + "%";
  document.getElementById("gLoBar").style.background = loColor;
  document.getElementById("gHiBar").style.width = hiFrac + "%";
  document.getElementById("gHiBar").style.background = hiColor;

  // Chart
  pushChartData(f);

  // Event log (only on fault change)
  if (level !== lastLevel) {
    addEvent(data.alert_message, conf.badgeClass);
    lastLevel = level;
  }
}

let lastLevel = "green";

function addEvent(text, badgeClass) {
  const log = document.getElementById("eventLog");
  const time = new Date().toLocaleTimeString();
  const badgeMap = {
    "badge-ok": "NORMAL",
    "badge-warn": "ALERT",
    "badge-crit": "CRITICAL",
    "badge-info": "INFO",
  };
  const div = document.createElement("div");
  div.className = "event";
  div.innerHTML = `<span class="badge ${badgeClass}">${badgeMap[badgeClass] || "INFO"}</span><div><div class="event-text">${text}</div><div class="event-time">${time}</div></div>`;
  log.insertBefore(div, log.firstChild);
  if (log.children.length > 10) log.removeChild(log.lastChild);
}

// ============================================================
// DEMO SIMULATION (when no ESP32 connected)
// ============================================================
const DEMO_DATA = {
  HEALTHY: {
    supply_temp: 14.2,
    room_temp: 26.5,
    vibration_mag: 9100,
    vibration_std: 35,
    gyroscope: 5,
    current: 7.8,
    low_pressure: 75,
    high_pressure: 290,
    compressor_on: 1,
  },
  LOW_REFRIGERANT: {
    supply_temp: 21.5,
    room_temp: 31.0,
    vibration_mag: 9800,
    vibration_std: 70,
    gyroscope: 12,
    current: 9.5,
    low_pressure: 32,
    high_pressure: 175,
    compressor_on: 1,
  },
  BLOCKED_CONDENSER: {
    supply_temp: 18.0,
    room_temp: 29.0,
    vibration_mag: 10800,
    vibration_std: 130,
    gyroscope: 18,
    current: 13.5,
    low_pressure: 85,
    high_pressure: 420,
    compressor_on: 1,
  },
  BEARING_WEAR: {
    supply_temp: 15.0,
    room_temp: 27.0,
    vibration_mag: 16500,
    vibration_std: 450,
    gyroscope: 65,
    current: 10.5,
    low_pressure: 72,
    high_pressure: 295,
    compressor_on: 1,
  },
};

function simulate(fault) {
  const base = DEMO_DATA[fault];
  const noise = (v, p = 0.04) => v * (1 + (Math.random() - 0.5) * p);
  const payload = {
    supply_temp: parseFloat(noise(base.supply_temp).toFixed(1)),
    room_temp: parseFloat(noise(base.room_temp).toFixed(1)),
    vibration_mag: Math.round(noise(base.vibration_mag, 0.05)),
    vibration_std: parseFloat(noise(base.vibration_std, 0.1).toFixed(1)),
    gyroscope: parseFloat(noise(base.gyroscope, 0.1).toFixed(1)),
    current: parseFloat(noise(base.current, 0.06).toFixed(2)),
    low_pressure: parseFloat(noise(base.low_pressure, 0.04).toFixed(1)),
    high_pressure: parseFloat(noise(base.high_pressure, 0.04).toFixed(1)),
    compressor_on: base.compressor_on,
  };

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((r) => r.json())
    .then((data) => {
      if (!data.error) updateUI(data);
    })
    .catch((err) => console.error("Predict error:", err));
}

// ============================================================
// AUTO REFRESH from /history (when ESP32 is connected)
// ============================================================
function pollLatest() {
  fetch("/status")
    .then((r) => r.json())
    .then((s) => {
      document.getElementById("liveDot").style.background = s.rf_ready
        ? "var(--green)"
        : "var(--gray)";
    })
    .catch(() => {});
}

// ============================================================
// PWA — INSTALL PROMPT
// ============================================================
let deferredPrompt = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferredPrompt = e;
  document.getElementById("installBanner").style.display = "flex";
});

document.getElementById("installBtn").addEventListener("click", () => {
  if (deferredPrompt) {
    deferredPrompt.prompt();
    deferredPrompt.userChoice.then(() => {
      document.getElementById("installBanner").style.display = "none";
      deferredPrompt = null;
    });
  }
});

// ============================================================
// OFFLINE DETECTION
// ============================================================
window.addEventListener("online", () => {
  document.getElementById("offlineBanner").style.display = "none";
});
window.addEventListener("offline", () => {
  document.getElementById("offlineBanner").style.display = "block";
});

// ============================================================
// SERVICE WORKER REGISTRATION
// ============================================================
if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register("/sw.js")
    .then(() => console.log("SW registered"))
    .catch((e) => console.log("SW error:", e));
}

function scrollTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ============================================================
// INIT
// ============================================================
initChart();
setInterval(pollLatest, 5000);
pollLatest();

// Auto simulate HEALTHY on load for demo
setTimeout(() => simulate("HEALTHY"), 1000);
setInterval(() => {
  if (document.getElementById("demoCard").style.display !== "none") {
    simulate("HEALTHY");
  }
}, 3000);
