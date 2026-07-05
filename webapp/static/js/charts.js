// Chart.js yardımcıları: olasılık ve dağılım grafikleri.
window.SentimentCharts = (function () {
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function palette() {
    return {
      negatif: cssVar("--negatif") || "#c0392b",
      notr: cssVar("--notr") || "#b8862f",
      pozitif: cssVar("--pozitif") || "#2f8f57",
      grid: cssVar("--border") || "#e2e7f0",
      text: cssVar("--text-soft") || "#5b6679",
    };
  }

  var store = {};

  // İsimli bir canvas üzerine grafik çizer, öncekini temizler.
  function render(id, config) {
    if (store[id]) store[id].destroy();
    var el = document.getElementById(id);
    if (!el) return null;
    store[id] = new Chart(el, config);
    return store[id];
  }

  function doughnut(id, probs) {
    var c = palette();
    return render(id, {
      type: "doughnut",
      data: {
        labels: ["Negatif", "Nötr", "Pozitif"],
        datasets: [{
          data: [probs.negatif, probs.notr, probs.pozitif],
          backgroundColor: [c.negatif, c.notr, c.pozitif],
          borderWidth: 0,
        }],
      },
      options: {
        cutout: "62%",
        plugins: {
          legend: { position: "bottom", labels: { color: c.text, boxWidth: 12 } },
          tooltip: { callbacks: { label: function (x) { return x.label + ": %" + x.parsed; } } },
        },
      },
    });
  }

  function distribution(id, counts) {
    var c = palette();
    return render(id, {
      type: "bar",
      data: {
        labels: ["Negatif", "Nötr", "Pozitif"],
        datasets: [{
          data: [counts.negatif, counts.notr, counts.pozitif],
          backgroundColor: [c.negatif, c.notr, c.pozitif],
          borderRadius: 6,
        }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: c.text }, grid: { display: false } },
          y: { ticks: { color: c.text }, grid: { color: c.grid }, beginAtZero: true },
        },
      },
    });
  }

  function metricBars(id, rows) {
    var c = palette();
    return render(id, {
      type: "bar",
      data: {
        labels: rows.map(function (r) { return r.model; }),
        datasets: [{
          label: "F1-Macro",
          data: rows.map(function (r) { return r.f1; }),
          backgroundColor: c.pozitif,
          borderRadius: 6,
        }],
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: { min: 0, max: 1, ticks: { color: c.text }, grid: { color: c.grid } },
          y: { ticks: { color: c.text }, grid: { display: false } },
        },
      },
    });
  }

  return { doughnut: doughnut, distribution: distribution, metricBars: metricBars };
})();
