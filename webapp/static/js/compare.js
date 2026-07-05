// Karşılaştırma sayfası: F1-Macro çubuk grafiği.
(function () {
  var el = document.getElementById("metrics-data");
  if (!el) return;
  var rows;
  try { rows = JSON.parse(el.textContent); } catch (e) { return; }
  if (!rows || !rows.length) return;

  SentimentCharts.metricBars("metric-chart", rows);
})();
