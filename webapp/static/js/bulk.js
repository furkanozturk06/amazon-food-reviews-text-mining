// Toplu analiz sayfası mantığı.
(function () {
  var textEl = document.getElementById("bulk-text");
  var modelEl = document.getElementById("bulk-model");
  var fileEl = document.getElementById("bulk-file");
  var btn = document.getElementById("bulk-btn");
  if (!btn) return;

  var resultBox = document.getElementById("bulk-result");
  var lastRows = [];

  var LABELS = { negatif: "Negatif", notr: "Nötr", pozitif: "Pozitif" };

  fileEl.addEventListener("change", function () {
    var f = fileEl.files[0];
    if (!f) return;
    var reader = new FileReader();
    reader.onload = function () { textEl.value = stripHeader(reader.result); };
    reader.readAsText(f, "utf-8");
  });

  // CSV başlığı gibi görünen ilk satırı eler (text/review/yorum vb.).
  function stripHeader(content) {
    var lines = content.split(/\r?\n/);
    if (lines.length && /^(text|review|yorum|comment|summary)\b/i.test(lines[0].trim())) {
      lines.shift();
    }
    // CSV ise ilk sütunu al
    return lines.map(function (l) { return l.split(",")[0].trim(); })
                .filter(Boolean).join("\n");
  }

  btn.addEventListener("click", function () {
    var texts = textEl.value.split(/\r?\n/).map(function (l) { return l.trim(); }).filter(Boolean);
    if (!texts.length) { textEl.focus(); return; }

    btn.disabled = true;
    btn.textContent = "Analiz ediliyor...";

    fetch("/api/toplu", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: texts, model: modelEl.value }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) { alert(data.error); return; }
        render(data);
      })
      .catch(function () { alert("Toplu analiz sırasında bir hata oluştu."); })
      .finally(function () {
        btn.disabled = false;
        btn.textContent = "Analiz et";
      });
  });

  function render(data) {
    resultBox.hidden = false;
    lastRows = data.results;

    var c = data.counts, total = data.total;
    document.getElementById("bulk-summary").innerHTML =
      line("Toplam", total) +
      line("Negatif", c.negatif + "  (%" + pct(c.negatif, total) + ")") +
      line("Nötr", c.notr + "  (%" + pct(c.notr, total) + ")") +
      line("Pozitif", c.pozitif + "  (%" + pct(c.pozitif, total) + ")") +
      line("Model", data.model_title);

    SentimentCharts.distribution("bulk-chart", c);

    var body = document.querySelector("#bulk-table tbody");
    body.innerHTML = data.results.map(function (r, i) {
      return "<tr><td>" + (i + 1) + "</td><td>" + escapeHtml(clip(r.text)) + "</td>" +
        '<td><span class="tag ' + r.slug + '">' + r.label + "</span></td>" +
        "<td>%" + r.confidence + "</td></tr>";
    }).join("");
  }

  document.getElementById("download-btn").addEventListener("click", function () {
    if (!lastRows.length) return;
    var rows = [["yorum", "tahmin", "guven"]];
    lastRows.forEach(function (r) { rows.push([r.text, r.label, r.confidence]); });
    var csv = rows.map(function (row) {
      return row.map(function (v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(",");
    }).join("\r\n");
    var blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "tahmin_sonuclari.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  function line(label, value) {
    return '<div class="summary-line"><span>' + label + "</span><b>" + value + "</b></div>";
  }
  function pct(n, total) { return total ? Math.round(n / total * 100) : 0; }
  function clip(s) { return s.length > 120 ? s.slice(0, 120) + "…" : s; }
  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
})();
