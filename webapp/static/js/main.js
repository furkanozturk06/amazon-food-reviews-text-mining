// Tekli tahmin sayfası mantığı.
(function () {
  var textEl = document.getElementById("review-text");
  var summaryEl = document.getElementById("review-summary");
  var modelEl = document.getElementById("model-select");
  var btn = document.getElementById("predict-btn");
  if (!btn) return;

  var card = document.getElementById("result-card");
  var allCard = document.getElementById("all-models-card");

  // Her kategori için örnek havuzu; tıklandığında rastgele (üst üste aynısı gelmez) biri seçilir.
  var EXAMPLES = {
    olumlu: [
      "Absolutely delicious and fresh, the best snack I have bought in years.",
      "This coffee is rich, smooth and arrived perfectly fresh. Highly recommend.",
      "Amazing flavor and great quality, my whole family loves it.",
      "Perfect taste and fast delivery, I will definitely order again.",
      "Wonderful product, it exceeded my expectations and is worth every penny.",
      "So tasty and well packaged, one of my favorite purchases this year.",
    ],
    notr: [
      "It was okay, nothing special. The taste is average and the packaging is fine.",
      "Decent product but a bit overpriced for what you actually get.",
      "Not bad, not great. It does the job but I probably would not reorder.",
      "The flavor is acceptable, though it could be a little stronger.",
      "It is fine overall, average quality and a reasonable price.",
      "Reasonable taste, but nothing that really stands out from similar products.",
    ],
    olumsuz: [
      "Terrible quality, stale and tasteless. A complete waste of money.",
      "Very disappointed, the product arrived damaged and smelled strange.",
      "Awful taste and the package was almost empty. I would not buy again.",
      "This was a bad experience, the item was expired on arrival.",
      "Poor quality and overpriced, I really regret this purchase.",
      "Not good at all, the flavor was off and the texture was unpleasant.",
    ],
  };
  var lastPick = {};

  document.querySelectorAll(".chip").forEach(function (chip) {
    chip.addEventListener("click", function () {
      var cat = chip.getAttribute("data-category");
      var pool = EXAMPLES[cat] || [];
      if (!pool.length) return;
      var i = Math.floor(Math.random() * pool.length);
      if (pool.length > 1 && i === lastPick[cat]) i = (i + 1) % pool.length;
      lastPick[cat] = i;
      textEl.value = pool[i];
      textEl.focus();
    });
  });

  btn.addEventListener("click", run);
  textEl.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run();
  });

  function run() {
    var text = textEl.value.trim();
    if (!text) { textEl.focus(); return; }

    btn.disabled = true;
    btn.textContent = "Analiz ediliyor...";

    var summary = summaryEl ? summaryEl.value.trim() : "";

    fetch("/api/tahmin", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, summary: summary, model: modelEl.value }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) { alert(data.error); return; }
        renderResult(data);
        renderAllModels(data.all_models);
      })
      .catch(function () { alert("Tahmin sırasında bir hata oluştu."); })
      .finally(function () {
        btn.disabled = false;
        btn.textContent = "Analiz et";
      });
  }

  function renderResult(d) {
    card.hidden = false;

    var badge = document.getElementById("verdict-badge");
    badge.textContent = d.label;
    badge.className = "badge " + d.slug;
    document.getElementById("verdict-conf").textContent =
      "Güven: %" + d.confidence + " · " + d.model_title;

    SentimentCharts.doughnut("prob-chart", d.probabilities);

    var order = [["negatif", "Negatif"], ["notr", "Nötr"], ["pozitif", "Pozitif"]];
    document.getElementById("prob-bars").innerHTML = order.map(function (o) {
      var pct = d.probabilities[o[0]];
      return '<div class="prob-row"><span class="pname">' + o[1] + '</span>' +
        '<span class="prob-track"><span class="prob-fill ' + o[0] + '" style="width:' + pct + '%"></span></span>' +
        '<span class="prob-val">%' + pct + '</span></div>';
    }).join("");

    var s = d.stats;
    document.getElementById("stats-row").innerHTML =
      pill(s.word_count, "kelime") + pill(s.char_count, "karakter") +
      pill(s.polarity, "polarite") + pill(s.subjectivity, "öznellik");

    renderHighlights(d.highlights);

    var tags = d.influential_words.map(function (w) {
      var cls = w.direction > 0 ? "pos" : (w.direction < 0 ? "neg" : "");
      return '<span class="wtag ' + cls + '">' + escapeHtml(w.word) + "</span>";
    });
    document.getElementById("word-tags").innerHTML =
      tags.length ? tags.join("") : '<span class="muted small">Modelin tanıdığı belirgin kelime bulunamadı.</span>';
  }

  // Canlı LIME vurgusu: temizlenmiş girdideki her kelimeyi katkısına göre boya.
  function renderHighlights(items) {
    var box = document.getElementById("highlight-box");
    if (!box) return;
    if (!items || !items.length) {
      box.innerHTML = '<span class="muted small">Vurgulanacak kelime bulunamadı.</span>';
      return;
    }
    box.innerHTML = items.map(function (h) {
      if (!h.known) {
        return '<span class="hl hl-unknown" title="modelin sözlüğünde yok">' +
          escapeHtml(h.token) + "</span>";
      }
      var mag = Math.min(Math.abs(h.intensity), 1);     // 0..1
      var alpha = (0.12 + mag * 0.78).toFixed(2);        // görünür taban + ölçek
      var rgb = h.direction > 0 ? "46,125,50" : (h.direction < 0 ? "198,40,40" : "120,120,120");
      var title = (h.direction > 0 ? "destekler" : (h.direction < 0 ? "karşı" : "nötr")) +
        " · etki " + mag.toFixed(2);
      return '<span class="hl" title="' + title + '" style="background:rgba(' +
        rgb + "," + alpha + ')">' + escapeHtml(h.token) + "</span>";
    }).join(" ");
  }

  function renderAllModels(rows) {
    if (!rows || !rows.length) return;
    allCard.hidden = false;
    var body = allCard.querySelector("tbody");
    body.innerHTML = rows.map(function (r) {
      return "<tr><td>" + r.title + "</td><td>" + r.kind + "</td>" +
        '<td><span class="tag ' + r.slug + '">' + r.label + "</span></td>" +
        "<td>%" + r.confidence + "</td></tr>";
    }).join("");
  }

  function pill(value, label) {
    return '<div class="stat-pill"><b>' + value + "</b>" + label + "</div>";
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
})();
