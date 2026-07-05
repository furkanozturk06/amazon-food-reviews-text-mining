"""Rapor için ekran görüntüleri üretir (Selenium, headless)."""
import os, time, base64

OUT = os.path.join(os.path.dirname(__file__), "rapor", "img")
os.makedirs(OUT, exist_ok=True)
BASE = "http://127.0.0.1:5000"

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def make_driver():
    # Önce Chrome, olmazsa Edge dene
    for name in ("chrome", "edge"):
        try:
            if name == "chrome":
                from selenium.webdriver.chrome.options import Options
                opts = Options()
            else:
                from selenium.webdriver.edge.options import Options
                opts = Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--window-size=1500,1700")
            opts.add_argument("--force-device-scale-factor=1.25")
            opts.add_argument("--hide-scrollbars")
            d = webdriver.Chrome(options=opts) if name == "chrome" else webdriver.Edge(options=opts)
            print("Tarayıcı:", name)
            return d
        except Exception as e:
            print(name, "olmadı:", str(e)[:120])
    raise SystemExit("Tarayıcı bulunamadı")


def full_png(driver, path):
    # tüm sayfa yüksekliğinde görüntü (Chrome DevTools)
    h = driver.execute_script("return document.body.parentNode.scrollHeight")
    driver.set_window_size(1500, min(h + 120, 6000))
    time.sleep(0.6)
    driver.save_screenshot(path)
    print("kaydedildi:", os.path.basename(path))


def shoot_single(driver, summary, text, model, fname):
    driver.get(BASE + "/")
    time.sleep(1.0)
    if summary:
        driver.find_element(By.ID, "review-summary").send_keys(summary)
    driver.find_element(By.ID, "review-text").send_keys(text)
    from selenium.webdriver.support.ui import Select
    Select(driver.find_element(By.ID, "model-select")).select_by_value(model)
    driver.find_element(By.ID, "predict-btn").click()
    # sonucun gelmesini bekle: rozet dolana ve buton eski haline dönene kadar
    WebDriverWait(driver, 60).until(
        lambda d: d.find_element(By.ID, "result-card").is_displayed()
        and d.find_element(By.ID, "verdict-badge").text.strip() not in ("", "—")
        and d.find_element(By.ID, "predict-btn").text.strip() == "Analiz et")
    # tüm modeller tablosu dolsun + grafik çizilsin
    WebDriverWait(driver, 30).until(
        lambda d: len(d.find_elements(By.CSS_SELECTOR, "#all-models-table tbody tr")) > 0)
    time.sleep(1.2)
    full_png(driver, os.path.join(OUT, fname))


d = make_driver()
try:
    # 1) Olumsuz + başlık + ensemble -> sonuç, canlı LIME, tüm modeller
    shoot_single(d, "Awful, do not buy",
                 "This product is terrible, stale and a complete waste of money.",
                 "ensemble", "01_tekli_olumsuz_ensemble.png")
    # 2) Olumlu + başlık -> yeşil vurgular
    shoot_single(d, "Best coffee ever",
                 "This coffee is rich, smooth and arrived perfectly fresh. Highly recommend.",
                 "lr", "02_tekli_olumlu_lime.png")
    # 3) OOV örneği: shit -> sözlükte yok (noktalı altçizgi)
    shoot_single(d, "", "shit", "ensemble", "03_oov_shit.png")

    # 3b) Yakın plan: sonuç kartı + LIME kutusu (olumsuz örnek tekrar)
    d.get(BASE + "/"); time.sleep(0.8)
    d.find_element(By.ID, "review-summary").send_keys("Awful, do not buy")
    d.find_element(By.ID, "review-text").send_keys(
        "This product is terrible, stale and a complete waste of money.")
    from selenium.webdriver.support.ui import Select
    Select(d.find_element(By.ID, "model-select")).select_by_value("ensemble")
    d.find_element(By.ID, "predict-btn").click()
    WebDriverWait(d, 60).until(
        lambda dr: dr.find_element(By.ID, "result-card").is_displayed()
        and dr.find_element(By.ID, "verdict-badge").text.strip() not in ("", "—"))
    time.sleep(1.5)
    d.find_element(By.ID, "result-card").screenshot(os.path.join(OUT, "06_sonuc_karti.png"))
    d.find_element(By.ID, "highlight-box").screenshot(os.path.join(OUT, "07_lime_yakin.png"))
    print("kaydedildi: 06/07 element görüntüleri")

    # 4) Karşılaştırma sayfası (leaderboard + ablation görseli)
    d.get(BASE + "/karsilastirma"); time.sleep(1.5)
    full_png(d, os.path.join(OUT, "04_karsilastirma.png"))

    # 5) Ana sayfa boş (giriş ekranı)
    d.get(BASE + "/"); time.sleep(1.0)
    full_png(d, os.path.join(OUT, "05_giris.png"))
finally:
    d.quit()
print("BİTTİ")
