"""Rapor için KOMPAKT (kırpılmış, element bazlı) ekran görüntüleri."""
import os, time
OUT = os.path.join(os.path.dirname(__file__), "rapor", "img")
BASE = "http://127.0.0.1:5000"
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.chrome.options import Options

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1500,2200")
opts.add_argument("--force-device-scale-factor=2")   # keskin (retina) görüntü
opts.add_argument("--hide-scrollbars")
d = webdriver.Chrome(options=opts)


def shot(el, name):
    time.sleep(0.3)
    el.screenshot(os.path.join(OUT, name))
    print("kaydedildi:", name)


try:
    # --- index: giriş kartı + tüm modeller kartı ---
    d.get(BASE + "/"); time.sleep(1.0)
    d.find_element(By.ID, "review-summary").send_keys("Awful, do not buy")
    d.find_element(By.ID, "review-text").send_keys(
        "This product is terrible, stale and a complete waste of money.")
    Select(d.find_element(By.ID, "model-select")).select_by_value("ensemble")
    # giriş kartını (sonuçtan önce) çek
    shot(d.find_elements(By.CSS_SELECTOR, "section.card")[0], "09_giris_kart.png")
    d.find_element(By.ID, "predict-btn").click()
    WebDriverWait(d, 60).until(
        lambda dr: dr.find_element(By.ID, "result-card").is_displayed()
        and dr.find_element(By.ID, "verdict-badge").text.strip() not in ("", "—"))
    WebDriverWait(d, 30).until(
        lambda dr: len(dr.find_elements(By.CSS_SELECTOR, "#all-models-table tbody tr")) > 0)
    time.sleep(1.0)
    shot(d.find_element(By.ID, "result-card"), "06_sonuc_karti.png")
    shot(d.find_element(By.ID, "highlight-box"), "07_lime_yakin.png")
    shot(d.find_element(By.ID, "all-models-card"), "10_tum_modeller.png")

    # --- compare: leaderboard tablosu + F1 grafiği ---
    d.get(BASE + "/karsilastirma"); time.sleep(1.5)
    cards = d.find_elements(By.CSS_SELECTOR, "section.card")
    shot(cards[0], "11_leaderboard.png")     # Test kümesi metrikleri
    time.sleep(1.0)
    shot(cards[1], "12_f1_grafik.png")       # F1-Macro karşılaştırması

    # --- OOV shit: sadece sonuç kartı (kompakt) ---
    d.get(BASE + "/"); time.sleep(0.8)
    d.find_element(By.ID, "review-text").send_keys("shit")
    Select(d.find_element(By.ID, "model-select")).select_by_value("ensemble")
    d.find_element(By.ID, "predict-btn").click()
    WebDriverWait(d, 60).until(
        lambda dr: dr.find_element(By.ID, "result-card").is_displayed()
        and dr.find_element(By.ID, "verdict-badge").text.strip() not in ("", "—"))
    time.sleep(1.0)
    shot(d.find_element(By.ID, "result-card"), "13_oov_kart.png")
finally:
    d.quit()
print("BİTTİ")
