"""DistilBERT'i int8 dinamik quantization ile küçültür (ayrı dosya: model_int8.pt).
Orijinal fp32 (model.safetensors) yerinde kalır. Hızlı bir doğruluk kontrolü yapar."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DBERT = "models/v2/distilbert"
out = os.path.join(DBERT, "model_int8.pt")

print("fp32 model yükleniyor...")
tok = AutoTokenizer.from_pretrained(DBERT)
m = AutoModelForSequenceClassification.from_pretrained(DBERT); m.eval()

print("int8 dinamik quantization...")
qm = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
qm.eval()
torch.save(qm, out)
mb = os.path.getsize(out) / 1024 / 1024
print(f"kaydedildi -> {out}  ({mb:.1f} MB)")

# hızlı doğruluk kontrolü: fp32 vs int8 aynı tahmini veriyor mu
tests = ["This product is absolutely delicious and fresh, the best ever!",
         "Terrible quality, stale and tasteless. Waste of money.",
         "It was okay, nothing special, average taste."]
LAB = ["Negatif", "Nötr", "Pozitif"]
print("\nfp32 vs int8 karşılaştırma:")
with torch.no_grad():
    for t in tests:
        enc = tok([t], truncation=True, max_length=192, return_tensors="pt")
        p32 = torch.softmax(m(**enc).logits, -1)[0]
        p8 = torch.softmax(qm(**enc).logits, -1)[0]
        print(f"  fp32={LAB[p32.argmax()]}(%{p32.max()*100:.1f})  int8={LAB[p8.argmax()]}(%{p8.max()*100:.1f})")
print("\nBitti.")
