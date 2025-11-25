# MYTH Technical Assignment - Quick Start Guide

## 🚀 Hemen Başla (5 Dakikada)

### 1. Environment Kurulumu
```bash
# Python environment oluştur
conda create -n sketch python=3.9 -y
conda activate sketch

# Temel kütüphaneler
pip install torch torchvision torchaudio
pip install ndjson matplotlib pillow numpy tqdm
pip install clean-fid einops
```

### 2. Veriyi İndir
```bash
# Google Cloud'dan Quick Draw veriyi çek
mkdir -p data
gsutil -m cp gs://quickdraw_dataset/full/simplified/cat.ndjson ./data
gsutil -m cp gs://quickdraw_dataset/full/simplified/bus.ndjson ./data
gsutil -m cp gs://quickdraw_dataset/full/simplified/rabbit.ndjson ./data

# subset.zip'i extract et (assignment'tan gelen)
unzip subset.zip
```

### 3. Notebook'u Aç
```bash
jupyter notebook sketch_diffusion_solution.ipynb
```

### 4. Çalıştır!
Notebook'taki tüm cell'leri sırayla çalıştır (Run All).

---

## ⚡ Hızlı Test Modu

Tüm sistemi test etmek için (her model ~30dk):

```python
# Notebook'ta bu parametreleri kullan:
epochs = 10          # 50 yerine
batch_size = 32      # 64 yerine  
num_samples = 1000   # FID için (2000 yerine)
```

---

## 📊 Beklenen Süreler

### Full Training (GPU gerekli):
- **Cat model**: ~4-6 saat (50 epoch)
- **Bus model**: ~4-6 saat (50 epoch)
- **Rabbit model**: ~4-6 saat (50 epoch)
- **FID/KID hesaplama**: ~1-2 saat
- **Toplam**: ~15-20 saat

### Quick Test (GPU):
- **Her model**: ~30 dakika (10 epoch)
- **Toplam**: ~2-3 saat

### CPU (önerilmez):
- Her model ~2-3 gün sürer, GPU şart!

---

## 🎯 Ne Yapmalısın?

### Öncelik Sırası:
1. ✅ Veriyi anla (1 gün)
2. ✅ Model'i eğit (3 gün) - **EN ÖNEMLİ**
3. ✅ GIF'leri üret (1 gün)
4. ✅ FID/KID hesapla (1 gün)
5. ✅ Notebook'u temizle (1 gün)

### Kritik Noktalar:
- **Model eğitimi**: En uzun süren kısım, erken başla
- **GPU**: Mutlaka GPU kullan (Colab/Kaggle ücretsiz)
- **Checkpoint**: Her 10 epoch'ta checkpoint kaydet
- **Test et**: İlk 1-2 epoch'ta quick test yap

---

## 🐛 Sorun Giderme

### Problem: "Out of Memory"
```python
batch_size = 16  # Küçült
```

### Problem: "Loss NaN oluyor"
```python
lr = 1e-5  # Learning rate'i azalt
```

### Problem: "gsutil bulunamadı"
```bash
pip install gsutil
# veya
conda install -c conda-forge gsutil
```

### Problem: "Generated sketches çok kötü"
```python
epochs = 100  # Daha fazla eğit
hidden_dim = 512  # Model'i büyüt
```

---

## 📦 Deliverables Checklist

Teslim etmeden önce kontrol et:

- [ ] ✅ Jupyter Notebook (tek dosya)
- [ ] ✅ 3 kategori için FID scores
- [ ] ✅ 3 kategori için KID scores  
- [ ] ✅ 9 adet GIF (3 kategori x 3 sample)
- [ ] ✅ Generated sample görselleştirmeleri
- [ ] ✅ Training loss grafikleri
- [ ] ✅ Method açıklaması (notebook'ta)
- [ ] ✅ Results discussion (notebook'ta)
- [ ] ✅ GitHub repository linki

---

## 💡 Pro Tips

### GPU Kullanımı:
- **Google Colab**: Ücretsiz T4 GPU (12 saat limit)
- **Kaggle**: Ücretsiz P100 GPU (30 saat/hafta)
- **Own GPU**: En iyi seçenek

### Colab'da Çalıştırma:
```python
# Colab'da bu cell'i çalıştır:
from google.colab import drive
drive.mount('/content/drive')

# Çalışmayı Drive'a kaydet
save_path = '/content/drive/MyDrive/myth_assignment/'
```

### Model Kaydı:
```python
# Her 10 epoch'ta kaydet
if (epoch + 1) % 10 == 0:
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, f'checkpoint_epoch_{epoch+1}.pt')
```

---

## 🎓 Sık Sorulan Sorular

**Q: CPU'da çalışır mı?**
A: Çalışır ama çok yavaş (~100x). GPU şart.

**Q: Kaç epoch yeterli?**
A: Minimum 30, ideal 50-100 epoch.

**Q: FID score ne kadar olmalı?**
A: Sketch domain'de 50-150 arası normal.

**Q: Hangi framework kullanmalıyım?**
A: PyTorch (notebook hazır zaten).

**Q: Data augmentation gerekli mi?**
A: Hayır ama eklersen bonus puan.

---

## 📚 Faydalı Kaynaklar

### Papers:
- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [SketchRNN](https://arxiv.org/abs/1704.03477)

### Code:
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [CleanFID](https://github.com/GaParmar/clean-fid)

### Tutorials:
- [DDPM Tutorial](https://huggingface.co/blog/annotated-diffusion)
- [Transformer Tutorial](https://jalammar.github.io/illustrated-transformer/)

---

## 🎯 Son Tavsiyeler

1. **Erken başla**: Model eğitimi uzun sürer
2. **Checkpoint kaydet**: Elektrik giderse boşa gitmesin
3. **Görselleştir**: Her aşamayı görselleştir (debug için)
4. **Test et**: Küçük subset'le test et önce
5. **Dokümante et**: Ne yaptığını açıkla

---

**İyi şanslar! Her sorunun olduğunda claude'a sor, hazırım! 🚀**
