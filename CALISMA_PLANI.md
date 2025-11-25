# MYTH Assignment - Çalışma Planı ve Yol Haritası

## 📋 Proje Özeti
Quick Draw dataset kullanarak cat, bus, rabbit kategorileri için stroke-by-stroke sketch üreten diffusion modelleri eğiteceğiz.

---

## 🗓️ 7 Günlük Çalışma Planı

### **Gün 1-2: Veri Hazırlığı ve Altyapı Kurulumu**

#### Yapılacaklar:
1. **Environment Setup**
   ```bash
   # Python 3.8+ environment oluştur
   conda create -n sketch_diffusion python=3.9
   conda activate sketch_diffusion
   
   # Gerekli kütüphaneleri kur
   pip install torch torchvision torchaudio
   pip install ndjson matplotlib pillow numpy scikit-learn scipy
   pip install clean-fid einops tqdm
   ```

2. **Dataset İndirme**
   ```bash
   # Google Cloud'dan veriyi indir
   mkdir data
   gsutil -m cp gs://quickdraw_dataset/full/simplified/cat.ndjson ./data
   gsutil -m cp gs://quickdraw_dataset/full/simplified/bus.ndjson ./data
   gsutil -m cp gs://quickdraw_dataset/full/simplified/rabbit.ndjson ./data
   ```

3. **Veri Analizi**
   - NDJSON formatını anla
   - Stroke yapısını incele (x, y koordinatları)
   - Train/test split'lerini yükle
   - Veri istatistiklerini çıkar (ortalama stroke sayısı, koordinat aralıkları)

#### Çıktılar:
- ✅ Çalışan environment
- ✅ İndirilmiş dataset
- ✅ Veri analiz notları

---

### **Gün 2-3: Model Tasarımı ve İlk Implementasyon**

#### Yapılacaklar:
1. **Data Pipeline**
   - `SketchDataset` class'ını implement et
   - Stroke'ları sequence'a çevir (dx, dy, pen_state)
   - Normalization stratejisi belirle
   - DataLoader test et

2. **Model Mimarisi**
   - Transformer-based denoiser implement et
   - Timestep embedding ekle
   - Category embedding ekle
   - Forward pass test et

3. **Diffusion Setup**
   - DDPM forward process (q_sample)
   - DDPM reverse process (p_sample)
   - Beta schedule seç (cosine öneriliyor)
   - Loss function test et

#### Test Adımları:
```python
# Küçük bir subset ile test et
batch = next(iter(train_loader))
model = SketchDiffusionModel(...)
output = model(batch)
print(output.shape)  # Expected: (batch_size, seq_len, 3)
```

#### Çıktılar:
- ✅ Çalışan data pipeline
- ✅ Test edilmiş model architecture
- ✅ Diffusion trainer class

---

### **Gün 3-5: Model Eğitimi**

#### Yapılacaklar:
1. **Training Loop Setup**
   - Optimizer: AdamW (lr=1e-4)
   - Scheduler: CosineAnnealingLR
   - Gradient clipping: 1.0
   - Checkpoint saving her 10 epoch

2. **Cat Modeli (Gün 3)**
   ```bash
   # 50 epoch eğit (yaklaşık 4-6 saat GPU'da)
   python train.py --category cat --epochs 50 --batch_size 64
   ```

3. **Bus Modeli (Gün 4)**
   ```bash
   python train.py --category bus --epochs 50 --batch_size 64
   ```

4. **Rabbit Modeli (Gün 5)**
   ```bash
   python train.py --category rabbit --epochs 50 --batch_size 64
   ```

#### Monitoring:
- Loss'un düzenli azaldığını kontrol et
- Overfit olup olmadığını kontrol et
- Her 10 epoch'ta sample generation yap

#### Çıktılar:
- ✅ 3 eğitilmiş model checkpoint'i
- ✅ Training loss grafikleri
- ✅ Intermediate sample'lar

---

### **Gün 5-6: Generation ve Visualization**

#### Yapılacaklar:
1. **Sample Generation**
   - Her kategori için 20 sketch üret
   - Sequence'ları stroke'lara çevir
   - PNG olarak kaydet

2. **GIF Animasyonları**
   - Stroke-by-stroke generation GIF'leri oluştur
   - Her kategori için 3 adet (örnek assignment'taki gibi)
   - Duration: 50ms per frame

3. **Visualization**
   - Real vs Generated karşılaştırması
   - Grid layout (4x5)
   - High resolution export (300 DPI)

#### Kod Örneği:
```python
# Her kategori için
samples = generate_samples(model, diffusion, category_id, num_samples=20)

for i, sample in enumerate(samples):
    # Static image
    strokes = sequence_to_strokes(sample)
    img = draw_sketch(strokes)
    img.save(f'{category}_sample_{i}.png')
    
    # Animated GIF
    create_generation_gif(sample, f'{category}_gen_{i}.gif')
```

#### Çıktılar:
- ✅ 60 adet generated sketch (20x3)
- ✅ 9 adet animasyon GIF (3x3)
- ✅ Comparison figürleri

---

### **Gün 6: Quantitative Evaluation**

#### Yapılacaklar:
1. **FID/KID Hazırlık**
   - Test set sketch'lerini PNG'ye çevir (2000 adet x 3 kategori)
   - Generated sketch'leri PNG'ye çevir (2000 adet x 3 kategori)
   - Tüm imajları 299x299 resize et (Inception input size)

2. **Metric Hesaplama**
   ```python
   from cleanfid import fid
   
   # Her kategori için
   fid_score = fid.compute_fid(real_dir, fake_dir, mode='clean')
   kid_score = fid.compute_kid(real_dir, fake_dir, mode='clean')
   ```

3. **Results Table**
   ```
   Category    FID       KID
   --------------------------------
   Cat         XX.XX     0.XXXX
   Bus         XX.XX     0.XXXX
   Rabbit      XX.XX     0.XXXX
   ```

#### Çıktılar:
- ✅ FID scores (3 adet)
- ✅ KID scores (3 adet)
- ✅ Results JSON file

---

### **Gün 7: Finalizasyon ve Dokümantasyon**

#### Yapılacaklar:
1. **Notebook Temizliği**
   - Tüm cell'leri sırayla çalıştır
   - Output'ları kontrol et
   - Gereksiz kod/comment'leri temizle
   - Markdown açıklamalarını gözden geçir

2. **Sonuç Analizi**
   - FID/KID skorlarını yorumla
   - Model başarılarını/kısıtlamalarını yaz
   - İyileştirme önerilerini ekle
   - Metodoloji açıklamalarını detaylandır

3. **Final Checks**
   - [ ] 3 adet trained model var mı?
   - [ ] 3 adet GIF var mı?
   - [ ] FID/KID skorları hesaplandı mı?
   - [ ] Tüm görselleştirmeler model'den mi üretildi?
   - [ ] Notebook baştan sona çalışıyor mu?
   - [ ] Random seed set edildi mi?

4. **GitHub Upload**
   ```bash
   git init
   git add sketch_diffusion_solution.ipynb
   git add results/
   git add README.md
   git commit -m "MYTH Technical Assignment - Sketch Generation"
   git push origin main
   ```

#### Çıktılar:
- ✅ Final notebook
- ✅ GitHub repository
- ✅ README.md

---

## 📁 Proje Klasör Yapısı

```
sketch-diffusion/
│
├── data/                          # Quick Draw NDJSON files
│   ├── cat.ndjson
│   ├── bus.ndjson
│   └── rabbit.ndjson
│
├── subset/                        # Train/test indices
│   ├── cat/indices.json
│   ├── bus/indices.json
│   └── rabbit/indices.json
│
├── checkpoints/                   # Training checkpoints
│   ├── cat_epoch_10.pt
│   ├── cat_epoch_20.pt
│   └── ...
│
├── models/                        # Final trained models
│   ├── cat_final.pt
│   ├── bus_final.pt
│   └── rabbit_final.pt
│
├── results/                       # Generated outputs
│   ├── cat_generated_samples.png
│   ├── cat_generation_1.gif
│   ├── cat_training_loss.png
│   ├── bus_generated_samples.png
│   ├── ...
│   └── evaluation_results.json
│
├── fid_eval/                      # FID/KID evaluation images
│   ├── cat/
│   │   ├── real/
│   │   └── fake/
│   └── ...
│
├── sketch_diffusion_solution.ipynb  # Main notebook
└── README.md
```

---

## ⚙️ Hyperparameter Önerileri

### Model Config:
```python
config = {
    'seq_len': 200,           # Sequence length
    'input_dim': 3,           # (dx, dy, pen)
    'hidden_dim': 256,        # Transformer hidden size
    'num_layers': 4,          # Transformer layers
    'num_heads': 4,           # Attention heads
    'dropout': 0.1,           # Dropout rate
}
```

### Training Config:
```python
train_config = {
    'epochs': 50,             # Training epochs (10-20 for quick test)
    'batch_size': 64,         # Batch size (32 if OOM)
    'lr': 1e-4,               # Learning rate
    'timesteps': 1000,        # DDPM timesteps
    'beta_schedule': 'cosine', # cosine or linear
}
```

---

## 🎯 Beklenen Sonuçlar

### Başarı Kriterleri:
1. **FID < 100**: İyi kalite sketch generation
2. **KID < 0.05**: Real ve fake distribution'lar yakın
3. **GIF'ler**: Smooth stroke-by-stroke generation
4. **Görsel Kalite**: Tanınabilir objeler

### Sık Karşılaşılan Sorunlar:

**Problem 1: Loss düşmüyor**
- Çözüm: Learning rate'i azalt (1e-5 dene)
- Çözüm: Batch size'ı küçült
- Çözüm: Gradient clipping ekle

**Problem 2: Generated sketch'ler kötü**
- Çözüm: Daha fazla epoch eğit (100+)
- Çözüm: Model kapasitesini artır (hidden_dim=512)
- Çözüm: Data augmentation ekle

**Problem 3: Out of Memory**
- Çözüm: Batch size'ı küçült (32 veya 16)
- Çözüm: Gradient accumulation kullan
- Çözüm: Mixed precision training (fp16)

---

## 🚀 Hızlı Test İçin

Eğer tüm pipeline'ı hızlıca test etmek istersen:

```python
# Quick test config
quick_config = {
    'epochs': 5,              # Sadece 5 epoch
    'batch_size': 32,
    'num_samples': 500,       # Training'de 500 sample
    'test_samples': 100,      # Test'te 100 sample
}
```

Bu şekilde her model ~30 dakikada eğitilir.

---

## 📚 Referanslar

1. **DDPM Paper**: Ho et al., "Denoising Diffusion Probabilistic Models"
2. **SketchRNN**: Ha & Eck, "A Neural Representation of Sketch Drawings"
3. **Transformer**: Vaswani et al., "Attention is All You Need"
4. **Clean-FID**: https://github.com/GaParmar/clean-fid
5. **Quick Draw Dataset**: https://quickdraw.withgoogle.com/data/

---

## ✅ Final Checklist

Teslim etmeden önce:

- [ ] Notebook tüm cell'leri çalışıyor
- [ ] 3 kategori için 3 FID score var
- [ ] 3 kategori için 3 KID score var
- [ ] 3 kategori için GIF animasyonları var
- [ ] Generated sample'lar kendi modelinden
- [ ] Random seed fixed (reproducibility)
- [ ] Açıklayıcı markdown cell'ler var
- [ ] Sonuçlar discuss edilmiş
- [ ] GitHub'a yüklenmiş

---

## 🎓 Bonus İyileştirmeler (Ekstra Puan için)

1. **Classifier-Free Guidance**: Generation kalitesini artırır
2. **Progressive Distillation**: Inference hızlandırır
3. **Multi-Category Model**: Tek model 3 kategori
4. **Stroke-Level Attention**: Stroke boundaries'e özel attention
5. **Perceptual Loss**: FID dışında ek metric

---

İyi çalışmalar! 🚀
