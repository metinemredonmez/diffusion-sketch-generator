# MYTH Sketch Diffusion - Kurulum Rehberi

## 📦 Proje Klasörünü Oluştur

Önce bir proje klasörü oluştur ve dosyaları oraya koy:

```bash
# Desktop'ta proje klasörü oluştur
cd ~/Desktop
mkdir myth_sketch_diffusion
cd myth_sketch_diffusion

# Dosyaları buraya taşı:
# - sketch_diffusion_solution.ipynb
# - subset.zip (assignment'tan gelen)
```

---

## 🐍 1. Conda Environment Kurulumu

```bash
# Environment oluştur
conda create -n sketch python=3.9 -y

# Aktifleştir
conda activate sketch

# Doğrula
python --version  # Python 3.9.x görmeli
```

---

## 📚 2. Kütüphaneleri Kur

### PyTorch (GPU Support)
```bash
# CUDA 11.8 için (NVIDIA GPU varsa)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# veya CPU only (yavaş olur, önerilmez)
pip install torch torchvision torchaudio
```

### Diğer Kütüphaneler
```bash
# Veri ve visualization
pip install ndjson matplotlib pillow numpy

# Training utilities  
pip install tqdm scikit-learn scipy

# Diffusion utilities
pip install einops

# Evaluation
pip install clean-fid

# Jupyter
pip install jupyter ipykernel

# Google Cloud (dataset indirmek için)
pip install gsutil
```

### Hepsini Tek Komutla:
```bash
pip install torch torchvision torchaudio ndjson matplotlib pillow numpy tqdm scikit-learn scipy einops clean-fid jupyter gsutil
```

---

## 📁 3. Proje Yapısını Oluştur

```bash
# Hala myth_sketch_diffusion klasöründeyken:

# Data klasörü
mkdir -p data

# subset.zip'i extract et
unzip subset.zip

# Diğer klasörler
mkdir -p checkpoints
mkdir -p models
mkdir -p results
mkdir -p fid_eval
```

### Şu ana kadar klasör yapın:
```
myth_sketch_diffusion/
├── data/                          (boş)
├── subset/                        (unzip'ten geldi)
│   ├── cat/indices.json
│   ├── bus/indices.json
│   └── rabbit/indices.json
├── checkpoints/                   (boş)
├── models/                        (boş)
├── results/                       (boş)
├── fid_eval/                      (boş)
└── sketch_diffusion_solution.ipynb
```

---

## 📥 4. Quick Draw Dataset İndir

Bu **en uzun süren kısım** (~10-15 dakika, 500MB).

```bash
# Hala myth_sketch_diffusion klasöründeyken:

# Cat
gsutil -m cp gs://quickdraw_dataset/full/simplified/cat.ndjson ./data/

# Bus  
gsutil -m cp gs://quickdraw_dataset/full/simplified/bus.ndjson ./data/

# Rabbit
gsutil -m cp gs://quickdraw_dataset/full/simplified/rabbit.ndjson ./data/
```

### gsutil Yoksa:
```bash
# Yükle
pip install gsutil

# Veya conda ile
conda install -c conda-forge gsutil
```

### İndirme İlerlemesi:
```
Copying gs://quickdraw_dataset/full/simplified/cat.ndjson...
/ [1 files][108.9 MiB/108.9 MiB]
```

---

## ✅ 5. Kurulum Kontrolü

```bash
# Environment aktif mi?
conda env list  # * sketch görmeli

# Python çalışıyor mu?
python -c "import torch; print(torch.__version__)"
python -c "import ndjson; print('ndjson OK')"

# Dosyalar var mı?
ls data/
# Çıktı: cat.ndjson  bus.ndjson  rabbit.ndjson

ls subset/
# Çıktı: bus/  cat/  rabbit/

# GPU var mı? (varsa)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🚀 6. Jupyter Notebook'u Başlat

```bash
# Hala myth_sketch_diffusion klasöründe ve sketch env aktifken:

jupyter notebook sketch_diffusion_solution.ipynb
```

Tarayıcı açılacak → Notebook görünecek → Çalıştırmaya başla!

---

## 🎯 İlk Test

Notebook'ta ilk birkaç cell'i çalıştır:

### Cell 1: Import'lar
```python
import torch
import ndjson
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Cell 2: Veri yükle
```python
# Cat'i test et
with open('./data/cat.ndjson', 'r') as f:
    data = ndjson.load(f)
print(f"Loaded {len(data)} cat sketches")
print("First sketch:", data[0]['drawing'][:2])  # İlk 2 stroke
```

Bu çalışıyorsa → ✅ Kurulum başarılı!

---

## 🐛 Sorun Giderme

### Problem: "conda: command not found"
```bash
# Anaconda/Miniconda yükle
# https://docs.conda.io/en/latest/miniconda.html
```

### Problem: "pip: command not found"
```bash
conda activate sketch
conda install pip
```

### Problem: "gsutil çalışmıyor"
```bash
# Alternatif: Manuel indir
# https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified
# İndir: cat.ndjson, bus.ndjson, rabbit.ndjson
# Taşı: myth_sketch_diffusion/data/ klasörüne
```

### Problem: "Out of Memory (OOM)"
```python
# Notebook'ta batch_size'ı küçült
batch_size = 16  # 64 yerine
```

### Problem: "CUDA out of memory"
```python
# Model'i küçült
hidden_dim = 128  # 256 yerine
num_layers = 2    # 4 yerine
```

---

## 📊 Beklenen Dosya Boyutları

```
data/
├── cat.ndjson      (~109 MB)
├── bus.ndjson      (~105 MB)
└── rabbit.ndjson   (~103 MB)

subset/
├── cat/indices.json    (~85 KB)
├── bus/indices.json    (~88 KB)
└── rabbit/indices.json (~87 KB)
```

---

## 🎓 Bir Sonraki Adım

Kurulum tamamsa → [QUICK_START.md](QUICK_START.md) dosyasını oku!

O dosyada:
- Notebook'u nasıl çalıştıracağın
- Model'i nasıl eğiteceğin
- Sonuçları nasıl alacağın

anlatılıyor.

---

## 💡 Pro Tips

### Tip 1: Terminal'i Açık Tut
Jupyter çalışırken terminal'i kapatma (Ctrl+C ile durdurabilirsin).

### Tip 2: Checkpoint Kaydet
Model eğitimi uzun sürer, her 10 epoch'ta checkpoint kaydet.

### Tip 3: GPU Kullan
Eğer GPU yoksa:
- Google Colab (ücretsiz, T4 GPU)
- Kaggle (ücretsiz, P100 GPU)

### Tip 4: Küçük Test Yap
İlk çalıştırmada:
```python
epochs = 5         # 50 yerine
batch_size = 16    # 64 yerine
```
Sistemi test et, sonra full eğitim yap.

---

## ✅ Final Checklist

Kurulum tamamlandıysa:

- [ ] ✓ Conda environment (sketch) oluşturuldu
- [ ] ✓ Tüm kütüphaneler yüklendi
- [ ] ✓ Proje klasörleri oluşturuldu
- [ ] ✓ subset/ extract edildi
- [ ] ✓ data/ klasöründe 3 NDJSON var
- [ ] ✓ Jupyter notebook açılıyor
- [ ] ✓ İlk cell'ler çalışıyor

**Hepsi ✓ ise → Eğitime başlayabilirsin! 🚀**

---

## 📞 Yardım

Takılırsan:
1. Hata mesajını oku
2. Google'da ara
3. Claude'a sor (ben buradayım!)

İyi çalışmalar! 💪
