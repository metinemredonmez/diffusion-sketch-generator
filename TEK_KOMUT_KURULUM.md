# MYTH Sketch Diffusion - Tek Komutla Kurulum

## 🚀 Hızlı Kurulum (Copy-Paste)

Terminal'i aç ve şunu çalıştır:

```bash
# 1. Desktop'ta proje oluştur
cd ~/Desktop && \
mkdir -p myth_sketch_diffusion && \
cd myth_sketch_diffusion && \
echo "✓ Proje klasörü oluşturuldu: ~/Desktop/myth_sketch_diffusion"

# 2. Conda environment (bu 2-3 dakika sürer)
conda create -n sketch python=3.9 -y && \
conda activate sketch && \
echo "✓ Conda environment hazır"

# 3. Tüm kütüphaneleri kur (5-10 dakika)
pip install torch torchvision torchaudio ndjson matplotlib pillow numpy tqdm scikit-learn scipy einops clean-fid jupyter gsutil && \
echo "✓ Tüm kütüphaneler yüklendi"

# 4. Klasörleri oluştur
mkdir -p data subset checkpoints models results fid_eval && \
echo "✓ Klasör yapısı hazır"

# 5. ŞİMDİ: 
# - subset.zip'i bu klasöre koy
# - sketch_diffusion_solution.ipynb'yi bu klasöre koy
# Sonra devam et:

# 6. subset.zip'i extract et (yukarıdaki adım 5'ten sonra)
unzip subset.zip && \
echo "✓ subset klasörü hazır"

# 7. Dataset indir (bu UZUN sürer, ~15 dakika)
echo "İndirme başlıyor... Lütfen bekle..."
gsutil -m cp gs://quickdraw_dataset/full/simplified/cat.ndjson ./data/ && \
gsutil -m cp gs://quickdraw_dataset/full/simplified/bus.ndjson ./data/ && \
gsutil -m cp gs://quickdraw_dataset/full/simplified/rabbit.ndjson ./data/ && \
echo "✓ Dataset indirildi"

# 8. Jupyter başlat
jupyter notebook sketch_diffusion_solution.ipynb
```

---

## ⚡ Alternatif: Adım Adım

Eğer yukarıdaki tek komut çalışmazsa, adım adım git:

### Adım 1: Proje Klasörü
```bash
cd ~/Desktop
mkdir myth_sketch_diffusion
cd myth_sketch_diffusion
```

### Adım 2: Conda Environment
```bash
conda create -n sketch python=3.9 -y
conda activate sketch
```

### Adım 3: Kütüphaneler
```bash
pip install torch torchvision torchaudio
pip install ndjson matplotlib pillow numpy tqdm
pip install scikit-learn scipy einops clean-fid
pip install jupyter gsutil
```

### Adım 4: Klasörler
```bash
mkdir data subset checkpoints models results fid_eval
```

### Adım 5: Dosyaları Taşı
```bash
# subset.zip'i myth_sketch_diffusion/ klasörüne kopyala
# sketch_diffusion_solution.ipynb'yi myth_sketch_diffusion/ klasörüne kopyala

# Sonra extract et:
unzip subset.zip
```

### Adım 6: Dataset
```bash
gsutil -m cp gs://quickdraw_dataset/full/simplified/cat.ndjson ./data/
gsutil -m cp gs://quickdraw_dataset/full/simplified/bus.ndjson ./data/
gsutil -m cp gs://quickdraw_dataset/full/simplified/rabbit.ndjson ./data/
```

### Adım 7: Başlat
```bash
jupyter notebook sketch_diffusion_solution.ipynb
```

---

## ✅ Kurulum Kontrolü

Tüm komutları çalıştırdıktan sonra:

```bash
# Klasör yapısı doğru mu?
ls -la
# Görmeli: data/ subset/ checkpoints/ models/ results/ fid_eval/ sketch_diffusion_solution.ipynb

# Data inmiş mi?
ls -lh data/
# Görmeli: cat.ndjson (109M)  bus.ndjson (105M)  rabbit.ndjson (103M)

# Environment aktif mi?
which python
# Görmeli: .../envs/sketch/bin/python

# PyTorch çalışıyor mu?
python -c "import torch; print(torch.__version__)"
# Görmeli: 2.x.x gibi bir versiyon
```

Hepsi ✓ ise → Jupyter açılacak → Notebook'u çalıştırabilirsin!

---

## 🐛 Sık Sorunlar

### "conda: command not found"
Anaconda yüklü değil. Yükle: https://docs.conda.io/en/latest/miniconda.html

### "gsutil: command not found"
```bash
pip install gsutil
# Hala olmazsa manuel indir: https://cloud.google.com/storage/docs/gsutil_install
```

### "Cannot create conda environment"
```bash
# Eski environment'ı sil
conda env remove -n sketch
# Tekrar dene
conda create -n sketch python=3.9 -y
```

### Dataset indirilemiyor
Manuel indir:
1. Git: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified
2. İndir: cat.ndjson, bus.ndjson, rabbit.ndjson
3. Taşı: myth_sketch_diffusion/data/ klasörüne

---

## 💡 Google Colab'da Çalıştırma

GPU yoksa Colab kullan:

```python
# Colab notebook'unda:

# 1. Drive'ı mount et
from google.colab import drive
drive.mount('/content/drive')

# 2. Kütüphaneleri yükle
!pip install ndjson clean-fid einops -q

# 3. Dataset indir
!mkdir -p data
!gsutil -m cp gs://quickdraw_dataset/full/simplified/cat.ndjson ./data/
!gsutil -m cp gs://quickdraw_dataset/full/simplified/bus.ndjson ./data/
!gsutil -m cp gs://quickdraw_dataset/full/simplified/rabbit.ndjson ./data/

# 4. subset.zip'i Drive'dan yükle ve extract et
!cp /content/drive/MyDrive/subset.zip ./
!unzip subset.zip

# 5. Notebook'u çalıştır
# sketch_diffusion_solution.ipynb'yi Colab'a upload et ve çalıştır
```

---

## 🎯 Kurulum Sonrası

Kurulum tamamsa → **QUICK_START.md** oku!

O dosyada:
- Notebook'u nasıl çalıştıracağın
- Model nasıl eğitilir
- Sonuçlar nasıl alınır

detaylandırılıyor.

---

**Hazırsın! Hadi başla! 🚀**
