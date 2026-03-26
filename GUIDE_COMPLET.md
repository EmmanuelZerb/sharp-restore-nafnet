# GUIDE COMPLET - Entrainement Modele Deblurring NAFNet

## RESUME

Tu vas entrainer un modele NAFNet pour deflouter des images.
- Modele : NAFNet (etat de l'art ECCV 2022)
- Dataset : GoPro (2103 images train, 1111 val)
- GPU recommande : RTX 4090 ou A5000 sur RunPod (~$0.50-0.70/h)

---

## ETAPE 1 : CREER UN POD RUNPOD

1. Va sur https://runpod.io
2. Cree un compte si pas deja fait
3. Ajoute des credits ($10-20 suffisent pour commencer)
4. Clique "Deploy" > "GPU Pods"
5. Choisis un GPU :
   - **RTX 4090** (24GB) - ~$0.70/h - Recommande
   - **RTX A5000** (24GB) - ~$0.50/h - Bon aussi
   - **RTX 3090** (24GB) - ~$0.40/h - Budget
6. Template : `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
7. Ajoute 50GB de stockage persistant
8. Lance le pod

---

## ETAPE 2 : SE CONNECTER

1. Une fois le pod demarre, clique "Connect"
2. Clique "Jupyter Lab" (HTTP Service port 8888)
3. Dans Jupyter, clique File > New > Notebook
4. Selectionne le kernel Python 3

---

## ETAPE 3 : SETUP (cellules a executer une par une)

### Cellule 3.1 - Installer les dependances
```python
!pip install torch torchvision numpy opencv-python pillow tqdm tensorboard einops gdown -q
```

### Cellule 3.2 - Creer les dossiers
```python
import os
os.makedirs("/workspace/sharp-restore-project/scripts", exist_ok=True)
os.makedirs("/workspace/sharp-restore-project/data", exist_ok=True)
os.makedirs("/workspace/sharp-restore-project/output/checkpoints", exist_ok=True)
os.chdir("/workspace/sharp-restore-project")
print("Setup done!")
```

---

## ETAPE 4 : TELECHARGER LE DATASET GOPRO

### Cellule 4.1 - Telecharger
```python
import gdown
import zipfile
from pathlib import Path

url = "https://drive.google.com/uc?id=1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2"
output = "/workspace/sharp-restore-project/data/GOPRO_Large.zip"

print("Downloading GoPro dataset (~9GB)...")
gdown.download(url, output, quiet=False)

print("Extracting...")
with zipfile.ZipFile(output, 'r') as z:
    z.extractall("/workspace/sharp-restore-project/data/raw")

os.remove(output)
print("Done!")
```

### Cellule 4.2 - Preparer les paires d'images
```python
import shutil
from pathlib import Path
from tqdm import tqdm

# Trouver le bon dossier
raw_path = Path("/workspace/sharp-restore-project/data/raw")
gopro_path = None
for p in raw_path.iterdir():
    if p.is_dir() and (p / "train").exists():
        gopro_path = p
        break
if gopro_path is None:
    gopro_path = raw_path

output_dir = Path("/workspace/sharp-restore-project/data/processed")

for split in ["train", "test"]:
    split_path = gopro_path / split
    if not split_path.exists():
        print(f"Dossier {split} non trouve")
        continue

    out_split = "train" if split == "train" else "val"
    blur_out = output_dir / out_split / "blur"
    sharp_out = output_dir / out_split / "sharp"
    blur_out.mkdir(parents=True, exist_ok=True)
    sharp_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for seq_dir in tqdm(list(split_path.iterdir()), desc=f"Processing {split}"):
        if not seq_dir.is_dir():
            continue
        blur_dir = seq_dir / "blur"
        sharp_dir = seq_dir / "sharp"
        if not blur_dir.exists() or not sharp_dir.exists():
            continue
        for blur_img in blur_dir.glob("*.png"):
            sharp_img = sharp_dir / blur_img.name
            if sharp_img.exists():
                name = f"{seq_dir.name}_{blur_img.name}"
                shutil.copy(blur_img, blur_out / name)
                shutil.copy(sharp_img, sharp_out / name)
                count += 1
    print(f"{out_split}: {count} pairs")

print("Dataset ready!")
```

---

## ETAPE 5 : DEFINIR LE MODELE ET ENTRAINER

### Cellule 5.1 - Imports
```python
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

os.chdir("/workspace/sharp-restore-project")
```

### Cellule 5.2 - Definir NAFNet
```python
class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps
    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)

class NAFBlock(nn.Module):
    def __init__(self, c, dw=2, ffn=2):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c*dw, 1)
        self.conv2 = nn.Conv2d(c*dw, c*dw, 3, 1, 1, groups=c*dw)
        self.conv3 = nn.Conv2d(c*dw//2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c*dw//2, c*dw//2, 1))
        self.conv4 = nn.Conv2d(c, c*ffn, 1)
        self.conv5 = nn.Conv2d(c*ffn//2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y1, y2 = y.chunk(2, dim=1)
        y = y1 * y2
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta
        y = self.norm2(x)
        y = self.conv4(y)
        y1, y2 = y.chunk(2, dim=1)
        y = y1 * y2
        y = self.conv5(y)
        return x + y * self.gamma

class NAFNet(nn.Module):
    def __init__(self, width=32, blks=[1,1,1,28], mids=1):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, 3, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = width
        for n in blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c*2, 2, 2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(mids)])
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for n in reversed(blks):
            self.ups.append(nn.Sequential(nn.Conv2d(c, c*2, 1), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(1)]))
        self.ps = 2 ** len(blks)
    def forward(self, x):
        B, C, H, W = x.shape
        ph = (self.ps - H % self.ps) % self.ps
        pw = (self.ps - W % self.ps) % self.ps
        x = F.pad(x, (0, pw, 0, ph))
        z = self.intro(x)
        encs = []
        for enc, down in zip(self.encoders, self.downs):
            z = enc(z)
            encs.append(z)
            z = down(z)
        z = self.middle(z)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(encs)):
            z = up(z)
            z = z + skip
            z = dec(z)
        z = self.ending(z) + x
        return z[:, :, :H, :W]

print("NAFNet defined!")
```

### Cellule 5.3 - Definir le Dataset
```python
class ImageDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, patch_size=256, augment=True):
        self.blur_dir = Path(blur_dir)
        self.sharp_dir = Path(sharp_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.pairs = []
        for b in sorted(self.blur_dir.glob("*.png")):
            s = self.sharp_dir / b.name
            if s.exists():
                self.pairs.append((b, s))
        print(f"Found {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur = cv2.cvtColor(cv2.imread(str(self.pairs[idx][0])), cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(cv2.imread(str(self.pairs[idx][1])), cv2.COLOR_BGR2RGB)
        h, w = blur.shape[:2]
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        blur = blur[top:top+self.patch_size, left:left+self.patch_size]
        sharp = sharp[top:top+self.patch_size, left:left+self.patch_size]
        if self.augment:
            if random.random() > 0.5:
                blur = np.fliplr(blur).copy()
                sharp = np.fliplr(sharp).copy()
            if random.random() > 0.5:
                blur = np.flipud(blur).copy()
                sharp = np.flipud(sharp).copy()
        blur = torch.from_numpy(blur.transpose(2,0,1)).float() / 255.0
        sharp = torch.from_numpy(sharp.transpose(2,0,1)).float() / 255.0
        return blur, sharp

print("Dataset class defined!")
```

### Cellule 5.4 - Lancer l'entrainement
```python
# Configuration
TRAIN_BLUR = "./data/processed/train/blur"
TRAIN_SHARP = "./data/processed/train/sharp"
VAL_BLUR = "./data/processed/val/blur"
VAL_SHARP = "./data/processed/val/sharp"
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3
OUTPUT = "./output"

# Setup
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Model
model = NAFNet(width=32).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Data
train_loader = DataLoader(
    ImageDataset(TRAIN_BLUR, TRAIN_SHARP, 256, True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    ImageDataset(VAL_BLUR, VAL_SHARP, 256, False),
    batch_size=1, num_workers=2
)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 1e-6)
scaler = GradScaler()
best_psnr = 0

# Training loop
for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for blur, sharp in pbar:
        blur = blur.to(device)
        sharp = sharp.to(device)
        optimizer.zero_grad()
        with autocast():
            pred = model(blur)
            loss = F.l1_loss(pred, sharp)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    total_psnr = 0
    with torch.no_grad():
        for blur, sharp in val_loader:
            blur = blur.to(device)
            sharp = sharp.to(device)
            pred = torch.clamp(model(blur), 0, 1)
            mse = F.mse_loss(pred, sharp)
            psnr = 10 * torch.log10(1 / (mse + 1e-8))
            total_psnr += psnr.item()
    avg_psnr = total_psnr / len(val_loader)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}dB")

    # Save best
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), f"{OUTPUT}/checkpoints/best.pth")
        print(f"  -> New best: {best_psnr:.2f}dB")

    # Save every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"{OUTPUT}/checkpoints/epoch_{epoch+1}.pth")

# Save final
torch.save(model.state_dict(), f"{OUTPUT}/model_weights.pth")
print(f"\nTraining complete! Best PSNR: {best_psnr:.2f}dB")
```

---

## ETAPE 6 : TESTER LE MODELE

### Cellule 6.1 - Fonction d'inference
```python
def restore_image(image_path, output_path, model, device):
    # Charger l'image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convertir en tensor
    tensor = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        output = torch.clamp(output, 0, 1)

    # Convertir en image
    result = output.squeeze(0).cpu().numpy().transpose(1,2,0)
    result = (result * 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # Sauvegarder
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")

# Exemple d'utilisation :
# restore_image("./test_blur.jpg", "./test_restored.jpg", model, device)
```

### Cellule 6.2 - Charger et tester
```python
# Charger le meilleur modele
model = NAFNet(width=32).to(device)
model.load_state_dict(torch.load("./output/checkpoints/best.pth"))
model.eval()
print("Model loaded!")

# Pour tester sur une image, uploade une image floue dans Jupyter
# puis execute :
# restore_image("/workspace/ton_image.jpg", "/workspace/resultat.jpg", model, device)
```

---

## ETAPE 7 : RECUPERER LE MODELE

1. Dans Jupyter, navigue vers `/workspace/sharp-restore-project/output/`
2. Clique droit sur `model_weights.pth` > Download
3. Ou sur `checkpoints/best.pth` > Download

---

## ETAPE 8 : UTILISER LE MODELE SUR TON MAC

1. Installe les dependances :
```bash
pip install torch torchvision opencv-python numpy
```

2. Cree un fichier `inference.py` avec le code du modele NAFNet (Cellule 5.2)

3. Ajoute cette fonction :
```python
import torch
import cv2
import numpy as np

# Charger le modele
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = NAFNet(width=32).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

def deblur(input_path, output_path):
    img = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        output = torch.clamp(model(tensor), 0, 1)

    result = output.squeeze(0).cpu().numpy().transpose(1,2,0)
    result = (result * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Done: {output_path}")

# Utilisation
deblur("image_floue.jpg", "image_nette.jpg")
```

---

## RESUME DES COMMANDES IMPORTANTES

| Action | Commande |
|--------|----------|
| Voir les fichiers | `!ls -la ./data/processed/train/blur/` |
| Voir l'espace disque | `!df -h` |
| Voir la memoire GPU | `!nvidia-smi` |
| Arreter l'entrainement | Bouton Stop dans Jupyter |
| Reprendre | Relancer les cellules 5.1 a 5.4 |

---

## CONSEILS

1. **PSNR cible** : > 28dB = bon, > 30dB = tres bon, > 32dB = excellent
2. **Si memoire GPU insuffisante** : Reduire BATCH_SIZE a 4 ou 2
3. **Pour de meilleurs resultats** : Augmenter EPOCHS a 200
4. **Sauvegardes** : Le modele est sauvegarde tous les 10 epochs

---

## POUR LINKEDIN

Une fois l'entrainement termine, tu peux presenter :
- Avant/Apres sur des images test
- PSNR atteint
- Architecture NAFNet (ECCV 2022)
- Nombre de parametres (~4M)
- Dataset utilise (GoPro)

Bon courage !
