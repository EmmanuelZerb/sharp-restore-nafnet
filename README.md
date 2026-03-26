# Sharp Restore - Image Deblurring with NAFNet

Projet de restauration d'images (deblurring) utilisant NAFNet, un modele etat de l'art leger et performant.

## Architecture

Ce projet utilise **NAFNet** (Nonlinear Activation Free Network) - ECCV 2022, qui offre:
- Excellentes performances (meilleur que DeblurGAN, SRN-Deblur)
- Faible consommation VRAM
- Inference rapide

## Structure du projet

```
sharp-restore-project/
├── scripts/
│   ├── prepare_dataset.py   # Preparation des donnees
│   ├── nafnet.py            # Architecture du modele
│   ├── train.py             # Script d'entrainement
│   └── inference.py         # Script d'inference
├── configs/                  # Configurations
├── data/                     # Donnees (a creer)
└── requirements.txt
```

---

# GUIDE COMPLET RUNPOD

## Etape 1: Choisir le bon GPU

| GPU | VRAM | Prix/h | Recommandation |
|-----|------|--------|----------------|
| RTX 3090 | 24 GB | ~$0.40 | **Bon rapport qualite/prix** |
| RTX 4090 | 24 GB | ~$0.70 | Rapide, recommande |
| A5000 | 24 GB | ~$0.50 | Stable, bon choix |
| A100 40GB | 40 GB | ~$1.50 | Pour gros datasets |

**Recommandation**: RTX 4090 ou A5000 pour un bon equilibre cout/performance.

## Etape 2: Configuration RunPod

### 2.1 Creer un Pod

1. Va sur [runpod.io](https://runpod.io)
2. Clique "Deploy" > "GPU Pods"
3. Selectionne ton GPU (RTX 4090 recommande)
4. **Template**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
5. **Volume**: Ajoute au moins 50GB de stockage persistant
6. Lance le pod

### 2.2 Se connecter

Une fois le pod demarre:
- Clique "Connect" > "Start Web Terminal" ou
- Utilise SSH avec la commande fournie

## Etape 3: Installation

```bash
# Cloner le projet (upload tes fichiers ou clone depuis git)
cd /workspace

# Si tu as git avec ton repo:
# git clone https://github.com/TON_USER/sharp-restore-project.git

# Sinon, cree la structure et upload les fichiers via l'interface web
mkdir -p sharp-restore-project/scripts
cd sharp-restore-project

# Installer les dependances
pip install -r requirements.txt
```

## Etape 4: Preparer le Dataset

### Option A: Telecharger les datasets publics (recommande pour commencer)

```bash
cd /workspace/sharp-restore-project

# Telecharger GoPro dataset (~7GB, ~3000 paires)
python scripts/prepare_dataset.py \
    --output ./data \
    --download-gopro

# Optionnel: ajouter RealBlur (~12GB supplementaires)
python scripts/prepare_dataset.py \
    --output ./data \
    --download-realblur
```

### Option B: Utiliser tes propres images

Si tu as des images nettes (sharp) que tu veux utiliser:

```bash
# Upload tes images dans /workspace/my_sharp_images/
# Le script va creer des paires blur/sharp automatiquement

python scripts/prepare_dataset.py \
    --output ./data \
    --custom-images /workspace/my_sharp_images \
    --synthetic-severity medium
```

Les severites disponibles:
- `light`: flou leger (pour images deja un peu degradees)
- `medium`: flou moyen (recommande)
- `heavy`: flou intense (pour entrainer sur cas difficiles)

### Option C: Combiner tout

```bash
python scripts/prepare_dataset.py \
    --output ./data \
    --download-gopro \
    --custom-images /workspace/my_sharp_images \
    --synthetic-severity medium \
    --create-patches \
    --patch-size 256
```

## Etape 5: Entrainement

### Configuration recommandee par GPU

**RTX 3090/4090/A5000 (24GB VRAM):**
```bash
python scripts/train.py \
    --train-blur ./data/processed/train/blur \
    --train-sharp ./data/processed/train/sharp \
    --val-blur ./data/processed/val/blur \
    --val-sharp ./data/processed/val/sharp \
    --model nafnet64 \
    --batch-size 8 \
    --patch-size 256 \
    --epochs 200 \
    --lr 1e-3 \
    --output ./output
```

**RTX 3080/A4000 (10-16GB VRAM):**
```bash
python scripts/train.py \
    --train-blur ./data/processed/train/blur \
    --train-sharp ./data/processed/train/sharp \
    --val-blur ./data/processed/val/blur \
    --val-sharp ./data/processed/val/sharp \
    --model nafnet32 \
    --batch-size 4 \
    --patch-size 256 \
    --epochs 200 \
    --lr 1e-3 \
    --output ./output
```

**A100 (40GB+ VRAM):**
```bash
python scripts/train.py \
    --train-blur ./data/processed/train/blur \
    --train-sharp ./data/processed/train/sharp \
    --val-blur ./data/processed/val/blur \
    --val-sharp ./data/processed/val/sharp \
    --model nafnet_deblur \
    --batch-size 16 \
    --patch-size 384 \
    --epochs 200 \
    --lr 1e-3 \
    --output ./output
```

### Surveiller l'entrainement

```bash
# Dans un autre terminal, lance TensorBoard
tensorboard --logdir ./output/logs --port 6006

# Accede via l'URL RunPod ou forwarde le port
```

### Reprendre un entrainement interrompu

```bash
python scripts/train.py \
    --train-blur ./data/processed/train/blur \
    --train-sharp ./data/processed/train/sharp \
    --resume ./output/checkpoints/epoch_50.pth \
    --output ./output
```

## Etape 6: Tester le modele

### Sur une seule image

```bash
python scripts/inference.py \
    --input /path/to/blurry_image.jpg \
    --output /path/to/restored_image.jpg \
    --weights ./output/checkpoints/best.pth \
    --model nafnet64
```

### Avec comparaison avant/apres

```bash
python scripts/inference.py \
    --input /path/to/blurry_image.jpg \
    --output /path/to/restored_image.jpg \
    --weights ./output/checkpoints/best.pth \
    --model nafnet64 \
    --comparison
```

### Sur un dossier d'images

```bash
python scripts/inference.py \
    --input /path/to/blurry_folder/ \
    --output /path/to/restored_folder/ \
    --weights ./output/checkpoints/best.pth \
    --model nafnet64
```

## Etape 7: Recuperer ton modele

```bash
# Copie les poids vers ton stockage persistant
cp ./output/model_weights.pth /workspace/

# Ou telecharge via l'interface web RunPod
# Le fichier est dans: /workspace/sharp-restore-project/output/model_weights.pth
```

---

# CONSEILS POUR DE BONS RESULTATS

## 1. Quantite de donnees

| Qualite attendue | Paires d'images necessaires |
|------------------|----------------------------|
| Basique | 1,000 - 3,000 |
| Bon | 5,000 - 10,000 |
| Excellent | 20,000+ |

## 2. Diversite des donnees

Pour de meilleurs resultats, ton dataset devrait contenir:
- Differents types de scenes (interieur, exterieur, portraits)
- Differents niveaux de flou
- Differentes conditions d'eclairage

## 3. Augmentation automatique

Le script d'entrainement applique automatiquement:
- Rotations aleatoires
- Flips horizontaux/verticaux
- Crops aleatoires

## 4. Hyperparametres importants

```python
# Si le modele sous-apprend (loss ne descend pas)
--lr 2e-3  # Augmenter le learning rate

# Si le modele sur-apprend (val_loss remonte)
--lr 5e-4  # Diminuer le learning rate
--perceptual-weight 0.1  # Augmenter perceptual loss

# Pour plus de nettete
--ssim-weight 0.2  # Augmenter SSIM loss
```

## 5. Signes d'un bon entrainement

- **PSNR**: devrait augmenter (bon: >28dB, excellent: >32dB)
- **SSIM**: devrait augmenter (bon: >0.85, excellent: >0.92)
- **Loss**: devrait diminuer progressivement

---

# POUR LINKEDIN

## Ce que tu peux presenter

1. **Avant/Apres**: Screenshots de comparaisons
2. **Metriques**: PSNR et SSIM atteints
3. **Code**: Lien vers ton repo GitHub
4. **Demo**: Video ou GIF du processus

## Structure post LinkedIn

```
J'ai entraine un modele de deep learning pour restaurer des images floues.

Tech stack:
- NAFNet (ECCV 2022)
- PyTorch
- Mixed Precision Training
- RunPod GPU Cloud

Resultats:
- PSNR: XX dB
- SSIM: 0.XX
- Dataset: XX,XXX images

[Image avant/apres]

#DeepLearning #ComputerVision #ImageProcessing #PyTorch
```

---

# TROUBLESHOOTING

## CUDA Out of Memory

```bash
# Reduire batch size
--batch-size 2

# Reduire patch size
--patch-size 128

# Utiliser modele plus leger
--model nafnet32
```

## Entrainement trop lent

```bash
# Augmenter num_workers
--num-workers 8

# Utiliser patches pre-calcules
python scripts/prepare_dataset.py --create-patches
```

## Resultats flous/artefacts

- Augmenter `--epochs` (minimum 100-200)
- Verifier que le dataset est correct (blur != sharp)
- Reduire `--lr` si instable

---

# RESSOURCES

- [NAFNet Paper](https://arxiv.org/abs/2204.04676)
- [GoPro Dataset](https://seungjunnah.github.io/Datasets/gopro)
- [RunPod Docs](https://docs.runpod.io/)
