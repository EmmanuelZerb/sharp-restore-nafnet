# Guide des Prompts pour Videos Motion Graphics

## Prompts pour Sharp Restore App

### Style Tech/Startup Moderne
```
A sleek modern app interface floating in 3D space, clean minimal design,
glowing UI elements, smooth camera rotation, tech startup aesthetic,
cinematic lighting, motion blur, 4k quality, professional product showcase,
dark gradient background with subtle particles
```

### Style Futuriste/Neon
```
Futuristic holographic app interface, neon blue and purple glow,
floating in cyberpunk environment, digital particles, smooth zoom animation,
high tech UI design, glass morphism effect, ray tracing reflections,
cinematic camera movement, 8k render quality
```

### Style Clean/Minimal
```
Minimalist app showcase on white background, soft shadows,
elegant floating animation, subtle parallax effect,
clean typography, professional product video,
smooth slow motion rotation, studio lighting, Apple style presentation
```

### Style Abstract/Artistique
```
Abstract flowing shapes transforming into app interface,
liquid metal morphing animation, iridescent colors,
smooth organic motion, artistic tech visualization,
premium quality render, creative product showcase
```

### Before/After Deblurring Demo
```
Split screen comparison, blurry photo transforming to sharp image,
magical restoration effect, glowing particles,
smooth wipe transition, professional photo editing software demo,
clean modern interface, satisfying transformation animation
```

## Structure d'un Bon Prompt

1. **Sujet principal**: Ce que tu veux montrer
2. **Style visuel**: L'esthetique (moderne, futuriste, minimal...)
3. **Mouvement**: Type d'animation (rotation, zoom, floating...)
4. **Eclairage**: (cinematic, studio, neon, soft...)
5. **Qualite**: (4k, 8k, professional, high quality...)

## Parametres Recommandes

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| Frames | 16-32 | Duree de la video (16 frames = 2s a 8fps) |
| Resolution | 512x512 | Plus stable, moins de VRAM |
| CFG Scale | 7-8 | Adherence au prompt |
| Steps | 20-30 | Qualite du rendu |
| Sampler | DPM++ 2M | Bon equilibre qualite/vitesse |

## Negative Prompts Recommandes

```
ugly, blurry, low quality, distorted, watermark, text overlay,
bad anatomy, worst quality, jpeg artifacts, noise, grainy,
oversaturated, underexposed, amateur, unprofessional
```

## Tips

- **Courtes videos**: 16 frames = ~2 secondes (ideal pour les loops)
- **GPU 8GB**: Reste en 512x512, 16 frames max
- **GPU 12GB+**: Tu peux monter en 768x768, 24 frames
- **Loop parfait**: Ajoute "seamless loop" dans le prompt
