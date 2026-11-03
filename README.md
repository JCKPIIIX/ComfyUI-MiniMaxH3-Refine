# ComfyUI-MiniMaxH3-Refine

Nœud **MiniMax H3 Refine (fl2va)** : prend une vidéo brouillon H3 (image + son),
générée avec trop peu de steps, et prépare le latent AV et le conditioning positif
pour la corriger avec un sampler à **faible denoise** sur le checkpoint **fl2va**
(beaucoup plus léger que le ref2va, qui garde la vidéo de référence dans la séquence
à chaque step).

## Installation

Copier le dossier `ComfyUI-MiniMaxH3-Refine` dans `ComfyUI/custom_nodes/`, puis redémarrer.
Aucune dépendance supplémentaire (utilise torch et torchaudio fournis avec ComfyUI).

Prérequis : une version de ComfyUI avec le support natif de MiniMax H3
(PR #15224) et l'API `comfy_api.latest`.

## Nœud

Catégorie : `model/conditioning/minimax`

| Entrée | Rôle |
|---|---|
| `clip` | Encodeur de texte H3 |
| `vae` | VAE vidéo H3 |
| `audio_vae` | VAE audio H3 |
| `prompt` | Même prompt (structuré) que pour le brouillon, soundscape inclus |
| `images` | Frames du brouillon |
| `audio` (opt.) | Audio du brouillon. Absent = silence |
| `source_fps` | FPS du brouillon (rééchantillonné à 24 fps si différent) |
| `width` / `height` | 0 = taille du brouillon arrondie au multiple de 32 |
| `first_frame` / `last_frame` (opt.) | Keyframes **propres**, réinjectées à chaque step |

Sorties : `positive`, `latent` (AV nested), `frames` (nombre de frames utilisé).

Le nœud coupe le brouillon à la plus grande longueur 17k+5 (ex. 130 -> 124 frames).

## Workflow conseillé

```
GetVideoComponents ──images/audio──► MiniMax H3 Refine (fl2va) ──positive──► BasicGuider (cfg 1) ──► SamplerCustomAdvanced ──► séparation AV ──► VAEDecode (vidéo)
                                                              └─latent──────────────────────────────────────────────────────►                                   └─► décodage VAE audio
```

- Modèle : checkpoint **fl2va** (idéalement avec `MiniMaxH3SigmaShift`, valeurs par défaut 12 / 3).
- `BasicGuider` avec cfg 1 (le DiT H3 refuse les batchs).
- `BasicScheduler` : 6 à 8 steps, **denoise bas**.

## Réglage du denoise

Avec un shift vidéo de 12, la plage utile est très basse. Valeurs indicatives
(σ vidéo de départ, calcul théorique à confirmer sur ton modèle) :

| denoise | σ vidéo de départ |
|---|---|
| 0.03 | ≈ 0.27 |
| 0.05 | ≈ 0.39 |
| 0.08 | ≈ 0.51 |
| 0.12 | ≈ 0.62 |
| 0.20 | ≈ 0.75 |

Protocole de test :
1. **Identité** : denoise ≈ 0.01, la sortie doit ressembler au brouillon. Une dérive de
   couleur ou de saturation indique un souci de normalisation du latent.
2. **Balayage** : même seed, 0.03 / 0.05 / 0.08 / 0.12 / 0.20, comparer image et son séparément.
3. L'audio reçoit moins de bruit que la vidéo à denoise égal. S'il reste trop proche du
   brouillon, augmenter le denoise ou `shift_audio`.

## Dépannage

- **"latent vidéo inattendu"** : vérifier que `vae` est bien le VAE vidéo H3.
- **"impossible d'encoder l'audio"** : le nœud essaie les layouts `[B,2,L]` et `[B,L,2]` ;
  vérifier la signature de `audio_vae.encode` dans ta version de ComfyUI.
- **Qualité figée** : ne pas mettre de frames du brouillon en `first_frame`/`last_frame`.
