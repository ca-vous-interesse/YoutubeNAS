# YouTubeNAS

YouTubeNAS est un PoC Python qui embarque un fichier binaire dans une vidéo 1080p, puis le reconstruit au décodage.  
Le principe est le même dans les deux scripts: une zone centrale conserve un aperçu vidéo visible, et les données sont portées autour avec protection FEC (Reed-Solomon + CRC).

Le dépôt contient **deux variantes** complémentaires:
- `YouTubeNas.py`: variante actuelle basée sur des bandes horizontales RVBY.
- `YouTubeNas 8x8.py`: variante historique basée sur une grille de tuiles 8x8.

## Prérequis

- Python 3.10+
- `ffmpeg` et `ffprobe` disponibles dans le `PATH`
- Dépendances Python:

```bash
pip install numpy opencv-python reedsolo
```

## Variante 1: `YouTubeNas.py` (bandes horizontales RVBY)

Variante principale orientée robustesse:
- Encodage en bandes horizontales de hauteur 3 px, avec symboles RVBY (2 bits par segment).
- Grande fenêtre centrale vidéo: `1656x928`.
- Header V3 répété (`x8`) pour améliorer la récupération au décodage.
- Métadonnées fichier embarquées (nom + type) pour restaurer un nom de sortie automatiquement.
- Option image fixe (`--in_image`) affichée en intro/outro (~5 s) et pendant l'extension si la data dépasse la vidéo source.

Utilisation:

```bash
# GUI
python3 YouTubeNas.py

# CLI encode
python3 YouTubeNas.py encode --in_video input.mp4 --in_data payload.bin --out_video out.mp4 --in_image cover.png

# CLI decode (écrit le fichier reconstruit dans un dossier)
python3 YouTubeNas.py decode --in_video out.mp4 --out_dir decoded
```

Options de décodage:
- `--no_force_1080p` pour ne pas forcer le scale en 1920x1080
- `--no_strict_magic` pour autoriser le mode permissif si header invalide

## Variante 2: `YouTubeNas 8x8.py` (tuiles 8x8 base-6)

Variante historique, utile comme référence:
- Encodage sur grille 8x8 avec zone active 6x6 et bord de garde gris.
- Palette data 6 couleurs: noir, blanc, rouge, vert, bleu, jaune.
- Codage `1 octet = 4 symboles base-6`.
- Fenêtre centrale vidéo: `912x512`.
- Décodage vers un fichier explicite (`--out_data`) sans restauration automatique de nom.

Utilisation:

```bash
# GUI
python3 "YouTubeNas 8x8.py"

# CLI encode
python3 "YouTubeNas 8x8.py" encode --in_video input.mp4 --in_data payload.bin --out_video out.mp4

# CLI decode
python3 "YouTubeNas 8x8.py" decode --in_video out.mp4 --out_data recovered.bin
```

Options de décodage:
- `--no_force_1080p` pour ne pas forcer le scale en 1920x1080
- `--no_strict_magic` pour autoriser le mode permissif si header invalide

## Limites

- Projet PoC: priorité à la lisibilité et à la résilience, pas aux performances maximales.
- La robustesse dépend de la chaîne de recompression/transcodage.
- Ce n'est ni un mécanisme de chiffrement, ni une stéganographie discrète.
