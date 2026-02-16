#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

# Ce script encode un fichier binaire dans une video 1080p lossless en mappant
# des symboles RVBY sur des bandes horizontales autour du centre. Un centre video
# reste visible (video source ou image fixe). Un decodage inverse reconstruit
# les donnees.
#
# Ajout important: correction d'erreurs (FEC) pour resister a la recompression
# AVC1 de YouTube:
# - CRC32 par bloc (detecte les corruptions residuelles)
# - Reed-Solomon RS(255,223) (corrige des octets errones)
# - Protection par blocs RS + CRC (compatible avec les anciens headers)
# - Parite renforcee au debut (UEP)
#
# Remarque: la FEC augmente la taille du flux encode, ce qui reduit la charge
# utile maximale si on ne veut pas allonger la video.

try:
    # Dependence externe pour Reed-Solomon. Si absente, on leve une erreur claire.
    import reedsolo
    _HAS_REEDSOLO = True
except Exception:
    reedsolo = None
    _HAS_REEDSOLO = False

# =========================
# Format / parametres PoC
# =========================
W, H = 1920, 1080
T = 8  # legacy (ancien mode tuiles)
SYMBOLS_PER_BYTE = 4  # legacy (ancien mode base-6)
LINE_BAND_HEIGHT = 3          # hauteur d'une bande horizontale de données (en pixels)
LINE_BITS_PER_SEGMENT = 2     # 2 bits par segment (4 couleurs)
LINE_SEGMENTS_PER_BYTE = 4    # 4 segments x 2 bits -> 1 byte
LINE_PALETTE_RVBY_BGR = np.array([
    [0, 0, 255],      # R (rouge)
    [0, 255, 0],      # V (vert)
    [255, 0, 0],      # B (bleu)
    [0, 255, 255],    # Y (jaune)
], dtype=np.uint8)

# Palette BGR legacy (ancien mode 6 couleurs, conservee pour compatibilite).
SYMBOL_COLORS_BGR = np.array([
    [0, 0, 0],        # noir
    [255, 255, 255],  # blanc
    [0, 0, 255],      # rouge
    [0, 255, 0],      # vert
    [255, 0, 0],      # bleu
    [0, 255, 255],    # jaune
], dtype=np.uint8)
GUARD_BGR = np.array([128, 128, 128], dtype=np.uint8)  # legacy (ancien bord de garde)

# Header V1 (historique, sans FEC)
MAGIC_V1 = b"VDAT01\0\0"  # 8 bytes
HDR_V1_FMT = "<8sQ"       # magic + payload_size uint64
HDR_V1_SIZE = struct.calcsize(HDR_V1_FMT)

# Header V2 (avec FEC simple)
MAGIC_V2 = b"VDAT02\0\0"  # 8 bytes
# Champs: magic, payload_size, block_data_bytes, rs_parity_bytes, champ_reserve
HDR_V2_FMT = "<8sQ I H H"
HDR_V2_SIZE = struct.calcsize(HDR_V2_FMT)

# Header V3 (avec FEC + parite renforcee au debut)
MAGIC_V3 = b"VDAT03\0\0"  # 8 bytes
# Champs: magic, payload_size, block_data_bytes, rs_parity_bytes, champ_reserve,
#         strong_rs_parity_bytes, strong_prefix_pct
HDR_V3_FMT = "<8sQ I H H H H"
HDR_V3_SIZE = struct.calcsize(HDR_V3_FMT)

# Parametres FEC (RS(255,223) + CRC32)
RS_N = 255
RS_PARITY_BYTES = 32  # RS(255,223) -> 32 bytes parite, corrige ~16 bytes
RS_K = RS_N - RS_PARITY_BYTES
CRC_BYTES = 4
BLOCK_DATA_BYTES = RS_K - CRC_BYTES  # 223 - 4 = 219 bytes utiles par bloc
HEADER_RESERVED_VALUE = 1  # champ reserve conserve pour compatibilite

# Parite renforcee sur le debut du payload
RS_PARITY_BYTES_STRONG = 64  # RS(255,191) -> 64 bytes parite, corrige ~32 bytes
STRONG_PREFIX_PCT = 10       # protege les premiers 10% bytes avec parite forte

# Centre vidéo fixe (multiples de 8)
CENTER_W, CENTER_H = 1656, 928
CENTER_X, CENTER_Y = 132, 76

# Métadonnées fichier (prefixe fixe avant payload binaire)
FILE_META_SIZE = 100
FILE_META_MAGIC = b"YNM1"
FILE_META_NAME_BYTES = 72
FILE_META_TYPE_BYTES = 20
HEADER_REPEAT_COUNT = 8  # repetition du header pour meilleure robustesse au decode
IMAGE_PAD_SECONDS = 5.0  # image fixe en intro/outro (secondes)

# Coins sync legacy (ancien mode tuiles), conserves sans usage actif.
# TL=0, TR=3, BL=1, BR=2
SYNC = {
    (0, 0): 0,
    (0, (W // T) - 1): 3,
    ((H // T) - 1, 0): 1,
    ((H // T) - 1, (W // T) - 1): 2,
}

# Seuils de decodage couleur tolerants (post-compression).
DEC_BLACK_V_MAX = 70
DEC_WHITE_V_MIN = 150
DEC_WHITE_S_MAX = 60
DEC_COLOR_S_MIN = 35
DEC_HUE_TOL = 26

def _build_base6_tables() -> Tuple[np.ndarray, np.ndarray]:
    """
    But:
        Construire les tables legacy de conversion octet <-> 4 symboles base-6.

    Fonctionnement:
        Parcourt les 256 octets, calcule leurs digits base-6 et remplit
        les tables historiques (non utilisees par le mode RVBY actuel):
        - table directe: byte -> 4 symboles
        - table inverse: quartet symboles -> byte (ou -1 si invalide)

    Paramètres d'entrée:
        Aucun.

    Sortie:
        Tuple[np.ndarray, np.ndarray]: tables directe et inverse.
    """
    byte_to_syms = np.zeros((256, SYMBOLS_PER_BYTE), dtype=np.uint8)
    syms_to_byte = np.full((6, 6, 6, 6), -1, dtype=np.int16)
    for b in range(256):
        x = b
        d3 = x % 6
        x //= 6
        d2 = x % 6
        x //= 6
        d1 = x % 6
        x //= 6
        d0 = x % 6
        byte_to_syms[b] = np.array([d0, d1, d2, d3], dtype=np.uint8)
        syms_to_byte[d0, d1, d2, d3] = b
    return byte_to_syms, syms_to_byte

def classify_data_symbols_bgr_tolerant(data_colors: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    But:
        Classer des couleurs BGR en symboles legacy (palette 6 couleurs).

    Fonctionnement:
        Utilise HSV avec règles noir/blanc, puis classification de teinte
        pour rouge/jaune/vert/bleu, avec fallback RGB en cas ambigu.
        Cette fonction est conservée pour compatibilité mais n'est pas utilisée
        par le chemin RVBY 2 bits actuel.

    Paramètres d'entrée:
        data_colors (np.ndarray): couleurs moyennes BGR, shape (N, 3).

    Sortie:
        Tuple[np.ndarray, int]: symboles (uint8 0..5), compteur incertain.
    """
    if data_colors.size == 0:
        return np.zeros((0,), dtype=np.uint8), 0

    bgr_u8 = np.clip(np.rint(data_colors), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_u8.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int16)
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]

    syms = np.full((bgr_u8.shape[0],), -1, dtype=np.int16)
    uncertain = 0

    # Noir prioritaire si luminance basse.
    mask_black = (v <= DEC_BLACK_V_MAX)
    syms[mask_black] = 0

    # Blanc: faible saturation + luminance haute.
    mask_white = (s <= DEC_WHITE_S_MAX) & (v >= DEC_WHITE_V_MIN) & (~mask_black)
    syms[mask_white] = 1

    rem = np.where(syms < 0)[0]
    if rem.size > 0:
        h_r = h[rem]
        s_r = s[rem]
        v_r = v[rem]

        # Teintes OpenCV: rouge=0, jaune=30, vert=60, bleu=120.
        hue_centers = np.array([0, 30, 60, 120], dtype=np.int16)
        hue_to_sym = np.array([2, 5, 3, 4], dtype=np.int16)
        hd = np.abs(h_r[:, None] - hue_centers[None, :])
        hd = np.minimum(hd, 180 - hd)
        best_idx = hd.argmin(axis=1)
        best_dist = hd[np.arange(rem.size), best_idx]

        strong_color = (s_r >= DEC_COLOR_S_MIN)
        rem_syms = np.empty((rem.size,), dtype=np.int16)
        rem_syms[strong_color] = hue_to_sym[best_idx[strong_color]]
        rem_syms[~strong_color] = np.where(v_r[~strong_color] >= 128, 1, 0)

        uncertain += int(np.sum(best_dist[strong_color] > DEC_HUE_TOL))
        uncertain += int(np.sum(~strong_color))

        # Fallback RGB pour teintes trop éloignées.
        weak = np.where(strong_color & (best_dist > DEC_HUE_TOL))[0]
        if weak.size > 0:
            weak_abs = rem[weak]
            vivid_palette = SYMBOL_COLORS_BGR[[2, 3, 4, 5]].astype(np.float32)
            d2 = np.sum((data_colors[weak_abs, None, :] - vivid_palette[None, :, :]) ** 2, axis=2)
            rem_syms[weak] = np.array([2, 3, 4, 5], dtype=np.int16)[d2.argmin(axis=1)]

        syms[rem] = rem_syms

    return syms.astype(np.uint8), uncertain

BYTE_TO_SYMBOLS, SYMBOLS4_TO_BYTE = _build_base6_tables()  # tables legacy

# =========================
# Utilitaires FFmpeg
# =========================

@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float

def _run(cmd: list) -> Tuple[int, str, str]:
    """
    But:
        Exécuter une commande externe et récupérer ses flux.

    Fonctionnement:
        Lance le process, attend sa fin puis renvoie code/stdout/stderr.

    Paramètres d'entrée:
        cmd (list): commande et arguments.

    Sortie:
        Tuple[int, str, str]: code retour, stdout, stderr.
    """
    # Execute un sous-processus et retourne (code, stdout, stderr).
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def _start_process_stderr_pump(proc: subprocess.Popen,
                               label: str,
                               log_cb):
    """
    But:
        Relayer le flux stderr d'un process vers le logger applicatif.

    Fonctionnement:
        Lit stderr en binaire, segmente sur \\n / \\r puis transmet chaque
        ligne non vide à `log_cb` avec un préfixe de contexte.

    Paramètres d'entrée:
        proc (subprocess.Popen): process à surveiller.
        label (str): préfixe de contexte log.
        log_cb: callback de log (ou `None`).

    Sortie:
        Optional[threading.Thread]: thread pump démarré, sinon `None`.
    """
    if not log_cb or proc.stderr is None:
        # Decision: pas de callback ou pas de pipe stderr -> pas de pump.
        return None

    def _pump():
        """
        But:
            Lire stderr jusqu'à EOF et publier les lignes.

        Fonctionnement:
            Agrège les chunks puis émet les segments séparés par retour ligne.

        Paramètres d'entrée:
            Aucun (capture la fermeture).

        Sortie:
            None.
        """
        try:
            pending = b""
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                pending += chunk
                while True:
                    idx_n = pending.find(b"\n")
                    idx_r = pending.find(b"\r")
                    idxs = [i for i in (idx_n, idx_r) if i >= 0]
                    if not idxs:
                        break
                    idx = min(idxs)
                    line = pending[:idx]
                    pending = pending[idx + 1:]
                    msg = line.decode("utf-8", errors="replace").strip()
                    if msg:
                        log_cb(f"{label}: {msg}")
            if pending:
                msg = pending.decode("utf-8", errors="replace").strip()
                if msg:
                    log_cb(f"{label}: {msg}")
        except Exception:
            # Decision: un pump ne doit jamais faire echouer l'encodage/decode.
            pass

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    return t

def _join_process_stderr_pump(pump_thread):
    """
    But:
        Joindre proprement le thread de relay stderr.

    Fonctionnement:
        Attend brièvement la fin du thread si présent.

    Paramètres d'entrée:
        pump_thread: thread optionnel.

    Sortie:
        None.
    """
    if pump_thread is not None:
        pump_thread.join(timeout=1.0)

def ffprobe_info(path: str) -> VideoInfo:
    """
    But:
        Lire les informations vidéo de base (largeur, hauteur, fps).

    Fonctionnement:
        Interroge ffprobe, parse les champs utiles puis normalise le fps.

    Paramètres d'entrée:
        path (str): chemin de la vidéo.

    Sortie:
        VideoInfo: dimensions et fréquence d'images.
    """
    # Interroge ffprobe pour recuperer largeur/hauteur/fps de la video.
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=0",
        path
    ]
    code, out, err = _run(cmd)
    if code != 0:
        # Decision: on stoppe si ffprobe echoue (info video indispensable).
        raise RuntimeError(f"ffprobe failed: {err.strip()}")

    kv = {}
    for line in out.splitlines():
        if "=" in line:
            # Decision: on ignore les lignes qui ne sont pas des paires cle=valeur.
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    w = int(kv.get("width", "0"))
    h = int(kv.get("height", "0"))
    # Decision: on prefere avg_frame_rate, sinon r_frame_rate, sinon 30/1.
    fr = kv.get("avg_frame_rate") or kv.get("r_frame_rate") or "30/1"
    try:
        num, den = fr.split("/")
        # Formule: fps = num/den (si den=0 -> fallback 30.0).
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    except Exception:
        # Decision: si parsing rate invalide, fallback 30.0.
        fps = 30.0

    if w <= 0 or h <= 0:
        # Decision: dimensions invalides -> erreur (impossible de decoder).
        raise RuntimeError("ffprobe could not read width/height.")

    if fps <= 0 or fps > 1000:
        # Decision: fps aberrant -> fallback raisonnable.
        fps = 30.0

    return VideoInfo(width=w, height=h, fps=fps)

def ffprobe_nb_frames(path: str) -> Optional[int]:
    """
    But:
        Récupérer le nombre de frames depuis les métadonnées.

    Fonctionnement:
        Lit `nb_frames` via ffprobe et valide la valeur.

    Paramètres d'entrée:
        path (str): chemin de la vidéo.

    Sortie:
        Optional[int]: nombre de frames ou `None` si indisponible.
    """
    # Tente de recuperer nb_frames via metadata (rapide, mais parfois absent).
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    code, out, err = _run(cmd)
    if code != 0:
        # Decision: ffprobe n'a rien renvoye de fiable.
        return None
    val = out.strip()
    if not val:
        # Decision: valeur vide -> inconnue.
        return None
    try:
        n = int(val)
        # Decision: on n'accepte que des valeurs positives.
        return n if n > 0 else None
    except Exception:
        return None

def ffprobe_duration_seconds(path: str) -> Optional[float]:
    """
    But:
        Récupérer la durée d'une vidéo en secondes.

    Fonctionnement:
        Interroge ffprobe (`format=duration`) et valide la valeur.

    Paramètres d'entrée:
        path (str): chemin de la vidéo.

    Sortie:
        Optional[float]: durée en secondes ou `None`.
    """
    # Recupere la duree (format) pour estimer le nombre de frames si besoin.
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    code, out, err = _run(cmd)
    if code != 0:
        # Decision: ffprobe n'a rien renvoye de fiable.
        return None
    val = out.strip()
    if not val:
        # Decision: valeur vide -> inconnue.
        return None
    try:
        dur = float(val)
        # Decision: on n'accepte que des valeurs positives.
        return dur if dur > 0 else None
    except Exception:
        return None

def format_bytes(n: int) -> str:
    """
    But:
        Formater une taille en octets dans une unité lisible.

    Fonctionnement:
        Convertit en B/KiB/MiB/GiB/TiB selon la magnitude.

    Paramètres d'entrée:
        n (int): taille en octets.

    Sortie:
        str: taille formatée.
    """
    # Formate une taille en unites lisibles (base 1024).
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    val = float(n)
    for u in units:
        # Decision: choisir l'unite courante si < 1024 ou derniere unite.
        if val < 1024 or u == units[-1]:
            if u == "B":
                # Decision: pas de decimales pour les bytes.
                return f"{int(val)} {u}"
            return f"{val:.2f} {u}"
        val /= 1024.0
    return f"{int(n)} B"

def _safe_filename_component(s: str) -> str:
    """
    But:
        Normaliser un composant de nom de fichier.

    Fonctionnement:
        Remplace les caractères risqués par `_` et supprime les points/espaces
        de bord pour éviter les noms invalides.

    Paramètres d'entrée:
        s (str): composant brut.

    Sortie:
        str: composant normalisé.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ ")
    cleaned = "".join(ch if ch in allowed else "_" for ch in s)
    cleaned = cleaned.strip(" .")
    return cleaned or "file"

def _build_file_meta_block(file_path: str) -> bytes:
    """
    But:
        Construire le préfixe fixe de 100 octets (nom + type).

    Fonctionnement:
        Encode le nom de base et l'extension (sans point) en UTF-8 tronqué
        dans un bloc binaire de taille fixe.

    Paramètres d'entrée:
        file_path (str): chemin du fichier source.

    Sortie:
        bytes: bloc metadata de taille `FILE_META_SIZE`.
    """
    base = os.path.basename(file_path)
    stem, ext = os.path.splitext(base)
    name = _safe_filename_component(stem)
    raw_type = ext[1:] if ext.startswith(".") else ext
    ftype = _safe_filename_component(raw_type) if raw_type else ""

    name_b = name.encode("utf-8", errors="ignore")[:FILE_META_NAME_BYTES]
    type_b = ftype.encode("utf-8", errors="ignore")[:FILE_META_TYPE_BYTES]

    block = bytearray(FILE_META_SIZE)
    block[0:4] = FILE_META_MAGIC
    block[4] = len(name_b)
    block[5] = len(type_b)
    # bytes 6..7 réservés (0)
    block[8:8 + len(name_b)] = name_b
    block[8 + FILE_META_NAME_BYTES:8 + FILE_META_NAME_BYTES + len(type_b)] = type_b
    return bytes(block)

def _split_meta_and_data(payload: bytes) -> Tuple[Optional[str], bytes]:
    """
    But:
        Extraire le nom cible depuis les 100 octets de metadata.

    Fonctionnement:
        Si la signature metadata est valide, retourne le nom de fichier
        reconstruit et les données utiles (sans le bloc metadata).

    Paramètres d'entrée:
        payload (bytes): payload décodé.

    Sortie:
        Tuple[Optional[str], bytes]: (nom_fichier, data_sans_meta).
    """
    if len(payload) < FILE_META_SIZE:
        return None, payload
    block = payload[:FILE_META_SIZE]
    if block[0:4] != FILE_META_MAGIC:
        return None, payload

    name_len = int(block[4])
    type_len = int(block[5])
    if name_len > FILE_META_NAME_BYTES or type_len > FILE_META_TYPE_BYTES:
        return None, payload

    name_b = block[8:8 + name_len]
    type_b = block[8 + FILE_META_NAME_BYTES:8 + FILE_META_NAME_BYTES + type_len]
    name = _safe_filename_component(name_b.decode("utf-8", errors="ignore"))
    ftype = _safe_filename_component(type_b.decode("utf-8", errors="ignore")) if type_len > 0 else ""

    filename = f"{name}.{ftype}" if ftype else name
    return filename, payload[FILE_META_SIZE:]

def _write_decoded_payload(out_dir: str, payload: bytes, log_cb=None) -> str:
    """
    But:
        Écrire le payload décodé dans un répertoire cible.

    Fonctionnement:
        Utilise le nom présent dans le bloc metadata si disponible, sinon
        un nom de secours.

    Paramètres d'entrée:
        out_dir (str): répertoire de sortie.
        payload (bytes): payload décodé.
        log_cb: callback de log optionnel.

    Sortie:
        str: chemin final écrit.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_name, data = _split_meta_and_data(payload)
    filename = meta_name if meta_name else "decoded_payload.bin"
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "wb") as f:
        f.write(data)
    if log_cb:
        if meta_name:
            log_cb(f"Nom restauré depuis metadata: {filename}")
        else:
            log_cb("Metadata fichier absente/invalide: nom de secours utilisé.")
    return out_path

def _ceil_div(a: int, b: int) -> int:
    """
    But:
        Effectuer une division entière arrondie au supérieur.

    Fonctionnement:
        Utilise `(a + b - 1) // b` avec garde pour `a <= 0`.

    Paramètres d'entrée:
        a (int): numérateur.
        b (int): dénominateur.

    Sortie:
        int: résultat arrondi au supérieur.
    """
    # Formule: ceil(a / b) avec entiers.
    if b <= 0:
        # Decision: denominateur invalide -> erreur explicite.
        raise ValueError("ceil_div denominator must be > 0")
    if a <= 0:
        # Decision: numerateur nul/negatif -> resultat 0.
        return 0
    return (a + b - 1) // b

def _strong_block_data(rs_parity: int) -> int:
    """
    But:
        Calculer la taille utile data d'un bloc RS fort.

    Fonctionnement:
        Soustrait la parité RS et le CRC à la taille RS_N.

    Paramètres d'entrée:
        rs_parity (int): octets de parité.

    Sortie:
        int: taille utile data.
    """
    # Formule: block_data = RS_N - rs_parity - CRC_BYTES.
    return RS_N - rs_parity - CRC_BYTES

def _compute_blocks_for_payload(payload_size: int,
                                block_data_norm: int,
                                block_data_strong: int,
                                strong_prefix_pct: int) -> Tuple[int, int, int]:
    """
    But:
        Calculer combien de blocs FEC sont nécessaires.

    Fonctionnement:
        Répartit le payload en préfixe protégé fort + reste normal.

    Paramètres d'entrée:
        payload_size (int): taille payload brute.
        block_data_norm (int): data utile par bloc normal.
        block_data_strong (int): data utile par bloc fort.
        strong_prefix_pct (int): pourcentage du préfixe fort.

    Sortie:
        Tuple[int, int, int]: (blocs forts, blocs normaux, total).
    """
    # Calcule nb blocs forts + nb blocs normaux necessaires pour un payload.
    strong_prefix_bytes = _ceil_div(payload_size * strong_prefix_pct, 100)
    strong_blocks = _ceil_div(strong_prefix_bytes, block_data_strong)
    remaining = max(0, payload_size - strong_prefix_bytes)
    normal_blocks = _ceil_div(remaining, block_data_norm)
    total_blocks = strong_blocks + normal_blocks
    return strong_blocks, normal_blocks, total_blocks

def _max_payload_from_blocks(total_blocks: int,
                             block_data_norm: int,
                             block_data_strong: int,
                             strong_prefix_pct: int) -> int:
    """
    But:
        Trouver le payload maximal pour un budget de blocs donné.

    Fonctionnement:
        Effectue une recherche binaire sur la taille payload admissible.

    Paramètres d'entrée:
        total_blocks (int): nombre total de blocs disponibles.
        block_data_norm (int): data utile par bloc normal.
        block_data_strong (int): data utile par bloc fort.
        strong_prefix_pct (int): pourcentage du préfixe fort.

    Sortie:
        int: taille payload maximale.
    """
    # Recherche binaire du payload max pour un nb de blocs donne.
    if total_blocks <= 0:
        # Decision: aucun bloc dispo -> payload 0.
        return 0
    # Bornes: 0..(total_blocks * block_data_norm).
    lo, hi = 0, total_blocks * block_data_norm
    while lo < hi:
        mid = (lo + hi + 1) // 2
        strong_blocks, normal_blocks, needed = _compute_blocks_for_payload(
            mid, block_data_norm, block_data_strong, strong_prefix_pct
        )
        if needed <= total_blocks:
            lo = mid
        else:
            hi = mid - 1
    return lo

def estimate_max_payload_bytes(in_video: str) -> Optional[int]:
    """
    But:
        Estimer la capacité utile maximum d'une vidéo source.

    Fonctionnement:
        Estime le nombre de frames, la capacité binaire/frame, puis retranche
        l'overhead FEC pour obtenir le payload brut max.

    Paramètres d'entrée:
        in_video (str): chemin de la vidéo source.

    Sortie:
        Optional[int]: capacité en octets ou `None` si non calculable.
    """
    # Calcule la capacite max du payload (donnees brutes) dans la duree video.
    # Note: la FEC ajoute une surcouche (RS + CRC + metadonnees FEC).
    try:
        vi = ffprobe_info(in_video)
    except Exception:
        return None

    nframes = ffprobe_nb_frames(in_video)
    if nframes is None:
        # Decision: si nb_frames absent, estimer via duree.
        duration = ffprobe_duration_seconds(in_video)
        if duration is None:
            # Decision: si duree inconnue, on abandonne.
            return None
        # Formule: frames = floor(duration * fps).
        nframes = int(duration * vi.fps)

    if nframes <= 0:
        # Decision: nombre de frames invalide -> inconnue.
        return None

    # Formule: bytes/frame = floor((nb_segments_data * 2) / 8).
    bytes_per_frame = DATA_BYTES_PER_FRAME
    # Formule: bytes disponibles = nframes * bytes/frame.
    total_bytes = nframes * bytes_per_frame
    # Formule: bytes FEC dispo = total_bytes - taille headers repetes.
    fec_bytes = total_bytes - (HDR_V3_SIZE * HEADER_REPEAT_COUNT)
    if fec_bytes <= 0:
        # Decision: pas de place apres header.
        return 0
    # Formule: nb_blocs = floor(fec_bytes / RS_N).
    blocks = fec_bytes // RS_N
    # Formule: payload_max via parite renforcee.
    block_data_strong = _strong_block_data(RS_PARITY_BYTES_STRONG)
    payload_max = _max_payload_from_blocks(
        blocks, BLOCK_DATA_BYTES, block_data_strong, STRONG_PREFIX_PCT
    )
    # Decision: reserver 100 octets pour metadata fichier (nom + type).
    return max(0, payload_max - FILE_META_SIZE)

def _require_reedsolo():
    """
    But:
        Vérifier la disponibilité du module `reedsolo`.

    Fonctionnement:
        Lève une erreur explicite si la dépendance est absente.

    Paramètres d'entrée:
        Aucun.

    Sortie:
        None.
    """
    # Verifie que reedsolo est disponible.
    if not _HAS_REEDSOLO:
        # Decision: on stoppe si la lib FEC n'est pas installee.
        raise RuntimeError("Module 'reedsolo' manquant. Installe-le pour activer la FEC.")

def _rs_codec(rs_parity: int):
    """
    But:
        Créer une instance RSCodec configurée.

    Fonctionnement:
        Vérifie la dépendance puis instancie `reedsolo.RSCodec`.

    Paramètres d'entrée:
        rs_parity (int): octets de parité RS.

    Sortie:
        RSCodec: codec prêt à encoder/décoder.
    """
    # Construit un codec RS avec le nombre d'octets de parite demande.
    _require_reedsolo()
    return reedsolo.RSCodec(rs_parity)

def fec_encode_payload_uep(payload: bytes,
                           block_data_bytes: int,
                           rs_parity: int,
                           strong_rs_parity: int,
                           strong_prefix_pct: int) -> bytes:
    """
    But:
        Encoder un payload en UEP (protection renforcée au début).

    Fonctionnement:
        Découpe en blocs, ajoute CRC32 puis encode RS fort/normal.

    Paramètres d'entrée:
        payload (bytes): données à encoder.
        block_data_bytes (int): taille utile des blocs normaux.
        rs_parity (int): parité RS normale.
        strong_rs_parity (int): parité RS forte.
        strong_prefix_pct (int): pourcentage de préfixe fort.

    Sortie:
        bytes: flux FEC encodé.
    """
    # Encode le payload avec parite renforcee sur le debut (UEP).
    # Decision: verifier la coherence block_data_bytes vs RS.
    expected_k = RS_N - rs_parity
    if block_data_bytes + CRC_BYTES != expected_k:
        raise RuntimeError("Parametres FEC incoherents (block_data_bytes + CRC != RS_K).")
    # Decision: verifier la coherence block_data_bytes_strong vs RS.
    strong_block_data = _strong_block_data(strong_rs_parity)
    expected_k_strong = RS_N - strong_rs_parity
    if strong_block_data + CRC_BYTES != expected_k_strong:
        raise RuntimeError("Parametres FEC forts incoherents (block_data_bytes + CRC != RS_K).")
    if strong_rs_parity <= rs_parity:
        # Decision: la parite forte doit etre > parite normale.
        raise RuntimeError("Parite forte doit etre > parite normale.")

    rsc_norm = _rs_codec(rs_parity)
    rsc_strong = _rs_codec(strong_rs_parity)

    strong_blocks, _normal_blocks, _total = _compute_blocks_for_payload(
        len(payload), block_data_bytes, strong_block_data, strong_prefix_pct
    )

    blocks = []
    offset = 0

    # Blocs forts: couvrent les premiers strong_prefix_bytes du payload.
    for _ in range(strong_blocks):
        chunk = payload[offset:offset + strong_block_data]
        offset += len(chunk)
        if len(chunk) < strong_block_data:
            # Decision: padding a zero du dernier bloc fort.
            chunk += b"\x00" * (strong_block_data - len(chunk))
        # Formule: crc32 sur le bloc (bytes).
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        # Formule: message = data + crc32 (4 bytes LE).
        msg = chunk + struct.pack("<I", crc)
        # Formule: RS encode -> RS_N bytes.
        codeword = rsc_strong.encode(msg)
        blocks.append(codeword)

    # Blocs normaux: pour le reste du payload.
    while offset < len(payload):
        chunk = payload[offset:offset + block_data_bytes]
        offset += len(chunk)
        if len(chunk) < block_data_bytes:
            # Decision: padding a zero du dernier bloc normal.
            chunk += b"\x00" * (block_data_bytes - len(chunk))
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        msg = chunk + struct.pack("<I", crc)
        codeword = rsc_norm.encode(msg)
        blocks.append(codeword)

    return b"".join(blocks)

def fec_decode_payload(encoded: bytes,
                       payload_size: int,
                       block_data_bytes: int,
                       rs_parity: int,
                       strong_rs_parity: int,
                       strong_prefix_pct: int,
                       log_cb=None) -> bytes:
    """
    But:
        Décoder un flux FEC pour reconstruire le payload brut.

    Fonctionnement:
        Découpe en blocs RS, décode (fort/normal), vérifie CRC et tronque le
        résultat à la taille payload attendue.

    Paramètres d'entrée:
        encoded (bytes): flux FEC encodé.
        payload_size (int): taille payload cible.
        block_data_bytes (int): taille utile des blocs normaux.
        rs_parity (int): parité RS normale.
        strong_rs_parity (int): parité RS forte.
        strong_prefix_pct (int): pourcentage de préfixe fort.
        log_cb: callback de log optionnel.

    Sortie:
        bytes: payload reconstruit.
    """
    # Decode le flux FEC en payload original (trim selon payload_size).
    # Decision: verifier la coherence block_data_bytes vs RS.
    expected_k = RS_N - rs_parity
    if block_data_bytes + CRC_BYTES != expected_k:
        raise RuntimeError("Parametres FEC incoherents (block_data_bytes + CRC != RS_K).")
    strong_block_data = _strong_block_data(strong_rs_parity)
    expected_k_strong = RS_N - strong_rs_parity
    if strong_block_data + CRC_BYTES != expected_k_strong:
        raise RuntimeError("Parametres FEC forts incoherents (block_data_bytes + CRC != RS_K).")

    rsc_norm = _rs_codec(rs_parity)
    rsc_strong = _rs_codec(strong_rs_parity)

    strong_blocks, _normal_blocks, _total = _compute_blocks_for_payload(
        payload_size, block_data_bytes, strong_block_data, strong_prefix_pct
    )

    out = bytearray()
    bad_rs = 0
    bad_crc = 0

    blocks = [encoded[i:i + RS_N] for i in range(0, len(encoded), RS_N)]
    for i, b in enumerate(blocks):
        if len(b) != RS_N:
            # Decision: bloc incomplet -> ignore.
            continue
        use_strong = (i < strong_blocks)
        rsc = rsc_strong if use_strong else rsc_norm
        block_data_len = strong_block_data if use_strong else block_data_bytes
        expected_k_local = RS_N - (strong_rs_parity if use_strong else rs_parity)
        try:
            res = rsc.decode(b)
            msg = res[0] if isinstance(res, tuple) else res
        except Exception:
            # Decision: erreur RS -> bloc degrade (on garde brut).
            bad_rs += 1
            msg = b[:expected_k_local]

        data = msg[:block_data_len]
        crc_expected = struct.unpack("<I", msg[block_data_len:block_data_len + CRC_BYTES])[0]
        crc_actual = zlib.crc32(data) & 0xFFFFFFFF
        if crc_actual != crc_expected:
            # Decision: CRC faux -> bloc probablement corrompu.
            bad_crc += 1
        out.extend(data)

    if log_cb:
        # Decision: logger le resume FEC seulement si callback fourni.
        log_cb(f"FEC decode: blocs RS invalides={bad_rs}, CRC invalides={bad_crc}")

    return bytes(out[:payload_size])

def start_ffmpeg_raw_reader(path: str,
                            out_w: Optional[int] = None,
                            out_h: Optional[int] = None,
                            capture_stderr: bool = False) -> subprocess.Popen:
    """
    But:
        Démarrer ffmpeg en lecture raw BGR24.

    Fonctionnement:
        Lit la piste vidéo, applique un scale optionnel, et envoie les frames
        raw sur stdout.

    Paramètres d'entrée:
        path (str): vidéo source.
        out_w (Optional[int]): largeur de sortie optionnelle.
        out_h (Optional[int]): hauteur de sortie optionnelle.
        capture_stderr (bool): active la capture stderr si True.

    Sortie:
        subprocess.Popen: process ffmpeg lecteur.
    """
    # Demarre ffmpeg pour produire des frames raw BGR24 sur stdout.
    vf = []
    if out_w and out_h:
        # Decision: n'appliquer le scale que si les deux dimensions sont fournies.
        # scaling (bilinear) pour forcer 1920x1080 au decode
        vf = ["-vf", f"scale={out_w}:{out_h}"]

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", path,
        "-map", "0:v:0",
        *vf,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]
    stderr_target = subprocess.PIPE if capture_stderr else subprocess.DEVNULL
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_target)

def start_ffmpeg_lossless_writer(out_path: str,
                                 fps: float,
                                 audio_source: str,
                                 capture_stderr: bool = False) -> subprocess.Popen:
    """
    But:
        Démarrer ffmpeg en écriture vidéo lossless.

    Fonctionnement:
        Reçoit des frames raw BGR24 sur stdin, copie l'audio d'une source et
        encode en x264 lossless dans un conteneur MP4.

    Paramètres d'entrée:
        out_path (str): chemin de sortie.
        fps (float): cadence image.
        audio_source (str): source audio à copier.
        capture_stderr (bool): active la capture stderr si True.

    Sortie:
        subprocess.Popen: process ffmpeg writer.
    """
    # Demarre ffmpeg: lit rawvideo en stdin et copie l'audio de audio_source.
    # Encode toujours en x264 lossless (MP4).
    vcodec = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "0", "-pix_fmt", "yuv444p"]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}",
        "-r", f"{fps}",
        "-i", "-",            # video stdin
        "-i", audio_source,   # audio copy
        "-map", "0:v:0",
        "-map", "1:a?",
        *vcodec,
        "-c:a", "copy",
        out_path
    ]
    stderr_target = subprocess.PIPE if capture_stderr else subprocess.DEVNULL
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr_target)

def force_mp4_path(path: str) -> str:
    """
    But:
        Forcer une extension .mp4 sur un chemin de sortie vidéo.

    Fonctionnement:
        Si le chemin n'a pas l'extension .mp4, remplace (ou ajoute) le suffixe.

    Paramètres d'entrée:
        path (str): chemin d'entrée.

    Sortie:
        str: chemin normalisé avec extension .mp4.
    """
    root, ext = os.path.splitext(path)
    if ext.lower() == ".mp4":
        return path
    if ext:
        return root + ".mp4"
    return path + ".mp4"

# =========================
# Bitstream helpers
# =========================

class BitStream:
    """Flux de bits MSB-first depuis header + fichier. Si EOF => bits=0."""
    def __init__(self, header_bytes: bytes, file_path: Optional[str], chunk_bytes: int = 1 << 20):
        """
        But:
            Initialiser un lecteur de bits sur header + fichier optionnel.

        Fonctionnement:
            Charge d'abord les octets de header puis lit le fichier si fourni.

        Paramètres d'entrée:
            header_bytes (bytes): en-tête injecté en début de flux.
            file_path (Optional[str]): chemin de fichier binaire source.
            chunk_bytes (int): taille de lecture en octets.

        Sortie:
            None.
        """
        # Initialise l'etat: header en memoire + fichier binaire sur disque.
        self.header = memoryview(header_bytes)
        self.hoff = 0
        self.f = open(file_path, "rb") if file_path else None
        self.chunk_bytes = chunk_bytes
        self.bitbuf = np.zeros((0,), dtype=np.uint8)
        self.eof = False

    def close(self):
        """
        But:
            Fermer proprement la ressource fichier.

        Fonctionnement:
            Ferme le handle si présent, en ignorant les erreurs.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            None.
        """
        # Ferme le fichier binaire si possible.
        try:
            if self.f:
                self.f.close()
        except Exception:
            pass

    def _read_more_bytes(self) -> bytes:
        """
        But:
            Lire le prochain chunk d'octets du flux.

        Fonctionnement:
            Sert d'abord le header, puis le fichier; marque EOF si terminé.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            bytes: chunk lu (potentiellement vide).
        """
        if self.hoff < len(self.header):
            # Decision: consommer le header en priorite.
            take = min(len(self.header) - self.hoff, self.chunk_bytes)
            b = self.header[self.hoff:self.hoff + take].tobytes()
            self.hoff += take
            return b

        if self.f:
            b = self.f.read(self.chunk_bytes)
            if not b:
                # Decision: EOF si le fichier ne retourne plus d'octets.
                self.eof = True
            return b

        # Decision: pas de fichier -> EOF quand header termine.
        self.eof = True
        return b""

    def take_bits(self, nbits: int) -> np.ndarray:
        """
        But:
            Extraire un nombre fixe de bits du flux.

        Fonctionnement:
            Remplit le buffer de bits au besoin puis renvoie exactement `nbits`
            bits, avec padding à zéro si EOF.

        Paramètres d'entrée:
            nbits (int): nombre de bits demandés.

        Sortie:
            np.ndarray: tableau uint8 de bits (0/1).
        """
        # Retourne nbits bits (uint8 0/1), en remplissant de 0 a EOF.
        while self.bitbuf.size < nbits and not self.eof:
            b = self._read_more_bytes()
            if b:
                # Decision: convertir les octets lus en bits MSB-first.
                arr = np.frombuffer(b, dtype=np.uint8)
                bits = np.unpackbits(arr)  # MSB-first
                # bitbuf petit (souvent), concat ok en PoC
                self.bitbuf = np.concatenate([self.bitbuf, bits])

        if self.bitbuf.size >= nbits:
            # Decision: assez de bits -> servir exactement nbits.
            out = self.bitbuf[:nbits].copy()
            self.bitbuf = self.bitbuf[nbits:]
            return out

        # EOF => pad zeros
        if self.bitbuf.size > 0:
            # Decision: plus de bytes mais bits restants -> padding a 0.
            out = np.concatenate([self.bitbuf, np.zeros((nbits - self.bitbuf.size,), dtype=np.uint8)])
            self.bitbuf = np.zeros((0,), dtype=np.uint8)
            return out

        # Decision: aucun bit -> nbits zeros.
        return np.zeros((nbits,), dtype=np.uint8)

# =========================
# Grilles / masques (encode/decode)
# =========================

def _band_available_spans(y0: int, y1: int) -> list:
    """
    But:
        Déterminer les segments X disponibles pour une bande Y.

    Fonctionnement:
        Si la bande chevauche la vidéo centrale, on utilise les barres
        gauche/droite; sinon la largeur complète est disponible.

    Paramètres d'entrée:
        y0 (int): borne haute (incluse) de la bande.
        y1 (int): borne basse (exclue) de la bande.

    Sortie:
        list: segments `(x0, x1)` disponibles.
    """
    center_y0 = CENTER_Y
    center_y1 = CENTER_Y + CENTER_H
    overlaps_center = not (y1 <= center_y0 or y0 >= center_y1)
    if not overlaps_center:
        return [(0, W)]

    spans = []
    if CENTER_X > 0:
        spans.append((0, CENTER_X))
    right_x0 = CENTER_X + CENTER_W
    if right_x0 < W:
        spans.append((right_x0, W))
    return spans

def compute_line_band_regions() -> list:
    """
    But:
        Pré-calculer les bandes data et leurs segments RVBY.

    Fonctionnement:
        Chaque bande hors centre utilise un segment pleine largeur.
        Chaque bande recouvrant le centre utilise deux segments: gauche/droite.

    Paramètres d'entrée:
        Aucun.

    Sortie:
        list: éléments `(y0, y1, chunks)`.
    """
    total_bands = H // LINE_BAND_HEIGHT
    regions = []
    for b in range(total_bands):
        y0 = int(b * LINE_BAND_HEIGHT)
        y1 = int(min(H, y0 + LINE_BAND_HEIGHT))
        spans = _band_available_spans(y0, y1)
        if len(spans) == 0:
            continue
        # Decision: ne jamais couper une ligne pleine; seule la ligne traversant
        # le centre est separee en deux barres (gauche/droite).
        chunks = [[(int(x0), int(x1))] for (x0, x1) in spans]
        regions.append((y0, y1, chunks))
    return regions

def bits_to_2bit_symbols(bits: np.ndarray) -> np.ndarray:
    """
    But:
        Convertir un flux de bits en symboles 2 bits.

    Fonctionnement:
        Groupe les bits par 2 et calcule la valeur entière 0..3.

    Paramètres d'entrée:
        bits (np.ndarray): bits uint8 (0/1), taille multiple de 2.

    Sortie:
        np.ndarray: symboles uint8.
    """
    if bits.size == 0:
        return np.zeros((0,), dtype=np.uint8)
    grp = bits.reshape(-1, LINE_BITS_PER_SEGMENT).astype(np.uint8)
    return ((grp[:, 0] << 1) | grp[:, 1]).astype(np.uint8)

def symbols2_to_bits(symbols: np.ndarray) -> np.ndarray:
    """
    But:
        Convertir des symboles 2 bits en flux binaire.

    Fonctionnement:
        Extrait les bits MSB->LSB de chaque symbole.

    Paramètres d'entrée:
        symbols (np.ndarray): symboles uint8 (0..3).

    Sortie:
        np.ndarray: bits uint8 (0/1).
    """
    if symbols.size == 0:
        return np.zeros((0,), dtype=np.uint8)
    return np.stack([(symbols >> 1) & 1, symbols & 1], axis=1).astype(np.uint8).reshape(-1)

def classify_rvby_majority_on_segments(frame: np.ndarray, y0: int, y1: int, segments: list) -> Tuple[int, bool]:
    """
    But:
        Classer une zone en RVBY par vote majoritaire.

    Fonctionnement:
        Échantillonne uniquement la ligne centrale de la bande (hauteur 3),
        classe chaque pixel vers la couleur RVBY la plus proche, puis prend la
        classe majoritaire.

    Paramètres d'entrée:
        frame (np.ndarray): frame BGR.
        y0 (int): borne haute de la bande.
        y1 (int): borne basse de la bande.
        segments (list): segments `(x0, x1)` de la zone utile.

    Sortie:
        Tuple[int, bool]: index RVBY 0..3 et drapeau d'incertitude.
    """
    if y1 <= y0:
        return 0, True

    # Decision: on prend strictement la ligne centrale pour éviter les bords
    # de bande souvent dégradés par la compression YouTube.
    y_mid = y0 + ((y1 - y0) // 2)
    row = frame[y_mid]

    samples = []
    for x0, x1 in segments:
        if x1 > x0:
            samples.append(row[x0:x1, :])
    if not samples:
        return 0, True

    # Decision: utiliser int32 pour eviter l'overflow au carre (255^2 > int16).
    pixels = np.concatenate(samples, axis=0).astype(np.int32)
    palette = LINE_PALETTE_RVBY_BGR.astype(np.int32)
    d2 = np.sum((pixels[:, None, :] - palette[None, :, :]) ** 2, axis=2)
    classes = d2.argmin(axis=1).astype(np.int16)

    counts = np.bincount(classes, minlength=4)
    best = int(np.argmax(counts))
    total = int(counts.sum())
    if total <= 0:
        return 0, True

    # Decision: incertain si la majorité est faible.
    uncertain = counts[best] < max(1, int(0.55 * total))
    return best, bool(uncertain)

DATA_LINE_REGIONS = compute_line_band_regions()
# Formule: symboles/frame = somme des segments de toutes les bandes.
DATA_LINE_SEGMENTS = []
for _y0, _y1, _chunks in DATA_LINE_REGIONS:
    for _seg in _chunks:
        DATA_LINE_SEGMENTS.append((_y0, _y1, _seg))
# Decision: garder un multiple de 4 segments pour alignement octet.
DATA_LINE_SEGMENTS_PER_FRAME = (len(DATA_LINE_SEGMENTS) // LINE_SEGMENTS_PER_BYTE) * LINE_SEGMENTS_PER_BYTE
DATA_LINE_SEGMENTS = DATA_LINE_SEGMENTS[:DATA_LINE_SEGMENTS_PER_FRAME]
DATA_BYTES_PER_FRAME = DATA_LINE_SEGMENTS_PER_FRAME // LINE_SEGMENTS_PER_BYTE

# =========================
# Video fit (center)
# =========================

def fit_into_center(frame_bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    But:
        Redimensionner une frame en conservant le ratio dans une zone cible.

    Fonctionnement:
        Applique un scale isotrope puis place l'image dans un canvas noir.

    Paramètres d'entrée:
        frame_bgr (np.ndarray): frame source.
        out_w (int): largeur cible.
        out_h (int): hauteur cible.

    Sortie:
        np.ndarray: frame centrée en BGR.
    """
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        # Decision: frame invalide -> canvas noir.
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Formule: facteur = min(scale_x, scale_y) pour conserver l'aspect.
    scale = min(out_w / w, out_h / h)
    # Formules: nouvelles dimensions = dimensions * scale.
    nw, nh = int(round(w * scale)), int(round(h * scale))
    # Decision: borne pour eviter 0 ou depassement.
    nw = max(1, min(out_w, nw))
    nh = max(1, min(out_h, nh))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    # Formules: offsets pour centrer dans le canvas.
    x = (out_w - nw) // 2
    y = (out_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas

def load_center_image(image_path: str, out_w: int, out_h: int) -> np.ndarray:
    """
    But:
        Charger une image fixe et l'ajuster au format centre.

    Fonctionnement:
        Lit l'image disque en BGR puis applique `fit_into_center`.

    Paramètres d'entrée:
        image_path (str): chemin image.
        out_w (int): largeur centre.
        out_h (int): hauteur centre.

    Sortie:
        np.ndarray: image centrée BGR.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Image fixe invalide/inlisible: {image_path}")
    return fit_into_center(img, out_w, out_h)

# =========================
# Encode
# =========================

def encode_video(in_video: str,
                 in_data: str,
                 out_video: str,
                 in_image: Optional[str] = None,
                 log_cb=None,
                 progress_cb=None,
                 stop_flag=None):
    """
    But:
        Encoder un fichier binaire dans une vidéo porteuse.

    Fonctionnement:
        Lit la vidéo source, encode le payload via FEC, mappe les octets sur
        des bandes RVBY (2 bits/segment), superpose le centre (vidéo/image) puis
        écrit la vidéo lossless.

    Paramètres d'entrée:
        in_video (str): vidéo source.
        in_data (str): fichier binaire à embarquer.
        out_video (str): vidéo de sortie.
        in_image (Optional[str]): image fixe optionnelle (intro/outro/extension).
        log_cb: callback de log optionnel.
        progress_cb: callback de progression optionnel.
        stop_flag: callback booléen d'arrêt optionnel.

    Sortie:
        None.
    """
    # Encode un fichier binaire dans une video: bandes horizontales + centre video.
    normalized_out_video = force_mp4_path(out_video)
    if normalized_out_video != out_video and log_cb:
        # Decision: notifier quand le suffixe de sortie est force en MP4.
        log_cb(f"Sortie forcée en MP4: {normalized_out_video}")

    vi = ffprobe_info(in_video)
    fps = vi.fps
    image_path = in_image.strip() if isinstance(in_image, str) else ""
    image_enabled = bool(image_path)
    intro_frames = int(round(IMAGE_PAD_SECONDS * fps)) if image_enabled else 0
    outro_frames = int(round(IMAGE_PAD_SECONDS * fps)) if image_enabled else 0

    input_file_size = os.path.getsize(in_data)
    file_meta = _build_file_meta_block(in_data)
    # Lecture complete du payload (necessaire pour FEC).
    with open(in_data, "rb") as f:
        payload = file_meta + f.read()
    payload_size = len(payload)

    # FEC: RS + CRC avec parite renforcee au debut.
    fec_payload = fec_encode_payload_uep(
        payload,
        block_data_bytes=BLOCK_DATA_BYTES,
        rs_parity=RS_PARITY_BYTES,
        strong_rs_parity=RS_PARITY_BYTES_STRONG,
        strong_prefix_pct=STRONG_PREFIX_PCT
    )

    # Formule: header V3 = MAGIC + taille payload + params FEC + parite forte.
    header = struct.pack(
        HDR_V3_FMT,
        MAGIC_V3,
        payload_size,
        BLOCK_DATA_BYTES,
        RS_PARITY_BYTES,
        HEADER_RESERVED_VALUE,
        RS_PARITY_BYTES_STRONG,
        STRONG_PREFIX_PCT
    )
    # BitStream lit d'abord le header répété, puis les octets FEC en memoire.
    bs = BitStream((header * HEADER_REPEAT_COUNT) + fec_payload, None)

    # Formule: bytes/frame = floor((nb_segments_data * 2) / 8).
    bytes_per_frame = DATA_BYTES_PER_FRAME
    if bytes_per_frame <= 0:
        # Decision: aucune capacite data disponible.
        raise RuntimeError("Aucune ligne disponible pour encoder les donnees.")

    # reader: output raw frames at original size
    reader = start_ffmpeg_raw_reader(in_video, capture_stderr=bool(log_cb))
    if reader.stdout is None:
        # Decision: si stdout absent, on ne peut pas lire les frames.
        raise RuntimeError("ffmpeg reader has no stdout")
    reader_log_pump = _start_process_stderr_pump(reader, "ffmpeg/read", log_cb)

    # writer: lossless video + copy audio
    writer = start_ffmpeg_lossless_writer(
        normalized_out_video,
        fps=fps,
        audio_source=in_video,
        capture_stderr=bool(log_cb)
    )
    if writer.stdin is None:
        # Decision: si stdin absent, on ne peut pas ecrire les frames.
        raise RuntimeError("ffmpeg writer has no stdin")
    writer_log_pump = _start_process_stderr_pump(writer, "ffmpeg/write", log_cb)

    # Formule: bytes/frame = largeur * hauteur * 3 (BGR24).
    frame_bytes = vi.width * vi.height * 3
    frame_idx = 0
    data_bytes_consumed = 0
    source_ended = False
    source_end_frame_idx = None

    if image_enabled:
        center_hold = load_center_image(image_path, CENTER_W, CENTER_H)
        if log_cb:
            log_cb(
                f"Image fixe activée: intro={intro_frames} frames (~{IMAGE_PAD_SECONDS:.1f}s), "
                f"outro min={outro_frames} frames (~{IMAGE_PAD_SECONDS:.1f}s)."
            )
    else:
        center_hold = np.zeros((CENTER_H, CENTER_W, 3), dtype=np.uint8)

    try:
        while True:
            if stop_flag and stop_flag():
                # Decision: arret demande -> sortie propre.
                if log_cb:
                    # Decision: logger seulement si callback fourni.
                    log_cb("Stop demandé. Arrêt propre…")
                break

            if frame_idx < intro_frames:
                # Decision: intro fixe avant démarrage des frames source.
                center = center_hold
            elif not source_ended:
                raw = reader.stdout.read(frame_bytes)
                have_video = (raw is not None and len(raw) == frame_bytes)
                if have_video:
                    # Decision: on utilise la frame video si disponible.
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((vi.height, vi.width, 3))
                    center = fit_into_center(frame, CENTER_W, CENTER_H)
                else:
                    # Decision: fin de video source; on bascule sur l'image fixe (ou noir si absente).
                    source_ended = True
                    source_end_frame_idx = frame_idx
                    center = center_hold
            else:
                # Decision: extension apres EOF source avec image fixe (ou noir si absente).
                center = center_hold

            # Data lines: bytes -> symboles RVBY (2 bits/segment).
            bits = bs.take_bits(bytes_per_frame * 8)
            data_bytes_consumed += (bits.size // 8)
            seg_values = bits_to_2bit_symbols(bits)

            # Construit la frame porteuse en barres RVBY autour du centre.
            out_bgr = np.zeros((H, W, 3), dtype=np.uint8)
            for i, (y0, y1, seg_spans) in enumerate(DATA_LINE_SEGMENTS):
                cidx = int(seg_values[i])
                color = LINE_PALETTE_RVBY_BGR[cidx]
                for x0, x1 in seg_spans:
                    out_bgr[y0:y1, x0:x1, :] = color

            # Inserer centre vidéo
            out_bgr[CENTER_Y:CENTER_Y + CENTER_H, CENTER_X:CENTER_X + CENTER_W] = center

            writer.stdin.write(out_bgr.tobytes())
            frame_idx += 1

            data_done = bool(bs.eof and bs.bitbuf.size == 0)
            if source_ended:
                if source_end_frame_idx is None:
                    source_end_frame_idx = frame_idx
                outro_done = (frame_idx - source_end_frame_idx) >= outro_frames
                if data_done and outro_done:
                    # Decision: stop quand data complete + outro minimal atteint.
                    break

            if progress_cb and frame_idx % 5 == 0:
                # Decision: rafraichir la progression toutes les 5 frames.
                progress_cb(frame_idx, data_bytes_consumed, payload_size)

    finally:
        bs.close()
        try:
            if reader.stdout:
                # Decision: fermer stdout si ouvert.
                reader.stdout.close()
        except Exception:
            pass
        reader.wait()
        try:
            if reader.stderr:
                # Decision: fermer stderr si ouvert.
                reader.stderr.close()
        except Exception:
            pass
        _join_process_stderr_pump(reader_log_pump)

        try:
            if writer.stdin:
                # Decision: fermer stdin si ouvert.
                writer.stdin.close()
        except Exception:
            pass
        writer.wait()
        try:
            if writer.stderr:
                # Decision: fermer stderr si ouvert.
                writer.stderr.close()
        except Exception:
            pass
        _join_process_stderr_pump(writer_log_pump)

    if log_cb:
        # Decision: logger le resume seulement si callback fourni.
        log_cb(
            f"Encodage terminé. Frames: {frame_idx}. Payload: {payload_size} bytes "
            f"(data={input_file_size} + meta={FILE_META_SIZE}). Fichier: {normalized_out_video}"
        )

# =========================
# Decode
# =========================

def decode_video(in_video: str,
                 out_dir: str,
                 force_scale_1080p: bool = True,
                 strict_magic: bool = True,
                 log_cb=None,
                 progress_cb=None,
                 stop_flag=None):
    """
    But:
        Décoder une vidéo encodée par le programme vers un fichier binaire.

    Fonctionnement:
        Lit les frames raw, décode les bandes de données, reconstitue les
        bytes, parse le header, applique le décodage FEC puis écrit le payload.

    Paramètres d'entrée:
        in_video (str): vidéo à décoder.
        out_dir (str): répertoire de sortie.
        force_scale_1080p (bool): force le redimensionnement 1080p au decode.
        strict_magic (bool): exige un magic valide.
        log_cb: callback de log optionnel.
        progress_cb: callback de progression optionnel.
        stop_flag: callback booléen d'arrêt optionnel.

    Sortie:
        None.
    """
    # Decode une video a bandes horizontales et reconstruit le fichier binaire.
    if not out_dir:
        # Decision: un repertoire de sortie est obligatoire.
        raise RuntimeError("Répertoire de sortie invalide.")
    # decode via ffmpeg -> rawvideo bgr24
    if force_scale_1080p:
        # Decision: forcer le scale 1920x1080 pour la grille fixe.
        reader = start_ffmpeg_raw_reader(in_video, out_w=W, out_h=H, capture_stderr=bool(log_cb))
        frame_w, frame_h = W, H
    else:
        # Decision: garder la taille d'origine (puis rescale si besoin).
        vi = ffprobe_info(in_video)
        reader = start_ffmpeg_raw_reader(in_video, capture_stderr=bool(log_cb))
        frame_w, frame_h = vi.width, vi.height

    if reader.stdout is None:
        # Decision: si stdout absent, impossible de lire les frames.
        raise RuntimeError("ffmpeg reader has no stdout")
    reader_log_pump = _start_process_stderr_pump(reader, "ffmpeg/read", log_cb)

    # Formule: bytes/frame = largeur * hauteur * 3 (BGR24).
    frame_bytes = frame_w * frame_h * 3

    # Formule: bytes/frame = floor((nb_segments_data * 2) / 8).
    bytes_per_frame = DATA_BYTES_PER_FRAME
    if bytes_per_frame <= 0:
        # Decision: aucune capacite data disponible.
        raise RuntimeError("Aucune ligne disponible pour decoder les donnees.")
    # Formule: symboles/frame = nb_segments_data (2 bits chacun).
    data_segments_per_frame = DATA_LINE_SEGMENTS_PER_FRAME

    out_bytes = bytearray()
    uncertain_line_symbols = 0

    payload_size = None
    header_ok = False
    header_version = None
    header_size = None
    header_data_offset = None
    fec_block_data_bytes = None
    fec_rs_parity = None
    header_reserved_field = None
    fec_strong_rs_parity = None
    fec_strong_prefix_pct = None
    encoded_needed = None
    last_majority_scan_len = -1

    def detect_header_repetitions(hsize: int) -> Optional[int]:
        """
        But:
            Détecter le nombre de headers consécutifs au début du flux.

        Fonctionnement:
            Compare des blocs de taille `hsize` égaux au premier header.
            Retourne `None` tant que l'information n'est pas stabilisée.

        Paramètres d'entrée:
            hsize (int): taille d'un header.

        Sortie:
            Optional[int]: nombre de répétitions validées.
        """
        if len(out_bytes) < hsize:
            return None
        first = bytes(out_bytes[:hsize])
        copies = 1
        while (copies < HEADER_REPEAT_COUNT and
               len(out_bytes) >= (copies + 1) * hsize and
               bytes(out_bytes[copies * hsize:(copies + 1) * hsize]) == first):
            copies += 1

        # Decision: tant qu'on ne sait pas si la repetition continue, on attend.
        if copies < HEADER_REPEAT_COUNT and len(out_bytes) < (copies + 1) * hsize:
            return None
        return copies

    def try_parse_header_by_majority() -> bool:
        """
        But:
            Récupérer un header corrompu via vote majoritaire sur les répétitions.

        Fonctionnement:
            Balaye un préfixe du flux, reconstruit un header "moyen" sur plusieurs
            copies, puis valide ses champs (V1/V2/V3).

        Paramètres d'entrée:
            Aucun (utilise les variables nonlocal).

        Sortie:
            bool: True si un header valide a été reconstruit.
        """
        nonlocal payload_size, header_ok
        nonlocal header_version, header_size, header_data_offset
        nonlocal fec_block_data_bytes, fec_rs_parity, header_reserved_field, encoded_needed
        nonlocal fec_strong_rs_parity, fec_strong_prefix_pct
        nonlocal last_majority_scan_len

        min_copies = min(HEADER_REPEAT_COUNT, 3)
        if len(out_bytes) < (HDR_V1_SIZE * min_copies):
            return False

        # Decision: eviter de rescanner a chaque frame tant que peu de bytes nouveaux.
        if last_majority_scan_len >= 0 and (len(out_bytes) - last_majority_scan_len) < 1024:
            return False
        last_majority_scan_len = len(out_bytes)

        scan_prefix = 4096
        min_confidence = 0.55

        def _magic_distance(a: bytes, b: bytes) -> int:
            return int(sum(1 for x, y in zip(a, b) if x != y))

        def _majority_header(offset: int, hsize: int, copies: int) -> Tuple[bytes, float]:
            block = np.frombuffer(out_bytes[offset:offset + (copies * hsize)], dtype=np.uint8).reshape((copies, hsize))
            voted = np.empty((hsize,), dtype=np.uint8)
            agree = 0
            for j in range(hsize):
                counts = np.bincount(block[:, j], minlength=256)
                winner = int(counts.argmax())
                voted[j] = np.uint8(winner)
                agree += int(counts[winner])
            confidence = float(agree) / float(copies * hsize)
            return voted.tobytes(), confidence

        specs = (
            (3, MAGIC_V3, HDR_V3_SIZE, HDR_V3_FMT),
            (2, MAGIC_V2, HDR_V2_SIZE, HDR_V2_FMT),
            (1, MAGIC_V1, HDR_V1_SIZE, HDR_V1_FMT),
        )

        for ver, mg, hsize, hfmt in specs:
            need = hsize * min_copies
            if len(out_bytes) < need:
                continue
            max_off = min(scan_prefix, len(out_bytes) - need)
            for off in range(max_off + 1):
                copies = min(HEADER_REPEAT_COUNT, (len(out_bytes) - off) // hsize)
                if copies < min_copies:
                    continue

                voted_header, conf = _majority_header(off, hsize, copies)
                mdist = _magic_distance(voted_header[:8], mg)
                if mdist > 2 or conf < min_confidence:
                    continue

                try:
                    fields = struct.unpack(hfmt, voted_header)
                except Exception:
                    continue

                if ver == 3:
                    _, sz, bdb, rsp, reserved, srsp, spct = fields
                    v_payload = int(sz)
                    v_bdb = int(bdb)
                    v_rsp = int(rsp)
                    v_reserved = int(reserved)
                    v_srsp = int(srsp)
                    v_spct = int(spct)
                    if (v_payload < 0 or
                        v_bdb <= 0 or v_bdb > RS_N or
                        v_rsp <= 0 or v_rsp >= RS_N or
                        v_srsp <= 0 or v_srsp >= RS_N or
                        v_spct < 0 or v_spct > 100):
                        continue
                    strong_block_data = _strong_block_data(v_srsp)
                    if strong_block_data <= 0:
                        continue
                    try:
                        _, _, total_blocks = _compute_blocks_for_payload(
                            v_payload, v_bdb, strong_block_data, v_spct
                        )
                    except Exception:
                        continue
                    payload_size = v_payload
                    header_version = 3
                    header_size = HDR_V3_SIZE
                    header_data_offset = off + (copies * HDR_V3_SIZE)
                    fec_block_data_bytes = v_bdb
                    fec_rs_parity = v_rsp
                    header_reserved_field = v_reserved
                    fec_strong_rs_parity = v_srsp
                    fec_strong_prefix_pct = v_spct
                    encoded_needed = total_blocks * RS_N
                    header_ok = True
                elif ver == 2:
                    _, sz, bdb, rsp, reserved = fields
                    v_payload = int(sz)
                    v_bdb = int(bdb)
                    v_rsp = int(rsp)
                    v_reserved = int(reserved)
                    if (v_payload < 0 or
                        v_bdb <= 0 or v_bdb > RS_N or
                        v_rsp <= 0 or v_rsp >= RS_N):
                        continue
                    try:
                        blocks = _ceil_div(v_payload, v_bdb)
                    except Exception:
                        continue
                    payload_size = v_payload
                    header_version = 2
                    header_size = HDR_V2_SIZE
                    header_data_offset = off + (copies * HDR_V2_SIZE)
                    fec_block_data_bytes = v_bdb
                    fec_rs_parity = v_rsp
                    header_reserved_field = v_reserved
                    encoded_needed = blocks * RS_N
                    header_ok = True
                else:
                    _, sz = fields
                    v_payload = int(sz)
                    if v_payload < 0:
                        continue
                    payload_size = v_payload
                    header_version = 1
                    header_size = HDR_V1_SIZE
                    header_data_offset = off + (copies * HDR_V1_SIZE)
                    header_ok = True

                if log_cb:
                    log_cb(
                        f"Header V{ver} récupéré par vote majoritaire "
                        f"(offset={off}, copies={copies}, confiance={conf:.2f}, magic_dist={mdist})."
                    )
                return True

        return False

    def parse_header_if_possible():
        """
        But:
            Tenter de parser le header quand suffisamment d'octets sont lus.

        Fonctionnement:
            Détecte la version de header (V1/V2/V3), extrait les paramètres FEC
            et initialise les conditions d'arrêt du décodage.

        Paramètres d'entrée:
            Aucun (utilise les variables nonlocal).

        Sortie:
            None.
        """
        # Tente de parser le header des que possible (V1 ou V2/V3), avec
        # resynchronisation sur MAGIC en cas de corruption en tete.
        nonlocal payload_size, header_ok
        nonlocal header_version, header_size, header_data_offset
        nonlocal fec_block_data_bytes, fec_rs_parity, header_reserved_field, encoded_needed
        nonlocal fec_strong_rs_parity, fec_strong_prefix_pct

        if header_ok:
            # Decision: deja parse, on ne refait pas.
            return

        # On attend au moins 8 bytes pour identifier la version.
        if len(out_bytes) < 8:
            # Decision: pas assez d'octets pour lire le magic.
            return

        known_magics = (MAGIC_V3, MAGIC_V2, MAGIC_V1)
        # Decision: scanner un prefixe raisonnable pour retrouver un magic.
        magic_scan_limit = 65536

        while True:
            if len(out_bytes) < 8:
                return
            magic = bytes(out_bytes[:8])
            if magic in known_magics:
                break

            search_len = min(len(out_bytes), magic_scan_limit)
            best_idx = None
            for mg in known_magics:
                idx = out_bytes.find(mg, 1, search_len)
                if idx != -1 and (best_idx is None or idx < best_idx):
                    best_idx = idx

            if best_idx is not None:
                if log_cb:
                    log_cb(f"Resync MAGIC: décalage de {best_idx} octets.")
                del out_bytes[:best_idx]
                continue

            if try_parse_header_by_majority():
                return

            if strict_magic:
                if len(out_bytes) >= magic_scan_limit:
                    if log_cb:
                        log_cb("MAGIC introuvable en mode strict: bascule automatique en mode permissif.")
                    payload_size = None
                    header_ok = True
                    return
                # Decision: attendre plus d'octets avant d'échouer.
                return
            # Mode permissif: on ne bloque pas.
            if log_cb:
                log_cb("MAGIC invalide, mode permissif: dump brut.")
            payload_size = None
            header_ok = True
            return

        if magic == MAGIC_V3:
            # Decision: header V3 (parite renforcee).
            if len(out_bytes) < HDR_V3_SIZE:
                # Decision: pas assez d'octets pour parser le header V3.
                return
            m, sz, bdb, rsp, reserved, srsp, spct = struct.unpack(HDR_V3_FMT, out_bytes[:HDR_V3_SIZE])
            payload_size = int(sz)
            header_version = 3
            header_size = HDR_V3_SIZE
            fec_block_data_bytes = int(bdb)
            fec_rs_parity = int(rsp)
            header_reserved_field = int(reserved)
            fec_strong_rs_parity = int(srsp)
            fec_strong_prefix_pct = int(spct)

            # Validation champs header avant calcul.
            if (payload_size < 0 or
                fec_block_data_bytes <= 0 or fec_block_data_bytes > RS_N or
                fec_rs_parity <= 0 or fec_rs_parity >= RS_N or
                fec_strong_rs_parity <= 0 or fec_strong_rs_parity >= RS_N or
                fec_strong_prefix_pct < 0 or fec_strong_prefix_pct > 100):
                if log_cb:
                    log_cb("Header V3 invalide (champs incohérents), tentative de resynchronisation.")
                del out_bytes[:1]
                return

            strong_block_data = _strong_block_data(fec_strong_rs_parity)
            if strong_block_data <= 0:
                if log_cb:
                    log_cb("Header V3 invalide (strong_block_data <= 0), tentative de resynchronisation.")
                del out_bytes[:1]
                return
            try:
                strong_blocks, normal_blocks, total_blocks = _compute_blocks_for_payload(
                    payload_size, fec_block_data_bytes, strong_block_data, fec_strong_prefix_pct
                )
            except Exception:
                if log_cb:
                    log_cb("Header V3 invalide (calcul blocs FEC), tentative de resynchronisation.")
                del out_bytes[:1]
                return
            header_copies = detect_header_repetitions(HDR_V3_SIZE)
            if header_copies is None:
                # Decision: attendre pour stabiliser la detection des repetitions.
                return
            header_data_offset = header_copies * HDR_V3_SIZE
            # Formule: bytes FEC requis = nb_blocs * RS_N.
            encoded_needed = total_blocks * RS_N
            header_ok = True
            if log_cb:
                # Decision: logger les parametres FEC forts.
                log_cb(
                    "Header V3 OK. Payload=%d | block=%d | rs_parity=%d | "
                    "strong_parity=%d | strong_pct=%d"
                    % (payload_size, fec_block_data_bytes, fec_rs_parity,
                       fec_strong_rs_parity, fec_strong_prefix_pct)
                )
                if header_copies > 1:
                    log_cb(f"Header V3 répété détecté: x{header_copies}")
        elif magic == MAGIC_V2:
            # Decision: header V2 (avec FEC).
            if len(out_bytes) < HDR_V2_SIZE:
                # Decision: pas assez d'octets pour parser le header V2.
                return
            m, sz, bdb, rsp, reserved = struct.unpack(HDR_V2_FMT, out_bytes[:HDR_V2_SIZE])
            payload_size = int(sz)
            header_version = 2
            header_size = HDR_V2_SIZE
            fec_block_data_bytes = int(bdb)
            fec_rs_parity = int(rsp)
            header_reserved_field = int(reserved)
            if (payload_size < 0 or
                fec_block_data_bytes <= 0 or fec_block_data_bytes > RS_N or
                fec_rs_parity <= 0 or fec_rs_parity >= RS_N):
                if log_cb:
                    log_cb("Header V2 invalide (champs incohérents), tentative de resynchronisation.")
                del out_bytes[:1]
                return
            # Formule: nb_blocs = ceil(payload_size / block_data_bytes).
            try:
                blocks = _ceil_div(payload_size, fec_block_data_bytes)
            except Exception:
                if log_cb:
                    log_cb("Header V2 invalide (calcul blocs FEC), tentative de resynchronisation.")
                del out_bytes[:1]
                return
            header_copies = detect_header_repetitions(HDR_V2_SIZE)
            if header_copies is None:
                return
            header_data_offset = header_copies * HDR_V2_SIZE
            # Formule: bytes FEC requis = nb_blocs * RS_N.
            encoded_needed = blocks * RS_N
            header_ok = True
            if log_cb:
                # Decision: logger les parametres FEC.
                log_cb(
                    "Header V2 OK. Payload=%d bytes | block=%d | rs_parity=%d"
                    % (payload_size, fec_block_data_bytes, fec_rs_parity)
                )
                if header_copies > 1:
                    log_cb(f"Header V2 répété détecté: x{header_copies}")
        elif magic == MAGIC_V1:
            # Decision: header V1 (sans FEC).
            if len(out_bytes) < HDR_V1_SIZE:
                # Decision: pas assez d'octets pour parser le header V1.
                return
            m, sz = struct.unpack(HDR_V1_FMT, out_bytes[:HDR_V1_SIZE])
            payload_size = int(sz)
            if payload_size < 0:
                if log_cb:
                    log_cb("Header V1 invalide (payload négatif), tentative de resynchronisation.")
                del out_bytes[:1]
                return
            header_copies = detect_header_repetitions(HDR_V1_SIZE)
            if header_copies is None:
                return
            header_data_offset = header_copies * HDR_V1_SIZE
            header_version = 1
            header_size = HDR_V1_SIZE
            header_ok = True
            if log_cb:
                # Decision: logger le header si callback fourni.
                log_cb(f"Header V1 OK. Payload attendu: {payload_size} bytes.")
                if header_copies > 1:
                    log_cb(f"Header V1 répété détecté: x{header_copies}")

    frame_idx = 0
    try:
        while True:
            if stop_flag and stop_flag():
                # Decision: arret demande -> sortie propre.
                if log_cb:
                    # Decision: logger seulement si callback fourni.
                    log_cb("Stop demandé. Arrêt propre…")
                break

            raw = reader.stdout.read(frame_bytes)
            if raw is None or len(raw) != frame_bytes:
                # Decision: plus de frames completes -> stop.
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((frame_h, frame_w, 3))

            # Si pas scale force, on ramene au 1080p pour garder la grille fixe
            if frame_w != W or frame_h != H:
                # Decision: resize pour retrouver la grille 1920x1080.
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

            # Mesure chaque segment RVBY (2 bits chacun).
            seg_values = np.empty((data_segments_per_frame,), dtype=np.uint8)
            for i, (y0, y1, seg_spans) in enumerate(DATA_LINE_SEGMENTS):
                cidx, uncertain = classify_rvby_majority_on_segments(frame, y0, y1, seg_spans)
                seg_values[i] = np.uint8(cidx)
                if uncertain:
                    uncertain_line_symbols += 1

            bits_new = symbols2_to_bits(seg_values)
            bytes_new = np.packbits(bits_new).astype(np.uint8)
            out_bytes.extend(bytes_new.tobytes())

            # Parse header si possible
            parse_header_if_possible()

            # Stop condition selon version
            if header_ok and payload_size is not None:
                if header_version == 1:
                    # Formule: bytes utiles = len(out_bytes) - taille header.
                    have = len(out_bytes) - header_data_offset
                    if have >= payload_size:
                        # Decision: on a recupere tout le payload -> stop.
                        break
                elif header_version in (2, 3) and encoded_needed is not None:
                    # Formule: bytes FEC = len(out_bytes) - taille header.
                    have = len(out_bytes) - header_data_offset
                    if have >= encoded_needed:
                        # Decision: on a recupere tout le flux FEC -> stop.
                        break

            frame_idx += 1
            if progress_cb and frame_idx % 5 == 0:
                # Decision: rafraichir la progression toutes les 5 frames.
                # Decision: si payload_size inconnu, on passe -1.
                progress_cb(frame_idx, len(out_bytes), payload_size if payload_size else -1)

    finally:
        try:
            if reader.stdout:
                # Decision: fermer stdout si ouvert.
                reader.stdout.close()
        except Exception:
            pass
        reader.wait()
        try:
            if reader.stderr:
                # Decision: fermer stderr si ouvert.
                reader.stderr.close()
        except Exception:
            pass
        _join_process_stderr_pump(reader_log_pump)

    if uncertain_line_symbols > 0 and log_cb:
        # Decision: signaler les segments couleur peu confiants.
        log_cb(f"Décodage lignes: segments RVBY incertains = {uncertain_line_symbols}")

    # Write output
    if header_ok and payload_size is not None and header_size is not None and header_data_offset is not None:
        if header_version == 1:
            # Decision: format V1 -> ecriture payload direct.
            if len(out_bytes) >= header_data_offset:
                payload = out_bytes[header_data_offset:header_data_offset + payload_size]
                out_path = _write_decoded_payload(out_dir, payload, log_cb=log_cb)
                if log_cb:
                    # Decision: logger le resume seulement si callback fourni.
                    log_cb(f"Décodage terminé. Frames lues: {frame_idx}. Écrit: {len(payload)} bytes. Fichier: {out_path}")
            else:
                if log_cb:
                    # Decision: logger l'incomplet si callback fourni.
                    log_cb("Décodage V1 incomplet: header present mais payload manquant.")
        elif header_version == 2:
            # Decision: format V2 -> decode FEC (sans parite forte).
            if encoded_needed is None:
                # Decision: si taille FEC inconnue, on force a zero.
                encoded_needed = 0
            encoded = out_bytes[header_data_offset:header_data_offset + encoded_needed]
            payload = fec_decode_payload(
                encoded,
                payload_size=payload_size,
                block_data_bytes=fec_block_data_bytes,
                rs_parity=fec_rs_parity,
                strong_rs_parity=fec_rs_parity,
                strong_prefix_pct=0,
                log_cb=log_cb
            )
            out_path = _write_decoded_payload(out_dir, payload, log_cb=log_cb)
            if log_cb:
                # Decision: logger le resume seulement si callback fourni.
                log_cb(f"Décodage terminé (V2 FEC). Frames lues: {frame_idx}. Écrit: {len(payload)} bytes. Fichier: {out_path}")
        elif header_version == 3:
            # Decision: format V3 -> decode FEC avec parite forte.
            if encoded_needed is None:
                # Decision: si taille FEC inconnue, on force a zero.
                encoded_needed = 0
            encoded = out_bytes[header_data_offset:header_data_offset + encoded_needed]
            payload = fec_decode_payload(
                encoded,
                payload_size=payload_size,
                block_data_bytes=fec_block_data_bytes,
                rs_parity=fec_rs_parity,
                strong_rs_parity=fec_strong_rs_parity,
                strong_prefix_pct=fec_strong_prefix_pct,
                log_cb=log_cb
            )
            out_path = _write_decoded_payload(out_dir, payload, log_cb=log_cb)
            if log_cb:
                # Decision: logger le resume seulement si callback fourni.
                log_cb(f"Décodage terminé (V3 FEC+strong). Frames lues: {frame_idx}. Écrit: {len(payload)} bytes. Fichier: {out_path}")
        else:
            # Decision: version inconnue -> dump brut.
            out_path = _write_decoded_payload(out_dir, bytes(out_bytes), log_cb=log_cb)
            if log_cb:
                # Decision: logger le resume seulement si callback fourni.
                log_cb(f"Décodage (brut) terminé. Frames lues: {frame_idx}. Écrit: {len(out_bytes)} bytes. Fichier: {out_path}")
    else:
        # Mode permissif / brut si header absent.
        out_path = _write_decoded_payload(out_dir, bytes(out_bytes), log_cb=log_cb)
        if log_cb:
            # Decision: logger le resume seulement si callback fourni.
            log_cb(f"Décodage (brut) terminé. Frames lues: {frame_idx}. Écrit: {len(out_bytes)} bytes. Fichier: {out_path}")

# =========================
# GUI Tkinter
# =========================

def run_gui():
    """
    But:
        Lancer l'interface graphique d'encodage/décodage.

    Fonctionnement:
        Construit les onglets Tkinter, démarre les workers en thread, et met
        à jour logs/progression via des queues.

    Paramètres d'entrée:
        Aucun.

    Sortie:
        None.
    """
    # Lance l'interface graphique Tkinter.
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    qlog = queue.Queue()
    qprog = queue.Queue()
    stop_event = threading.Event()
    settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "YouTubeNas.gui_settings.json")
    settings_save_job = None
    settings_loading = False

    def log(msg: str):
        """
        But:
            Envoyer un message de log vers la queue UI.

        Fonctionnement:
            Enfile le message dans `qlog`.

        Paramètres d'entrée:
            msg (str): message à afficher.

        Sortie:
            None.
        """
        # Envoie un message de log au thread UI.
        qlog.put(msg)

    def progress(*args):
        """
        But:
            Envoyer un événement de progression vers la queue UI.

        Fonctionnement:
            Enfile le tuple d'arguments dans `qprog`.

        Paramètres d'entrée:
            *args: données de progression.

        Sortie:
            None.
        """
        # Envoie un evenement de progression au thread UI.
        qprog.put(args)

    def stop_flag():
        """
        But:
            Exposer l'état d'arrêt demandé par l'utilisateur.

        Fonctionnement:
            Retourne l'état de l'event `stop_event`.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            bool: True si arrêt demandé.
        """
        # Indique si l'utilisateur a demande l'arret.
        return stop_event.is_set()

    def browse_file(var, save=False, types=None):
        """
        But:
            Ouvrir un sélecteur de fichier et renseigner une variable Tk.

        Fonctionnement:
            Utilise une boîte open/save selon `save`, puis affecte `var`.

        Paramètres d'entrée:
            var: variable Tk de destination.
            save (bool): mode sauvegarde si True.
            types: filtres de fichiers.

        Sortie:
            None.
        """
        # Ouvre un dialog de fichier (open/save) et renseigne la variable Tk.
        if save:
            # Decision: mode "save as".
            path = filedialog.asksaveasfilename(filetypes=types)
        else:
            # Decision: mode "open".
            path = filedialog.askopenfilename(filetypes=types)
        if path:
            # Decision: on ne set la variable que si un chemin est choisi.
            var.set(path)

    def browse_dir(var):
        """
        But:
            Ouvrir un sélecteur de répertoire et renseigner une variable Tk.

        Fonctionnement:
            Utilise une boîte de dialogue dossier puis affecte `var`.

        Paramètres d'entrée:
            var: variable Tk de destination.

        Sortie:
            None.
        """
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def start_worker(target, *args, **kwargs):
        """
        But:
            Lancer une tâche longue sans bloquer l'UI.

        Fonctionnement:
            Démarre un thread daemon exécutant `target`, avec capture d'erreur.

        Paramètres d'entrée:
            target: fonction à exécuter.
            *args: arguments positionnels.
            **kwargs: arguments nommés.

        Sortie:
            None.
        """
        # Lance une tache lourde dans un thread pour ne pas bloquer l'UI.
        stop_event.clear()
        def _w():
            """
            But:
                Exécuter la cible du worker avec gestion d'exception.

            Fonctionnement:
                Appelle la fonction cible et envoie l'erreur dans les logs
                en cas d'échec.

            Paramètres d'entrée:
                Aucun (capture la fermeture).

            Sortie:
                None.
            """
            try:
                target(*args, **kwargs)
            except Exception as e:
                # Decision: capture des erreurs pour les afficher dans le log.
                qlog.put(f"ERREUR: {e}")
        t = threading.Thread(target=_w, daemon=True)
        t.start()

    root = tk.Tk()
    root.title("YouTubeNAS (encode / decode)")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)

    # -------- Encode tab
    tab_enc = ttk.Frame(nb)
    nb.add(tab_enc, text="Encodage")

    in_video_var = tk.StringVar()
    in_data_var = tk.StringVar()
    in_image_var = tk.StringVar()
    out_video_var = tk.StringVar()
    max_payload_var = tk.StringVar(value="Capacité max données: —")

    row = 0
    ttk.Label(tab_enc, text="Vidéo source").grid(row=row, column=0, sticky="w")
    ttk.Entry(tab_enc, textvariable=in_video_var, width=70).grid(row=row, column=1, padx=6)
    ttk.Button(tab_enc, text="Choisir…", command=lambda: browse_file(in_video_var, save=False,
                                                                     types=[("Video", "*.mp4 *.mkv *.mov *.webm *.avi"), ("All", "*.*")])
               ).grid(row=row, column=2)
    row += 1
    ttk.Label(tab_enc, textvariable=max_payload_var).grid(row=row, column=0, columnspan=3, sticky="w")
    row += 1

    ttk.Label(tab_enc, text="Fichier données").grid(row=row, column=0, sticky="w")
    ttk.Entry(tab_enc, textvariable=in_data_var, width=70).grid(row=row, column=1, padx=6)
    ttk.Button(tab_enc, text="Choisir…", command=lambda: browse_file(in_data_var, save=False,
                                                                     types=[("All", "*.*")])
               ).grid(row=row, column=2)
    row += 1

    ttk.Label(tab_enc, text="Image fixe (optionnelle)").grid(row=row, column=0, sticky="w")
    ttk.Entry(tab_enc, textvariable=in_image_var, width=70).grid(row=row, column=1, padx=6)
    ttk.Button(tab_enc, text="Choisir…", command=lambda: browse_file(in_image_var, save=False,
                                                                     types=[("Image", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All", "*.*")])
               ).grid(row=row, column=2)
    row += 1

    ttk.Label(tab_enc, text="Sortie vidéo lossless").grid(row=row, column=0, sticky="w")
    ttk.Entry(tab_enc, textvariable=out_video_var, width=70).grid(row=row, column=1, padx=6)
    ttk.Button(tab_enc, text="Choisir…", command=lambda: browse_file(out_video_var, save=True,
                                                                     types=[("MP4", "*.mp4"), ("All", "*.*")])
               ).grid(row=row, column=2)
    row += 1

    ttk.Label(tab_enc, text="Codec: x264 lossless (MP4 forcé)").grid(row=row, column=0, columnspan=3, sticky="w")
    row += 1

    enc_pb = ttk.Progressbar(tab_enc, mode="indeterminate")
    enc_pb.grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
    row += 1

    enc_status = tk.StringVar(value="Prêt.")
    ttk.Label(tab_enc, textvariable=enc_status).grid(row=row, column=0, columnspan=3, sticky="w")
    row += 1

    def do_encode():
        """
        But:
            Valider les champs et lancer l'encodage depuis l'UI.

        Fonctionnement:
            Vérifie les chemins, active la barre de progression et démarre
            `encode_video` dans un worker thread.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            None.
        """
        # Valide l'input puis lance l'encodage.
        iv = in_video_var.get().strip()
        idf = in_data_var.get().strip()
        iim = in_image_var.get().strip()
        ov = out_video_var.get().strip()
        if not iv or not os.path.exists(iv):
            # Decision: video source invalide -> message d'erreur.
            messagebox.showerror("Erreur", "Vidéo source invalide.")
            return
        if not idf or not os.path.exists(idf):
            # Decision: fichier donnees invalide -> message d'erreur.
            messagebox.showerror("Erreur", "Fichier données invalide.")
            return
        if iim and (not os.path.exists(iim)):
            # Decision: image optionnelle fournie mais invalide.
            messagebox.showerror("Erreur", "Image fixe invalide.")
            return
        if not ov:
            # Decision: chemin de sortie manquant -> message d'erreur.
            messagebox.showerror("Erreur", "Chemin de sortie invalide.")
            return

        enc_status.set("Encodage en cours…")
        enc_pb.start(10)
        start_worker(
            encode_video,
            iv, idf, ov,
            in_image=iim,
            log_cb=log,
            progress_cb=lambda f, consumed, total: progress("enc", f, consumed, total),
            stop_flag=stop_flag
        )

    def do_stop():
        """
        But:
            Déclencher l'arrêt des traitements en cours.

        Fonctionnement:
            Positionne `stop_event` puis journalise l'action.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            None.
        """
        # Demande l'arret des traitements.
        stop_event.set()
        log("Stop demandé par l'utilisateur.")

    def update_capacity(*_):
        """
        But:
            Mettre à jour l'estimation de capacité max affichée.

        Fonctionnement:
            Si une vidéo source valide est définie, calcule la capacité et met
            à jour le label dédié.

        Paramètres d'entrée:
            *_: arguments ignorés (callback Tk).

        Sortie:
            None.
        """
        # Calcule et affiche la capacite max des donnees selon la video source.
        path = in_video_var.get().strip()
        if not path or not os.path.exists(path):
            max_payload_var.set("Capacité max données: —")
            return
        cap = estimate_max_payload_bytes(path)
        if cap is None:
            max_payload_var.set("Capacité max données: inconnue")
        else:
            max_payload_var.set(f"Capacité max données: {format_bytes(cap)} ({cap} octets)")

    btns = ttk.Frame(tab_enc)
    btns.grid(row=row, column=0, columnspan=3, sticky="w")
    ttk.Button(btns, text="Lancer encodage", command=do_encode).pack(side="left", padx=4)
    ttk.Button(btns, text="Stop", command=do_stop).pack(side="left", padx=4)
    in_video_var.trace_add("write", update_capacity)

    # -------- Decode tab
    tab_dec = ttk.Frame(nb)
    nb.add(tab_dec, text="Décodage")

    dec_in_video_var = tk.StringVar()
    dec_out_dir_var = tk.StringVar()
    force_scale_var = tk.BooleanVar(value=True)
    strict_magic_var = tk.BooleanVar(value=False)

    def collect_gui_settings() -> dict:
        """
        But:
            Capturer l'état courant des paramètres UI persistables.

        Fonctionnement:
            Lit les champs de l'onglet encodage/décodage et l'onglet actif.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            dict: paramètres sérialisables en JSON.
        """
        return {
            "in_video": in_video_var.get().strip(),
            "in_data": in_data_var.get().strip(),
            "in_image": in_image_var.get().strip(),
            "out_video": out_video_var.get().strip(),
            "dec_in_video": dec_in_video_var.get().strip(),
            "dec_out_dir": dec_out_dir_var.get().strip(),
            "force_scale_1080p": bool(force_scale_var.get()),
            "strict_magic": bool(strict_magic_var.get()),
            "current_tab": int(nb.index("current")),
        }

    def save_gui_settings():
        """
        But:
            Sauvegarder les paramètres UI sur disque.

        Fonctionnement:
            Écrit un JSON lisible dans le dossier du script.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            None.
        """
        nonlocal settings_save_job
        settings_save_job = None
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(collect_gui_settings(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"WARN paramètres UI non sauvegardés: {e}")

    def schedule_gui_settings_save(*_):
        """
        But:
            Planifier une sauvegarde différée des paramètres UI.

        Fonctionnement:
            Annule la sauvegarde en attente puis reprogramme une écriture
            après un court délai pour éviter d'écrire à chaque frappe.

        Paramètres d'entrée:
            *_: arguments ignorés (callbacks Tk).

        Sortie:
            None.
        """
        nonlocal settings_save_job
        if settings_loading:
            # Decision: ne pas sauvegarder pendant la phase de rechargement.
            return
        if settings_save_job is not None:
            root.after_cancel(settings_save_job)
        settings_save_job = root.after(300, save_gui_settings)

    def load_gui_settings():
        """
        But:
            Recharger les paramètres UI sauvegardés.

        Fonctionnement:
            Lit le JSON si présent, restaure les variables connues et l'onglet
            actif, puis recalcule la capacité affichée.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            None.
        """
        nonlocal settings_loading
        if not os.path.exists(settings_path):
            # Decision: premier lancement, aucun fichier de settings.
            return
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                # Decision: format inattendu -> ignore silencieusement.
                return
        except Exception as e:
            log(f"WARN paramètres UI non chargés: {e}")
            return

        settings_loading = True
        try:
            # Decision: ne restaurer que les types attendus.
            if isinstance(data.get("in_video"), str):
                in_video_var.set(data["in_video"])
            if isinstance(data.get("in_data"), str):
                in_data_var.set(data["in_data"])
            if isinstance(data.get("in_image"), str):
                in_image_var.set(data["in_image"])
            if isinstance(data.get("out_video"), str):
                out_video_var.set(data["out_video"])
            if isinstance(data.get("dec_in_video"), str):
                dec_in_video_var.set(data["dec_in_video"])
            if isinstance(data.get("dec_out_dir"), str):
                dec_out_dir_var.set(data["dec_out_dir"])
            elif isinstance(data.get("dec_out_data"), str):
                # Compat ancien format de settings.
                dec_out_dir_var.set(data["dec_out_data"])
            if isinstance(data.get("force_scale_1080p"), bool):
                force_scale_var.set(data["force_scale_1080p"])
            if isinstance(data.get("strict_magic"), bool):
                strict_magic_var.set(data["strict_magic"])

            tab_idx = data.get("current_tab")
            if isinstance(tab_idx, int):
                tab_count = int(nb.index("end"))
                if 0 <= tab_idx < tab_count:
                    nb.select(tab_idx)
        finally:
            settings_loading = False

        # Decision: force la mise a jour du label capacite apres restauration.
        update_capacity()

    row2 = 0
    ttk.Label(tab_dec, text="Vidéo à décoder").grid(row=row2, column=0, sticky="w")
    ttk.Entry(tab_dec, textvariable=dec_in_video_var, width=70).grid(row=row2, column=1, padx=6)
    ttk.Button(tab_dec, text="Choisir…", command=lambda: browse_file(dec_in_video_var, save=False,
                                                                     types=[("Video", "*.mp4 *.mkv *.webm *.mov *.avi"), ("All", "*.*")])
               ).grid(row=row2, column=2)
    row2 += 1

    ttk.Label(tab_dec, text="Répertoire de sortie").grid(row=row2, column=0, sticky="w")
    ttk.Entry(tab_dec, textvariable=dec_out_dir_var, width=70).grid(row=row2, column=1, padx=6)
    ttk.Button(tab_dec, text="Choisir…", command=lambda: browse_dir(dec_out_dir_var)).grid(row=row2, column=2)
    row2 += 1

    ttk.Checkbutton(tab_dec, text="Forcer scale vers 1920×1080 au décodage", variable=force_scale_var).grid(row=row2, column=1, sticky="w")
    row2 += 1
    ttk.Checkbutton(tab_dec, text="MAGIC strict (sinon dump brut)", variable=strict_magic_var).grid(row=row2, column=1, sticky="w")
    row2 += 1

    dec_pb = ttk.Progressbar(tab_dec, mode="indeterminate")
    dec_pb.grid(row=row2, column=0, columnspan=3, sticky="ew", pady=6)
    row2 += 1

    dec_status = tk.StringVar(value="Prêt.")
    ttk.Label(tab_dec, textvariable=dec_status).grid(row=row2, column=0, columnspan=3, sticky="w")
    row2 += 1

    def do_decode():
        """
        But:
            Valider les champs et lancer le décodage depuis l'UI.

        Fonctionnement:
            Vérifie les chemins, active la barre de progression et démarre
            `decode_video` dans un worker thread.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            None.
        """
        # Valide l'input puis lance le decodage.
        iv = dec_in_video_var.get().strip()
        od = dec_out_dir_var.get().strip()
        if not iv or not os.path.exists(iv):
            # Decision: video a decoder invalide -> message d'erreur.
            messagebox.showerror("Erreur", "Vidéo à décoder invalide.")
            return
        if not od or not os.path.isdir(od):
            # Decision: repertoire de sortie invalide -> message d'erreur.
            messagebox.showerror("Erreur", "Répertoire de sortie invalide.")
            return

        dec_status.set("Décodage en cours…")
        dec_pb.start(10)
        start_worker(
            decode_video,
            iv, od,
            force_scale_1080p=force_scale_var.get(),
            strict_magic=strict_magic_var.get(),
            log_cb=log,
            progress_cb=lambda f, have, total: progress("dec", f, have, total),
            stop_flag=stop_flag
        )

    btns2 = ttk.Frame(tab_dec)
    btns2.grid(row=row2, column=0, columnspan=3, sticky="w")
    ttk.Button(btns2, text="Lancer décodage", command=do_decode).pack(side="left", padx=4)
    ttk.Button(btns2, text="Stop", command=do_stop).pack(side="left", padx=4)
    in_video_var.trace_add("write", schedule_gui_settings_save)
    in_data_var.trace_add("write", schedule_gui_settings_save)
    in_image_var.trace_add("write", schedule_gui_settings_save)
    out_video_var.trace_add("write", schedule_gui_settings_save)
    dec_in_video_var.trace_add("write", schedule_gui_settings_save)
    dec_out_dir_var.trace_add("write", schedule_gui_settings_save)
    force_scale_var.trace_add("write", schedule_gui_settings_save)
    strict_magic_var.trace_add("write", schedule_gui_settings_save)
    nb.bind("<<NotebookTabChanged>>", schedule_gui_settings_save)

    # -------- Log panel (global)
    log_frame = ttk.Frame(root)
    log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    ttk.Label(log_frame, text="Logs").pack(anchor="w")
    txt = tk.Text(log_frame, height=12)
    txt.pack(fill="both", expand=True)

    def pump():
        """
        But:
            Rafraîchir périodiquement logs et progression de l'interface.

        Fonctionnement:
            Dépile les queues `qlog`/`qprog`, met à jour l'UI puis se replanifie
            via `root.after`.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            None.
        """
        # Boucle UI: consomme logs/progress et rafraichit l'interface.
        # logs
        while True:
            try:
                msg = qlog.get_nowait()
            except queue.Empty:
                break
            ts = time.strftime("%H:%M:%S")
            txt.insert("end", f"[{ts}] {msg}\n")
            txt.see("end")
            if msg.startswith("Encodage terminé"):
                # Decision: stop progressbar encodage.
                enc_pb.stop()
                enc_status.set("Terminé.")
            if msg.startswith("Décodage terminé") or msg.startswith("Décodage (brut) terminé"):
                # Decision: stop progressbar decodage.
                dec_pb.stop()
                dec_status.set("Terminé.")
            if msg.startswith("ERREUR:"):
                # Decision: en cas d'erreur, on stoppe tout.
                enc_pb.stop()
                dec_pb.stop()
                enc_status.set("Erreur.")
                dec_status.set("Erreur.")

        # progress events
        while True:
            try:
                ev = qprog.get_nowait()
            except queue.Empty:
                break
            kind = ev[0]
            if kind == "enc":
                # Decision: mise a jour du statut encodage.
                _k, frames, consumed, total = ev
                enc_status.set(f"Frames: {frames} | Données consommées: {consumed} / {total} bytes")
            elif kind == "dec":
                # Decision: mise a jour du statut decodage.
                _k, frames, have, total = ev
                if total and total > 0:
                    # Decision: on affiche la taille attendue si connue.
                    dec_status.set(f"Frames: {frames} | Bytes récupérés (incl header): {have} | Payload attendu: {total}")
                else:
                    # Decision: taille inconnue -> affichage minimal.
                    dec_status.set(f"Frames: {frames} | Bytes récupérés (incl header): {have}")

        root.after(200, pump)

    def on_close():
        """
        But:
            Gérer la fermeture de la fenêtre principale.

        Fonctionnement:
            Sauvegarde immédiatement les paramètres, puis ferme l'UI.

        Paramètres d'entrée:
            Aucun.

        Sortie:
            None.
        """
        nonlocal settings_save_job
        if settings_save_job is not None:
            root.after_cancel(settings_save_job)
            settings_save_job = None
        save_gui_settings()
        root.destroy()

    load_gui_settings()
    root.update_idletasks()
    req_w = root.winfo_reqwidth()
    req_h = root.winfo_reqheight()
    widened_w = max(1, int(round(req_w * 1.2)))
    root.geometry(f"{widened_w}x{req_h}")
    root.protocol("WM_DELETE_WINDOW", on_close)
    pump()
    root.mainloop()

# =========================
# CLI
# =========================

def run_cli():
    """
    But:
        Exécuter le programme en mode ligne de commande.

    Fonctionnement:
        Parse les arguments puis route vers encode ou decode.

    Paramètres d'entrée:
        Aucun.

    Sortie:
        None.
    """
    # Lance le mode ligne de commande (encode/decode).
    ap = argparse.ArgumentParser(description="YouTubeNAS (encode/decode)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_enc = sub.add_parser("encode", help="Encode data into a 1080p lossless video with a centered preview area.")
    ap_enc.add_argument("--in_video", required=True)
    ap_enc.add_argument("--in_data", required=True)
    ap_enc.add_argument("--out_video", required=True)
    ap_enc.add_argument("--in_image", default="", help="Optional fixed image shown 5s at start/end and on extension.")

    ap_dec = sub.add_parser("decode", help="Decode data from a horizontal-lines video.")
    ap_dec.add_argument("--in_video", required=True)
    ap_dec.add_argument("--out_dir", required=True)
    ap_dec.add_argument("--no_force_1080p", action="store_true")
    ap_dec.add_argument("--no_strict_magic", action="store_true")

    args = ap.parse_args()

    if args.cmd == "encode":
        # Decision: executer l'encodage.
        encode_video(
            args.in_video, args.in_data, args.out_video,
            in_image=args.in_image,
            log_cb=print,
            progress_cb=lambda f, consumed, total: print(f"Frames={f} consumed={consumed}/{total} bytes"),
            stop_flag=None
        )
    elif args.cmd == "decode":
        # Decision: executer le decodage.
        decode_video(
            args.in_video, args.out_dir,
            # Decision: les flags "no_*" inversent le comportement par defaut.
            force_scale_1080p=not args.no_force_1080p,
            strict_magic=not args.no_strict_magic,
            log_cb=print,
            progress_cb=lambda f, have, total: print(f"Frames={f} bytes={have} payload={total}"),
            stop_flag=None
        )

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Decision: pas d'arguments -> lancer la GUI.
        run_gui()
    else:
        # Decision: arguments presents -> lancer la CLI.
        run_cli()
