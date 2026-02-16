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
# des symboles couleur sur des tuiles 8x8. Un centre video reste
# visible pour l'apercu. Un decodage inverse reconstruit les donnees.
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
T = 8  # tuile 8x8
SYMBOLS_PER_BYTE = 4  # 4 symboles base-6 pour coder 1 octet

# Palette BGR des symboles data: noir, blanc, rouge, vert, bleu, jaune.
SYMBOL_COLORS_BGR = np.array([
    [0, 0, 0],        # noir
    [255, 255, 255],  # blanc
    [0, 0, 255],      # rouge
    [0, 255, 0],      # vert
    [255, 0, 0],      # bleu
    [0, 255, 255],    # jaune
], dtype=np.uint8)
GUARD_BGR = np.array([128, 128, 128], dtype=np.uint8)  # bord constant (1 px)

# Header V1 (historique, sans FEC)
MAGIC_V1 = b"VDAT01\0\0"  # 8 bytes
HDR_V1_FMT = "<8sQ"       # magic + payload_size uint64
HDR_V1_SIZE = struct.calcsize(HDR_V1_FMT)

# Header V2 (avec FEC simple)
MAGIC_V2 = b"VDAT02\0\0"  # 8 bytes
# Champs: magic, payload_size, block_data_bytes, rs_parity_bytes, champ_legacy
HDR_V2_FMT = "<8sQ I H H"
HDR_V2_SIZE = struct.calcsize(HDR_V2_FMT)

# Header V3 (avec FEC + parite renforcee au debut)
MAGIC_V3 = b"VDAT03\0\0"  # 8 bytes
# Champs: magic, payload_size, block_data_bytes, rs_parity_bytes, champ_legacy,
#         strong_rs_parity_bytes, strong_prefix_pct
HDR_V3_FMT = "<8sQ I H H H H"
HDR_V3_SIZE = struct.calcsize(HDR_V3_FMT)

# Parametres FEC (RS(255,223) + CRC32)
RS_N = 255
RS_PARITY_BYTES = 32  # RS(255,223) -> 32 bytes parite, corrige ~16 bytes
RS_K = RS_N - RS_PARITY_BYTES
CRC_BYTES = 4
BLOCK_DATA_BYTES = RS_K - CRC_BYTES  # 223 - 4 = 219 bytes utiles par bloc
INTERLEAVE_DEPTH = 1  # champ legacy conserve pour compatibilite (valeur fixee a 1)

# Parite renforcee sur le debut du payload
RS_PARITY_BYTES_STRONG = 64  # RS(255,191) -> 64 bytes parite, corrige ~32 bytes
STRONG_PREFIX_PCT = 10       # protege les premiers 10% bytes avec parite forte

# Centre vidéo fixe (multiples de 8)
CENTER_W, CENTER_H = 912, 512
CENTER_X, CENTER_Y = 504, 280

# Coins sync (tile coords) + symboles
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
        Construire les tables de conversion octet <-> 4 symboles base-6.

    Fonctionnement:
        Parcourt les 256 octets, calcule leurs digits base-6 et remplit:
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
        Classer des couleurs BGR en symboles data robustes à la compression.

    Fonctionnement:
        Utilise HSV avec règles noir/blanc, puis classification de teinte
        pour rouge/jaune/vert/bleu, avec fallback RGB en cas ambigu.

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

BYTE_TO_SYMBOLS, SYMBOLS4_TO_BYTE = _build_base6_tables()

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

    # Formule: bytes/frame = floor(nb_tuiles_data / 4 symboles par byte).
    bytes_per_frame = DATA_FLAT.size // SYMBOLS_PER_BYTE
    # Formule: bytes disponibles = nframes * bytes/frame.
    total_bytes = nframes * bytes_per_frame
    # Formule: bytes FEC dispo = total_bytes - taille header v3.
    fec_bytes = total_bytes - HDR_V3_SIZE
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
    return payload_max

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

def _interleave_blocks(blocks: list, depth: int, block_len: int) -> bytes:
    """
    But:
        Réorganiser des blocs RS en flux interleavé.

    Fonctionnement:
        Entrelace les octets de groupes de blocs pour lisser les erreurs burst.

    Paramètres d'entrée:
        blocks (list): liste de blocs bytes de longueur fixe.
        depth (int): profondeur d'entrelacement.
        block_len (int): longueur d'un bloc.

    Sortie:
        bytes: flux entrelacé.
    """
    # Reorganisation legacy des octets de blocs consecutifs.
    if depth <= 1:
        # Decision: pas de reorganisation si profondeur <= 1.
        return b"".join(blocks)
    out = bytearray()
    for i in range(0, len(blocks), depth):
        group = blocks[i:i + depth]
        for pos in range(block_len):
            for b in group:
                out.append(b[pos])
    return bytes(out)

def _deinterleave_blocks(data: bytes, depth: int, block_len: int) -> list:
    """
    But:
        Reconstituer des blocs RS depuis un flux interleavé.

    Fonctionnement:
        Inverse l'entrelacement effectué à l'encodage.

    Paramètres d'entrée:
        data (bytes): flux entrelacé.
        depth (int): profondeur d'entrelacement.
        block_len (int): longueur d'un bloc.

    Sortie:
        list: liste de blocs bytes.
    """
    # Reconstruction des blocs d'origine a partir du flux legacy.
    if depth <= 1:
        # Decision: pas de traitement special si profondeur <= 1.
        return [data[i:i + block_len] for i in range(0, len(data), block_len)]
    nblocks = len(data) // block_len
    blocks = [bytearray(block_len) for _ in range(nblocks)]
    idx = 0
    for i in range(0, nblocks, depth):
        group = blocks[i:i + depth]
        for pos in range(block_len):
            for gi in range(len(group)):
                blocks[i + gi][pos] = data[idx]
                idx += 1
    return [bytes(b) for b in blocks]

def fec_encode_payload_uep(payload: bytes,
                           block_data_bytes: int,
                           rs_parity: int,
                           interleave_depth: int,
                           strong_rs_parity: int,
                           strong_prefix_pct: int) -> bytes:
    """
    But:
        Encoder un payload en UEP (protection renforcée au début).

    Fonctionnement:
        Découpe en blocs, ajoute CRC32, encode RS fort/normal puis entrelace
        le flux de blocs.

    Paramètres d'entrée:
        payload (bytes): données à encoder.
        block_data_bytes (int): taille utile des blocs normaux.
        rs_parity (int): parité RS normale.
        interleave_depth (int): profondeur d'entrelacement.
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

    # Interleaving entre blocs RS.
    return _interleave_blocks(blocks, interleave_depth, RS_N)

def fec_decode_payload(encoded: bytes,
                       payload_size: int,
                       block_data_bytes: int,
                       rs_parity: int,
                       interleave_depth: int,
                       strong_rs_parity: int,
                       strong_prefix_pct: int,
                       log_cb=None) -> bytes:
    """
    But:
        Décoder un flux FEC pour reconstruire le payload brut.

    Fonctionnement:
        Dé-entrelace les blocs, décode RS (fort/normal), vérifie CRC et tronque
        le résultat à la taille payload attendue.

    Paramètres d'entrée:
        encoded (bytes): flux FEC encodé.
        payload_size (int): taille payload cible.
        block_data_bytes (int): taille utile des blocs normaux.
        rs_parity (int): parité RS normale.
        interleave_depth (int): profondeur d'entrelacement.
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

    strong_blocks, normal_blocks, _total = _compute_blocks_for_payload(
        payload_size, block_data_bytes, strong_block_data, strong_prefix_pct
    )

    out = bytearray()
    bad_rs = 0
    bad_crc = 0

    blocks = _deinterleave_blocks(encoded, interleave_depth, RS_N)
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

def build_inner_mask(tile_rows: int, tile_cols: int) -> np.ndarray:
    """
    But:
        Construire le masque actif 6x6 au centre de chaque tuile 8x8.

    Fonctionnement:
        Crée une tuile binaire puis la réplique sur toute la grille.

    Paramètres d'entrée:
        tile_rows (int): nombre de lignes de tuiles.
        tile_cols (int): nombre de colonnes de tuiles.

    Sortie:
        np.ndarray: masque uint8 de shape (H, W).
    """
    # Construit un masque 6x6 "actif" au centre de chaque tuile 8x8.
    tile = np.zeros((T, T), dtype=np.uint8)
    tile[1:7, 1:7] = 1  # 6x6 central
    return np.tile(tile, (tile_rows, tile_cols))

def compute_tile_sets() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    But:
        Pré-calculer les ensembles de tuiles utiles au pipeline.

    Fonctionnement:
        Détermine les tuiles au centre vidéo, les tuiles data hors centre/sync,
        les index sync et le masque 6x6 global.

    Paramètres d'entrée:
        Aucun.

    Sortie:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        in_center_flat, data_flat, sync_flat, inner_mask.
    """
    # Formule: nb tuiles = dimension / taille de tuile.
    tile_rows, tile_cols = H // T, W // T

    cx, cy, cw, ch = CENTER_X, CENTER_Y, CENTER_W, CENTER_H
    # Formules: conversion pixels -> coordonnees tuiles.
    tx0, ty0 = cx // T, cy // T
    tx1, ty1 = (cx + cw) // T, (cy + ch) // T

    in_center = np.zeros((tile_rows, tile_cols), dtype=bool)
    in_center[ty0:ty1, tx0:tx1] = True
    in_center_flat = in_center.reshape(-1)

    sync_flat = set()
    for (r, c), _sym in SYNC.items():
        # Formule: index lineaire = r * nb_cols + c.
        sync_flat.add(r * tile_cols + c)

    all_idx = np.arange(tile_rows * tile_cols, dtype=np.int32)
    # Decision: data = toutes tuiles hors centre.
    data_flat = all_idx[~in_center_flat]
    # Decision: exclure aussi les tuiles de synchro des donnees.
    data_flat = np.array([i for i in data_flat if i not in sync_flat], dtype=np.int32)

    inner_mask = build_inner_mask(tile_rows, tile_cols)

    return in_center_flat, data_flat, np.array(sorted(list(sync_flat)), dtype=np.int32), inner_mask

IN_CENTER_FLAT, DATA_FLAT, SYNC_FLAT, INNER_MASK = compute_tile_sets()

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

# =========================
# Encode
# =========================

def encode_video(in_video: str,
                 in_data: str,
                 out_video: str,
                 log_cb=None,
                 progress_cb=None,
                 stop_flag=None):
    """
    But:
        Encoder un fichier binaire dans une vidéo porteuse.

    Fonctionnement:
        Lit la vidéo source, encode le payload via FEC, mappe les octets sur
        des symboles couleur de tuiles, superpose le centre vidéo puis écrit
        la vidéo lossless.

    Paramètres d'entrée:
        in_video (str): vidéo source.
        in_data (str): fichier binaire à embarquer.
        out_video (str): vidéo de sortie.
        log_cb: callback de log optionnel.
        progress_cb: callback de progression optionnel.
        stop_flag: callback booléen d'arrêt optionnel.

    Sortie:
        None.
    """
    # Encode un fichier binaire dans une video: tuiles 8x8 couleur + centre video.
    normalized_out_video = force_mp4_path(out_video)
    if normalized_out_video != out_video and log_cb:
        # Decision: notifier quand le suffixe de sortie est force en MP4.
        log_cb(f"Sortie forcée en MP4: {normalized_out_video}")

    vi = ffprobe_info(in_video)
    fps = vi.fps

    payload_size = os.path.getsize(in_data)
    # Lecture complete du payload (necessaire pour FEC).
    with open(in_data, "rb") as f:
        payload = f.read()

    # FEC: RS + CRC avec parite renforcee au debut (champ legacy fixe).
    fec_payload = fec_encode_payload_uep(
        payload,
        block_data_bytes=BLOCK_DATA_BYTES,
        rs_parity=RS_PARITY_BYTES,
        interleave_depth=INTERLEAVE_DEPTH,
        strong_rs_parity=RS_PARITY_BYTES_STRONG,
        strong_prefix_pct=STRONG_PREFIX_PCT
    )

    # Formule: header v3 = MAGIC + taille payload + params FEC + params parite forte.
    header = struct.pack(
        HDR_V3_FMT,
        MAGIC_V3,
        payload_size,
        BLOCK_DATA_BYTES,
        RS_PARITY_BYTES,
        INTERLEAVE_DEPTH,
        RS_PARITY_BYTES_STRONG,
        STRONG_PREFIX_PCT
    )
    # BitStream lit d'abord le header, puis les octets FEC (en memoire).
    bs = BitStream(header + fec_payload, None)

    # Formule: nb tuiles = dimension / taille de tuile.
    tile_rows, tile_cols = H // T, W // T
    n_data_tiles = DATA_FLAT.size
    # Formule: bytes/frame = floor(nb_tuiles_data / 4 symboles par byte).
    bytes_per_frame = n_data_tiles // SYMBOLS_PER_BYTE
    if bytes_per_frame <= 0:
        # Decision: aucune capacite data disponible.
        raise RuntimeError("Aucune tuile disponible pour encoder les donnees.")
    # Formule: nombre de symboles utiles/frame = bytes/frame * 4.
    data_symbols_per_frame = bytes_per_frame * SYMBOLS_PER_BYTE

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

    try:
        while True:
            if stop_flag and stop_flag():
                # Decision: arret demande -> sortie propre.
                if log_cb:
                    # Decision: logger seulement si callback fourni.
                    log_cb("Stop demandé. Arrêt propre…")
                break

            raw = reader.stdout.read(frame_bytes)
            have_video = (raw is not None and len(raw) == frame_bytes)

            # condition d'arrêt: plus de vidéo ET plus de data (bits + buffer)
            if (not have_video) and bs.eof and bs.bitbuf.size == 0:
                # Decision: plus rien a lire ni a encoder -> stop.
                break

            if have_video:
                # Decision: on utilise la frame video si disponible.
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((vi.height, vi.width, 3))
                center = fit_into_center(frame, CENTER_W, CENTER_H)
            else:
                # Decision: sinon on met un centre noir.
                center = np.zeros((CENTER_H, CENTER_W, 3), dtype=np.uint8)

            # Construire la tile_map (symboles 0..5)
            tile_map = np.zeros((tile_rows * tile_cols,), dtype=np.uint8)

            # Sync corners
            for (r, c), sym in SYNC.items():
                # Formule: index lineaire dans la grille de tuiles.
                tile_map[r * tile_cols + c] = sym

            # Data tiles: bytes -> 4 symboles base-6.
            bits = bs.take_bits(bytes_per_frame * 8)
            data_bytes_consumed += (bits.size // 8)
            bytes_chunk = np.packbits(bits).astype(np.uint8)
            syms = BYTE_TO_SYMBOLS[bytes_chunk].reshape(-1)
            tile_map[DATA_FLAT[:data_symbols_per_frame]] = syms

            # Color map par tuile -> expand en pixels
            tile_map_2d = tile_map.reshape((tile_rows, tile_cols))
            # Formule: map symbole -> couleur BGR.
            color_map = SYMBOL_COLORS_BGR[tile_map_2d]  # (tile_rows, tile_cols, 3)
            # Formule: repeter chaque tuile T x T pour obtenir (H, W, 3).
            colors = color_map.repeat(T, axis=0).repeat(T, axis=1)  # (H, W, 3)

            # Guard: 6x6 central porte la couleur, bord = gris.
            mask3 = INNER_MASK[:, :, None].astype(np.uint16)
            out_bgr = (colors.astype(np.uint16) * mask3 +
                       GUARD_BGR.astype(np.uint16) * (1 - mask3)).astype(np.uint8)

            # Inserer centre vidéo
            out_bgr[CENTER_Y:CENTER_Y + CENTER_H, CENTER_X:CENTER_X + CENTER_W] = center

            writer.stdin.write(out_bgr.tobytes())
            frame_idx += 1

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
        log_cb(f"Encodage terminé. Frames: {frame_idx}. Payload: {payload_size} bytes. Fichier: {normalized_out_video}")

# =========================
# Decode
# =========================

def decode_video(in_video: str,
                 out_data: str,
                 force_scale_1080p: bool = True,
                 strict_magic: bool = True,
                 log_cb=None,
                 progress_cb=None,
                 stop_flag=None):
    """
    But:
        Décoder une vidéo encodée par le programme vers un fichier binaire.

    Fonctionnement:
        Lit les frames raw, décode les symboles de tuiles, reconstitue les
        bytes, parse le header, applique le décodage FEC puis écrit le payload.

    Paramètres d'entrée:
        in_video (str): vidéo à décoder.
        out_data (str): fichier binaire de sortie.
        force_scale_1080p (bool): force le redimensionnement 1080p au decode.
        strict_magic (bool): exige un magic valide.
        log_cb: callback de log optionnel.
        progress_cb: callback de progression optionnel.
        stop_flag: callback booléen d'arrêt optionnel.

    Sortie:
        None.
    """
    # Decode une video tuilee et reconstruit le fichier binaire.
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

    # Formule: nb tuiles = dimension / taille de tuile.
    tile_rows, tile_cols = H // T, W // T
    n_data_tiles = DATA_FLAT.size
    # Formule: bytes/frame = floor(nb_tuiles_data / 4 symboles par byte).
    bytes_per_frame = n_data_tiles // SYMBOLS_PER_BYTE
    if bytes_per_frame <= 0:
        # Decision: aucune capacite data disponible.
        raise RuntimeError("Aucune tuile disponible pour decoder les donnees.")
    # Formule: nombre de symboles utiles/frame = bytes/frame * 4.
    data_symbols_per_frame = bytes_per_frame * SYMBOLS_PER_BYTE

    out_bytes = bytearray()
    invalid_symbol_quartets = 0
    uncertain_color_symbols = 0

    payload_size = None
    header_ok = False
    header_version = None
    header_size = None
    fec_block_data_bytes = None
    fec_rs_parity = None
    fec_interleave_depth = None
    fec_strong_rs_parity = None
    fec_strong_prefix_pct = None
    encoded_needed = None

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
        # Tente de parser le header des que possible (V1 ou V2).
        nonlocal payload_size, header_ok
        nonlocal header_version, header_size
        nonlocal fec_block_data_bytes, fec_rs_parity, fec_interleave_depth, encoded_needed
        nonlocal fec_strong_rs_parity, fec_strong_prefix_pct

        if header_ok:
            # Decision: deja parse, on ne refait pas.
            return

        # On attend au moins 8 bytes pour identifier la version.
        if len(out_bytes) < 8:
            # Decision: pas assez d'octets pour lire le magic.
            return

        magic = bytes(out_bytes[:8])
        if magic == MAGIC_V3:
            # Decision: header V3 (parite renforcee).
            if len(out_bytes) < HDR_V3_SIZE:
                # Decision: pas assez d'octets pour parser le header V3.
                return
            m, sz, bdb, rsp, ild, srsp, spct = struct.unpack(HDR_V3_FMT, out_bytes[:HDR_V3_SIZE])
            payload_size = int(sz)
            header_version = 3
            header_size = HDR_V3_SIZE
            fec_block_data_bytes = int(bdb)
            fec_rs_parity = int(rsp)
            fec_interleave_depth = int(ild)
            fec_strong_rs_parity = int(srsp)
            fec_strong_prefix_pct = int(spct)

            strong_block_data = _strong_block_data(fec_strong_rs_parity)
            strong_blocks, normal_blocks, total_blocks = _compute_blocks_for_payload(
                payload_size, fec_block_data_bytes, strong_block_data, fec_strong_prefix_pct
            )
            # Formule: bytes FEC requis = nb_blocs * RS_N.
            encoded_needed = total_blocks * RS_N
            header_ok = True
            if log_cb:
                # Decision: logger les parametres FEC forts.
                log_cb(
                    "Header V3 OK. Payload=%d | block=%d | rs_parity=%d | interleave=%d | "
                    "strong_parity=%d | strong_pct=%d"
                    % (payload_size, fec_block_data_bytes, fec_rs_parity, fec_interleave_depth,
                       fec_strong_rs_parity, fec_strong_prefix_pct)
                )
        elif magic == MAGIC_V2:
            # Decision: header V2 (avec FEC).
            if len(out_bytes) < HDR_V2_SIZE:
                # Decision: pas assez d'octets pour parser le header V2.
                return
            m, sz, bdb, rsp, ild = struct.unpack(HDR_V2_FMT, out_bytes[:HDR_V2_SIZE])
            payload_size = int(sz)
            header_version = 2
            header_size = HDR_V2_SIZE
            fec_block_data_bytes = int(bdb)
            fec_rs_parity = int(rsp)
            fec_interleave_depth = int(ild)
            # Formule: nb_blocs = ceil(payload_size / block_data_bytes).
            blocks = _ceil_div(payload_size, fec_block_data_bytes)
            # Formule: bytes FEC requis = nb_blocs * RS_N.
            encoded_needed = blocks * RS_N
            header_ok = True
            if log_cb:
                # Decision: logger les parametres FEC.
                log_cb(
                    "Header V2 OK. Payload=%d bytes | block=%d | rs_parity=%d | interleave=%d"
                    % (payload_size, fec_block_data_bytes, fec_rs_parity, fec_interleave_depth)
                )
        elif magic == MAGIC_V1:
            # Decision: header V1 (sans FEC).
            if len(out_bytes) < HDR_V1_SIZE:
                # Decision: pas assez d'octets pour parser le header V1.
                return
            m, sz = struct.unpack(HDR_V1_FMT, out_bytes[:HDR_V1_SIZE])
            payload_size = int(sz)
            header_version = 1
            header_size = HDR_V1_SIZE
            header_ok = True
            if log_cb:
                # Decision: logger le header si callback fourni.
                log_cb(f"Header V1 OK. Payload attendu: {payload_size} bytes.")
        else:
            if strict_magic:
                # Decision: mode strict -> erreur si MAGIC invalide.
                raise RuntimeError("MAGIC invalide. (Recompression trop agressive ? ajoute FEC / répète header)")
            else:
                # Mode permissif: on ne bloque pas.
                if log_cb:
                    # Decision: logger le mode permissif si callback fourni.
                    log_cb("MAGIC invalide, mode permissif: dump brut.")
                payload_size = None
                header_ok = True  # on arrete de re-tester

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

            # Moyennes couleur 6x6 centrales de toutes les tuiles.
            tiles = frame.reshape(tile_rows, T, tile_cols, T, 3)
            # Formule: moyenne des pixels (1:7,1:7) pour chaque tuile.
            means = tiles[:, 1:7, :, 1:7, :].mean(axis=(1, 3))  # (tile_rows, tile_cols, 3)

            # Data colors (flat).
            means_flat = means.reshape(-1, 3)
            data_colors = means_flat[DATA_FLAT].astype(np.float32)

            # Classification tolerante en 6 couleurs.
            syms, uncertain = classify_data_symbols_bgr_tolerant(data_colors)
            uncertain_color_symbols += int(uncertain)

            # 4 symboles -> 1 byte via table inverse.
            syms_used = syms[:data_symbols_per_frame].reshape(-1, SYMBOLS_PER_BYTE)
            bytes_new = SYMBOLS4_TO_BYTE[
                syms_used[:, 0], syms_used[:, 1], syms_used[:, 2], syms_used[:, 3]
            ]
            invalid = (bytes_new < 0)
            if np.any(invalid):
                # Decision: quartet invalide -> byte 0.
                invalid_symbol_quartets += int(invalid.sum())
                bytes_new = bytes_new.copy()
                bytes_new[invalid] = 0
            out_bytes.extend(bytes_new.astype(np.uint8).tobytes())

            # Parse header si possible
            parse_header_if_possible()

            # Stop condition selon version
            if header_ok and payload_size is not None:
                if header_version == 1:
                    # Formule: bytes utiles = len(out_bytes) - taille header.
                    have = len(out_bytes) - header_size
                    if have >= payload_size:
                        # Decision: on a recupere tout le payload -> stop.
                        break
                elif header_version in (2, 3) and encoded_needed is not None:
                    # Formule: bytes FEC = len(out_bytes) - taille header.
                    have = len(out_bytes) - header_size
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

    if invalid_symbol_quartets > 0 and log_cb:
        # Decision: signaler les quartets invalides detectes au decode.
        log_cb(f"Décodage couleurs: quartets invalides remplacés par 0 = {invalid_symbol_quartets}")
    if uncertain_color_symbols > 0 and log_cb:
        # Decision: signaler les classifications couleur peu confiantes.
        log_cb(f"Décodage couleurs: symboles classés en mode tolérant = {uncertain_color_symbols}")

    # Write output
    if header_ok and payload_size is not None and header_size is not None:
        if header_version == 1:
            # Decision: format V1 -> ecriture payload direct.
            if len(out_bytes) >= header_size:
                payload = out_bytes[header_size:header_size + payload_size]
                with open(out_data, "wb") as f:
                    f.write(payload)
                if log_cb:
                    # Decision: logger le resume seulement si callback fourni.
                    log_cb(f"Décodage terminé. Frames lues: {frame_idx}. Écrit: {len(payload)} bytes.")
            else:
                if log_cb:
                    # Decision: logger l'incomplet si callback fourni.
                    log_cb("Décodage V1 incomplet: header present mais payload manquant.")
        elif header_version == 2:
            # Decision: format V2 -> decode FEC (sans parite forte).
            if encoded_needed is None:
                # Decision: si taille FEC inconnue, on force a zero.
                encoded_needed = 0
            encoded = out_bytes[header_size:header_size + encoded_needed]
            payload = fec_decode_payload(
                encoded,
                payload_size=payload_size,
                block_data_bytes=fec_block_data_bytes,
                rs_parity=fec_rs_parity,
                interleave_depth=fec_interleave_depth,
                strong_rs_parity=fec_rs_parity,
                strong_prefix_pct=0,
                log_cb=log_cb
            )
            with open(out_data, "wb") as f:
                f.write(payload)
            if log_cb:
                # Decision: logger le resume seulement si callback fourni.
                log_cb(f"Décodage terminé (V2 FEC). Frames lues: {frame_idx}. Écrit: {len(payload)} bytes.")
        elif header_version == 3:
            # Decision: format V3 -> decode FEC avec parite forte.
            if encoded_needed is None:
                # Decision: si taille FEC inconnue, on force a zero.
                encoded_needed = 0
            encoded = out_bytes[header_size:header_size + encoded_needed]
            payload = fec_decode_payload(
                encoded,
                payload_size=payload_size,
                block_data_bytes=fec_block_data_bytes,
                rs_parity=fec_rs_parity,
                interleave_depth=fec_interleave_depth,
                strong_rs_parity=fec_strong_rs_parity,
                strong_prefix_pct=fec_strong_prefix_pct,
                log_cb=log_cb
            )
            with open(out_data, "wb") as f:
                f.write(payload)
            if log_cb:
                # Decision: logger le resume seulement si callback fourni.
                log_cb(f"Décodage terminé (V3 FEC+strong). Frames lues: {frame_idx}. Écrit: {len(payload)} bytes.")
        else:
            # Decision: version inconnue -> dump brut.
            with open(out_data, "wb") as f:
                f.write(out_bytes)
            if log_cb:
                # Decision: logger le resume seulement si callback fourni.
                log_cb(f"Décodage (brut) terminé. Frames lues: {frame_idx}. Écrit: {len(out_bytes)} bytes.")
    else:
        # Mode permissif / brut si header absent.
        with open(out_data, "wb") as f:
            f.write(out_bytes)
        if log_cb:
            # Decision: logger le resume seulement si callback fourni.
            log_cb(f"Décodage (brut) terminé. Frames lues: {frame_idx}. Écrit: {len(out_bytes)} bytes.")

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
    root.title("Video Tiles Data PoC (encode / decode)")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)

    # -------- Encode tab
    tab_enc = ttk.Frame(nb)
    nb.add(tab_enc, text="Encodage")

    in_video_var = tk.StringVar()
    in_data_var = tk.StringVar()
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
        ov = out_video_var.get().strip()
        if not iv or not os.path.exists(iv):
            # Decision: video source invalide -> message d'erreur.
            messagebox.showerror("Erreur", "Vidéo source invalide.")
            return
        if not idf or not os.path.exists(idf):
            # Decision: fichier donnees invalide -> message d'erreur.
            messagebox.showerror("Erreur", "Fichier données invalide.")
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
    dec_out_data_var = tk.StringVar()
    force_scale_var = tk.BooleanVar(value=True)
    strict_magic_var = tk.BooleanVar(value=True)

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
            "out_video": out_video_var.get().strip(),
            "dec_in_video": dec_in_video_var.get().strip(),
            "dec_out_data": dec_out_data_var.get().strip(),
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
            if isinstance(data.get("out_video"), str):
                out_video_var.set(data["out_video"])
            if isinstance(data.get("dec_in_video"), str):
                dec_in_video_var.set(data["dec_in_video"])
            if isinstance(data.get("dec_out_data"), str):
                dec_out_data_var.set(data["dec_out_data"])
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

    ttk.Label(tab_dec, text="Sortie données").grid(row=row2, column=0, sticky="w")
    ttk.Entry(tab_dec, textvariable=dec_out_data_var, width=70).grid(row=row2, column=1, padx=6)
    ttk.Button(tab_dec, text="Choisir…", command=lambda: browse_file(dec_out_data_var, save=True,
                                                                     types=[("All", "*.*")])
               ).grid(row=row2, column=2)
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
        od = dec_out_data_var.get().strip()
        if not iv or not os.path.exists(iv):
            # Decision: video a decoder invalide -> message d'erreur.
            messagebox.showerror("Erreur", "Vidéo à décoder invalide.")
            return
        if not od:
            # Decision: chemin de sortie manquant -> message d'erreur.
            messagebox.showerror("Erreur", "Chemin de sortie invalide.")
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
    out_video_var.trace_add("write", schedule_gui_settings_save)
    dec_in_video_var.trace_add("write", schedule_gui_settings_save)
    dec_out_data_var.trace_add("write", schedule_gui_settings_save)
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
    ap = argparse.ArgumentParser(description="Video Tiles Data PoC (encode/decode)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_enc = sub.add_parser("encode", help="Encode data into a 1080p lossless video with a 512p center video.")
    ap_enc.add_argument("--in_video", required=True)
    ap_enc.add_argument("--in_data", required=True)
    ap_enc.add_argument("--out_video", required=True)

    ap_dec = sub.add_parser("decode", help="Decode data from a tiles video.")
    ap_dec.add_argument("--in_video", required=True)
    ap_dec.add_argument("--out_data", required=True)
    ap_dec.add_argument("--no_force_1080p", action="store_true")
    ap_dec.add_argument("--no_strict_magic", action="store_true")

    args = ap.parse_args()

    if args.cmd == "encode":
        # Decision: executer l'encodage.
        encode_video(
            args.in_video, args.in_data, args.out_video,
            log_cb=print,
            progress_cb=lambda f, consumed, total: print(f"Frames={f} consumed={consumed}/{total} bytes"),
            stop_flag=None
        )
    elif args.cmd == "decode":
        # Decision: executer le decodage.
        decode_video(
            args.in_video, args.out_data,
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
