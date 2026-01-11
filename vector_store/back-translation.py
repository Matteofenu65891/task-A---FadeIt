#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Back-translation mirata (IT↔ES/FR) con filtro di similarità per AE/LL/SL.
- Accetta una lista di esempi: {"text": str, "labels": [str, ...], ...}
- Genera 0..N parafrasi per esempio (parametrico), preservando le etichette originali.
- Applica filtro di similarità semantica per scartare parafrasi troppo lontane.
- Mantiene hashtag/URL/emoji senza alterarli.
"""

from typing import List, Dict, Iterable, Tuple
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Union

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from process import light_clean

# =========================
# Config & utility
# =========================

TARGET_LABELS = {
    'Ad-hominem', 'Appeal-to-authority', 'Appeal-to-emotion',
    'Causal-oversimplification', 'Cherry-picking', 'Circular-reasoning',
    'Doubt', 'Evading-the-burden-of-proof', 'False-analogy', 'False-dilemma',
    'Flag-waving', 'Hasty-generalization', 'Loaded-language',
    'Name-calling-or-labelling', 'Red-herring', 'Slippery-slope', 'Slogan',
    'Strawman', 'Thought-terminating-cliches', 'Vagueness'}  # AE/LL/SL

# Modelli traduzione (Helsinki-NLP - piccoli e affidabili per IT<->ES/FR)
MODELS = {
    ("it", "es"): "Helsinki-NLP/opus-mt-it-es",
    ("es", "it"): "Helsinki-NLP/opus-mt-es-it",
    ("it", "fr"): "Helsinki-NLP/opus-mt-it-fr",
    ("fr", "it"): "Helsinki-NLP/opus-tatoeba-fr-it",
}

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

URL_RE = re.compile(r"(https?://\S+)")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
USER_RE = re.compile(r"(?<!\w)@(\w+)")
MULTI_SPACE_RE = re.compile(r"\s+")

def _mask_specials(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Maschera URL, hashtag e @mention per evitare distorsioni durante la traduzione,
    poi li ripristina sulle parafrasi.
    """
    mapping = {}
    idx = 0

    def repl_url(m):
        nonlocal idx
        key = f"<URL_{idx}>"; idx += 1
        mapping[key] = m.group(0)
        return key

    def repl_hash(m):
        nonlocal idx
        key = f"<HASH_{idx}>"; idx += 1
        mapping[key] = f"#{m.group(1)}"
        return key

    def repl_user(m):
        nonlocal idx
        key = f"<USER_{idx}>"; idx += 1
        mapping[key] = f"@{m.group(1)}"
        return key

    text2 = URL_RE.sub(repl_url, text)
    text2 = HASHTAG_RE.sub(repl_hash, text2)
    text2 = USER_RE.sub(repl_user, text2)
    return text2, mapping

def _unmask_specials(text: str, mapping: Dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return MULTI_SPACE_RE.sub(" ", text).strip()

@dataclass
class BTConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8         # leggera diversità
    top_p: float = 0.95
    similarity_threshold: float = 0.75
    n_per_lang: int = 1              # quante parafrasi per lingua pivot
    langs: Tuple[str, ...] = ("es", "fr")
    seed: int = 13

class BackTranslator:
    def __init__(self, cfg: BTConfig):
        self.cfg = cfg
        random.seed(cfg.seed)

        # Carica pipeline di traduzione per le 4 direzioni necessarie
        self.pipes = {}
        for (src, tgt), model_id in MODELS.items():
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            self.pipes[(src, tgt)] = pipeline(
                "translation",
                model=mdl,
                tokenizer=tok,
                src_lang=src,
                tgt_lang=tgt,
                device_map="auto",
                truncation=True,
                max_length=256,
            )

        # Embedding per filtro di similarità
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def translate(self, text: str, src: str, tgt: str) -> str:
        pp = self.pipes[(src, tgt)]
        out = pp(
            text,
            max_length=256,
            num_beams=4,
        )[0]["translation_text"]
        return out

    def back_translate_once(self, text: str, pivot_lang: str) -> str:
        t_masked, mapping = _mask_specials(text)
        to_pivot = self.translate(t_masked, "it", pivot_lang)
        # leggera variazione stocastica in ritraduzione con sampling
        pp_back = self.pipes[(pivot_lang, "it")]
        back = pp_back(
            to_pivot,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            top_p=self.cfg.top_p,
            temperature=self.cfg.temperature,
            num_return_sequences=1,
        )[0]["translation_text"]
        return _unmask_specials(back, mapping)

    def cosine_sim(self, a: str, b: str) -> float:
        embs = self.embedder.encode([a, b], convert_to_tensor=True, show_progress_bar=False)
        sim = util.cos_sim(embs[0], embs[1]).item()
        return float(sim)

    def augment_example(self, text: str) -> List[str]:
        """Genera parafrasi filtrate per similarità, su più lingue pivot."""
        candidates = []
        for lg in self.cfg.langs:
            for _ in range(self.cfg.n_per_lang):
                try:
                    bt = self.back_translate_once(text, pivot_lang=lg)
                    candidates.append(bt)
                except Exception:
                    continue
        # Dedup soft
        uniq = []
        seen = set()
        for c in candidates:
            k = c.strip().lower()
            if k not in seen:
                seen.add(k)
                uniq.append(c)

        # Filtro di similarità
        kept = []
        for c in uniq:
            sim = self.cosine_sim(text, c)
            if sim >= self.cfg.similarity_threshold and c != text:
                kept.append(c)
        return kept

def augment_dataset(
    data: Iterable[Dict],
    cfg: BTConfig,
    max_augment_ratio: float = 1.0,
) -> List[Dict]:
    """
    data: iterabile di esempi {"text": str, "labels": [str, ...], ...}
    Ritorna lista estesa con esempi parafrasati (stesse labels).
    max_augment_ratio: limite per evitare di superare X volte gli originali ciblati.
    """
    bt = BackTranslator(cfg)
    out = []
    pool_targets = []

    for ex in data:
        labels = set(ex.get("labels", []))
        out.append(ex)
        if labels & TARGET_LABELS:
            pool_targets.append(ex)

    # Bilanciamento: non superare una moltiplicazione eccessiva
    max_new = int(len(pool_targets) * cfg.n_per_lang * len(cfg.langs) * max_augment_ratio)

    new_count = 0
    for ex in pool_targets:
        if new_count >= max_new:
            break
        paras = bt.augment_example(ex["text"])
        for p in paras:
            out.append({
                **ex,                # conserva metadati (post_id, topic, ecc.)
                "text": p,           # sostituisce il testo con la parafrasi
                "augmented": True,
                "augment_src": "back-translation",
            })

            print(f"Parafrasi generata: {p}\n")
            print("testo originale:", ex["text"], "\n")
            new_count += 1
            if new_count >= max_new:
                break

    return out

REQUIRED_COLS = [
    "post_id",
    "post_date",
    "post_topic_keywords",
    "post_text",
    "labels_a1",
    "labels_a2",
]

def save_augmented_to_tsv(
    original_data: Union[str, pd.DataFrame],
    augmented_rows: List[Dict],
    out_path: str,
    drop_duplicates: bool = True,
) -> None:

    # Carica originali se è un path
    if isinstance(original_data, str):
        # prova a capire se è TSV o CSV
        if original_data.lower().endswith((".tsv", ".txt")):
            df_orig = pd.read_csv(original_data, sep="\t", dtype=str, keep_default_na=False)
        else:
            df_orig = pd.read_csv(original_data, sep=",", dtype=str, keep_default_na=False)
    else:
        df_orig = original_data.copy()

    # Validazione colonne minime
    missing = [c for c in REQUIRED_COLS if c not in df_orig.columns]
    if missing:
        raise ValueError(f"Mancano colonne nell'originale: {missing}")

    # Costruisci DataFrame delle righe augmentate
    # (manteniamo SOLO le colonne richieste e lo stesso ordine)
    df_aug = pd.DataFrame(augmented_rows)
    missing_aug = [c for c in REQUIRED_COLS if c not in df_aug.columns]
    if missing_aug:
        raise ValueError(f"Mancano colonne nelle righe augmentate: {missing_aug}")

    df_aug = df_aug[REQUIRED_COLS].astype(str)

    # (Opzionale) rimuovi duplicati perfetti
    if drop_duplicates:
        df_aug = df_aug.drop_duplicates()

    # Scrivi TSV con utf-8 e senza indici
    df_aug.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

    print(f"✅ Salvate {len(df_aug)} righe augmentate in: {out_path}")

# -------------------------
# ESEMPIO D'USO
# -------------------------
if __name__ == "__main__":
    # Esempi minimi (notare labels mirate)
    samples = [
        {"text": "Che schifo! Queste persone ci invadono!", "labels": ["Appeal to emotion", "Loaded language"]},
        {"text": "Basta con le promesse vuote. Il 25 settembre vota X.", "labels": ["Slogan"]},
        {"text": "Secondo Tizio, è così e basta.", "labels": ["Appeal to authority", "Thought-terminating cliché"]},
    ]

    file_annotazioni="train-dev.tsv"
    samples=[]
    df = pd.read_csv(file_annotazioni, sep="\t", encoding="utf-8")

    #take 140 random row
    test_df = df.sample(n=500)
    train_df = df.drop(test_df.index)

    y_pred = []
    y_true = []
    i=0
    for row in test_df.itertuples():
        text=row.post_text
        text=light_clean(text)
        target=row.labels_a1
        samples.append({"text": text, "labels": target.split("|") if isinstance(target, str) else []})

    cfg = BTConfig(
        similarity_threshold=0.75,
        n_per_lang=2,
        langs=("es", "fr"),
        temperature=0.9,
        top_p=0.95,
        seed=42,
    )

    augmented = augment_dataset(samples, cfg, max_augment_ratio=1.0)

    df_orig = pd.read_csv("train-dev.tsv", sep="\t", dtype=str, keep_default_na=False)

    id=random.randint(0,10000)
    if augmented:
        for ex in augmented:
            text=ex["text"]
            target="|".join(ex.get("labels", []))
            post_id=id
            post_topic_keywords="back-translation"
            labels_a1=target
            labels_a2=target
            new_row={
                "post_id": post_id,
                "post_date": "2024-01-01",
                "post_topic_keywords": post_topic_keywords,
                "post_text": text,
                "labels_a1": labels_a1,
                "labels_a2": labels_a2,
            }
            df_orig = pd.concat([df_orig, pd.DataFrame([new_row])], ignore_index=True)
            id+=1

    df_orig.to_csv("train-dev-augmented.tsv", sep="\t", index=False, encoding="utf-8")