import csv
import unicodedata
from typing import List, Dict, Iterable, Optional
import csv
import re
import unicodedata
from typing import Dict, List, Iterator, Optional
import csv
import os
def _normalize(s: str, case_sensitive: bool) -> str:
    """
    Normalizza una stringa per confronti robusti:
    - NFKC per unificare caratteri simili (virgolette tipografiche, spazi, ecc.)
    - rimuove spazi doppi e strip
    - opzionale: casefold per confronto case-insensitive
    """
    if s is None:
        return ""
    # normalizza unicode (virgolette “ ” vs " ecc.)
    s = unicodedata.normalize("NFKC", s)
    # compatta whitespace
    s = " ".join(s.split())
    if not case_sensitive:
        s = s.casefold()
    return s

def light_clean(s: str, lowercase: bool = True) -> str:
    # 1) rimuovi SOLO i link (mantieni #, @, emoji, ecc.)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)

    # 2) rimuovi placeholder comuni
    s = s.replace("[URL]", " ").replace("[USER]", " ")

    # 3) normalizza apostrofi tipografici
    s = re.sub(r"[’`]", "'", s)

    # 4) rimuovi spazi/controlli strani e compatta whitespace
    s = s.replace("\u200b", "")              # zero-width space
    s = re.sub(r"\s+", " ", s).strip()

    # 5) minuscolo opzionale
    if lowercase:
        s = s.lower()

    return s

def find_rows_by_post_text(
    tsv_path: str,
    post_text_query: str,
    *,
    exact_match: bool = True,
    case_sensitive: bool = False,
    encoding: str = "utf-8",
) -> List[Dict[str, str]]:
    """
    Cerca nel file TSV le righe in cui la colonna 'post_text' corrisponde alla query.
    
    Parametri:
        tsv_path: percorso al file .tsv
        post_text_query: testo da cercare nella colonna 'post_text'
        exact_match: 
            - True  => corrispondenza esatta (dopo normalizzazione)
            - False => corrispondenza per sottostringa (query ⊆ post_text)
        case_sensitive: confronto case-sensitive (default False)
        encoding: encoding del file (default 'utf-8')
    
    Ritorna:
        Lista di righe (dizionari). Se nessuna corrispondenza, lista vuota.
    """
    results: List[Dict[str, str]] = []
    q_norm = _normalize(post_text_query, case_sensitive)

    with open(tsv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "post_text" not in (reader.fieldnames or []):
            raise ValueError("Colonna 'post_text' non trovata nel TSV.")

        for row in reader:
            text = row.get("post_text", "")
            # text=light_clean(text)

            if exact_match:
                if text == post_text_query:
                    results.append(row)
            else:
                if q_norm in post_text_query:
                    results.append(row)

    return results


def append_row_to_tsv(tsv_path: str, new_row: dict, encoding="utf-8"):
    file_exists = os.path.exists(tsv_path)
    needs_header = True
    fieldnames = list(new_row.keys())

    if file_exists and os.path.getsize(tsv_path) > 0:
        with open(tsv_path, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            # se il file ha header valido lo riuso, altrimenti lo (ri)scrivo
            if reader.fieldnames:
                fieldnames = reader.fieldnames
                needs_header = False

    # Scrivi (eventuale) header + riga
    write_header = not file_exists or needs_header
    with open(tsv_path, "a", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(new_row)

def _norm(s: str, case_sensitive: bool) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.split())
    return s if case_sensitive else s.casefold()

FINE_LABELS = ["Ad-hominem", "Appeal-to-authority", "Appeal-to-emotion", "Causal-oversimplification", 
    "Cherry-picking", "Circular-reasoning", "Doubt", "Evading-the-burden-of-proof", "False-analogy", 
    "False-dilemma", "Flag-waving", "Hasty-generalization", "Loaded-language", "Name-calling-or-labelling", 
    "Red-herring", "Slippery-slope", "Slogan", "Strawman", "Thought-terminating-cliches", "Vagueness"]

def allinea_etichette(tsv_path_lettura,tsv_path_scrittura):
        with open(tsv_path_lettura, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            righe_finali=[]
            for row in reader:
                label_a1 = row.get("labels_a1", "")
                list_label_predette=label_a1.split("|")
                label_finali_predette=[]
                for classe in FINE_LABELS:
                    if classe.replace("-","").replace(" ","").lower() in list_label_predette:
                        label_finali_predette.append(classe)
                
                row["labels_a1"]="|".join(label_finali_predette)
                row["labels_a2"]="|".join(label_finali_predette)
                fieldnames = list(row.keys())


                #remove column index
                if "Index" in fieldnames:
                    fieldnames.remove("Index")
                    row.pop("Index", None)

                with open(tsv_path_scrittura, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                    writer.writerow(row)


def iter_token_blocks(path: str, encoding: str = "utf-8") -> Iterator[Dict]:
    """
    Itera i blocchi del file token-level con header tipo:
      # post_id = ...
      # post_date = ...
      # post_topic_keywords = ...
      # post_text = ...
    seguito da righe: index \t token \t labels_a1 \t labels_a2

    Ritorna per ogni blocco un dict:
      {
        "meta": {"post_id": "...", "post_date": "...", "post_topic_keywords": "...", "post_text": "..."},
        "tokens": [ {"i": "1", "token": "Cioè", "labels_a1": "O", "labels_a2": "O"}, ... ]
      }
    """
    header_re = re.compile(r"^#\s*(post_id|post_date|post_topic_keywords|post_text)\s*=\s*(.*)$")
    block_meta = {}
    block_rows: List[Dict[str, str]] = []
    in_block = False

    def _emit_current():
        nonlocal block_meta, block_rows, in_block
        if block_meta:
            yield {"meta": block_meta, "tokens": block_rows}
        block_meta = {}
        block_rows = []
        in_block = False

    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:  # riga vuota = separatore blocchi
                # emetti blocco se c'è
                yield from _emit_current()
                continue

            if line.startswith("#"):
                m = header_re.match(line)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    # Se inizia un nuovo blocco e avevamo già header+righe, emetti quello precedente
                    if key == "post_id" and block_meta and block_rows:
                        yield from _emit_current()
                    block_meta[key] = val
                    in_block = True
                # altre righe commento: ignora
                continue

            # riga token (4 colonne separate da TAB)
            if in_block:
                parts = line.split("\t")
                if len(parts) >= 2:
                    # gestiamo file con 2, 3 o 4 colonne (qualche dataset può avere labels mancanti)
                    i = parts[0]
                    token = parts[1]
                    labels_a1 = parts[2] if len(parts) > 2 else ""
                    labels_a2 = parts[3] if len(parts) > 3 else ""
                    block_rows.append({"i": i, "token": token, "labels_a1": labels_a1, "labels_a2": labels_a2})

        # EOF: emetti eventuale blocco residuo
        if block_meta:
            yield {"meta": block_meta, "tokens": block_rows}

def find_blocks(
    path: str,
    *,
    post_id: Optional[str] = None,
    post_text: Optional[str] = None,
    exact_match: bool = True,
    case_sensitive: bool = False,
    encoding: str = "utf-8",
) -> List[Dict]:
    """
    Cerca blocchi per post_id o per post_text.
    - Se fornisci post_id, usa quello (match testuale).
    - Altrimenti cerca per post_text (esatto o sottostringa, dopo normalizzazione).
    Ritorna lista di blocchi (ognuno: {"meta": {...}, "tokens": [...]})
    """
    if not post_id and not post_text:
        raise ValueError("Specifica almeno 'post_id' o 'post_text'.")

    results: List[Dict] = []
    q_norm = _norm(post_text, case_sensitive) if post_text else None

    for block in iter_token_blocks(path, encoding=encoding):
        meta = block["meta"]
        if post_id and str(meta.get("post_id")) == str(post_id):
            results.append(block)
            continue

        if post_text:
            t_norm = _norm(meta.get("post_text", ""), case_sensitive)
            if exact_match and t_norm == q_norm:
                results.append(block)
            elif not exact_match and q_norm in t_norm:
                results.append(block)

    return results

# Esempi d'uso:
if __name__ == "__main__":
    path_lettura="output_test_GPT5.tsv"
    path_scrittura="output_predetto_finale_GPT53.tsv"

    allinea_etichette(path_lettura,path_scrittura)