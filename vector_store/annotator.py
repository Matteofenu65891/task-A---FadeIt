import re
from typing import List, Dict, Any, Tuple

def _tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    pattern = r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+|['’`´]+|[\[\]\(\)\{\}\.\,\;\:\!\?\"“”«»…–—\-]|[^\s]"
    tokens = []
    for m in re.finditer(pattern, text, flags=re.UNICODE):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens

def _find_all_occurrences(haystack: str, needle: str) -> List[Tuple[int, int]]:
    if not needle:
        return []
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())
    H = haystack
    N = norm(needle)
    if not N:
        return []
    first = re.escape(N.split(" ")[0])
    occurrences = []
    for m in re.finditer(first, H, flags=re.IGNORECASE|re.UNICODE):
        pattern = re.escape(N).replace(r"\ ", r"\s+")
        start_scan = max(0, m.start() - 50)
        end_scan   = min(len(H), m.start() + len(N) + 200)
        sub = H[start_scan:end_scan]
        m2 = re.search(pattern, sub, flags=re.IGNORECASE|re.UNICODE)
        if m2:
            s = start_scan + m2.start()
            e = start_scan + m2.end()
            occurrences.append((s, e))
    return sorted(set(occurrences))

def _char_spans_to_token_idxs(spans: List[Tuple[int, int]], tokens: List[Tuple[str,int,int]]) -> List[List[int]]:
    idx_spans = []
    for s,e in spans:
        covered = []
        for i,(_, ts, te) in enumerate(tokens):
            if max(s, ts) < min(e, te):
                covered.append(i)
        if covered:
            idx_spans.append(covered)
    return idx_spans

def _bio_from_token_spans(token_len: int, idx_spans: List[List[int]], label: str) -> List[str]:
    tags = ["O"] * token_len
    for span in idx_spans:
        if not span:
            continue
        if tags[span[0]] == "O":
            tags[span[0]] = f"B-{label}"
        else:
            # già presente qualcosa → mantieni esistente qui (verrà unito dopo)
            pass
        for j in span[1:]:
            if tags[j] == "O":
                tags[j] = f"I-{label}"
    return tags

def genera_annotazione_BIO(prediction: List[Dict[str, Any]], testo_originale: str) -> str:
    """
    Input:
      prediction: [{"text":..., "fallacia":..., "confidenza":...}, ...]
      testo_originale: stringa del post

    Output:
      Riga di header "# post_text = ..."
      Poi: index \t token \t BIO  (multiclasse, etichette unite con '|')
    """
    tokens = _tokenize_with_spans(testo_originale)
    if not tokens:
        return "# post_text = (vuoto)\n"

    prepared = []
    for p in prediction:
        spans_char = _find_all_occurrences(testo_originale, p.text)
        spans_tok  = _char_spans_to_token_idxs(spans_char, tokens)
        if spans_tok:
            prepared.append({
                "label": str(p.fallacia),
                "conf": float(p.confidenza),
                "spans": spans_tok,
            })

    # Se nessun match → tutto O
    if not prepared:
        header = f"# post_text = {testo_originale}\n"
        lines = [f"{i}\t{tok}\tO" for i,(tok,_,_) in enumerate(tokens, start=1)]
        return header + "\n".join(lines)

    # Costruisci i tag monoclasse per ogni etichetta
    mono = []
    for item in prepared:
        lab, conf, idx_spans = item["label"], item["conf"], item["spans"]
        mono.append((lab, conf, _bio_from_token_spans(len(tokens), idx_spans, lab)))

    # Merge multiclasse: per token, unisci tag non-O in ordine di conf decrescente
    merged = []
    for t in range(len(tokens)):
        cell = []
        for lab, conf, tags in sorted(mono, key=lambda x: (-x[1], x[0])):
            if tags[t] != "O":
                cell.append(tags[t])
        merged.append("|".join(cell) if cell else "O")

    header = f"# post_text = {testo_originale}\n"
    lines = [f"{i}\t{tok}\t{merged[i-1]}" for i,(tok,_,_) in enumerate(tokens, start=1)]
    return header + "\n".join(lines)

if __name__ == "__main__":
    prediction = [
    {"text": "boom di migranti infetti", "fallacia": "Loaded-language", "confidenza": 85.0},
    {"text": "migranti infetti", "fallacia": "Name-calling-or-labelling", "confidenza": 84.0},
    {"text": "e' boom di migranti infetti nei centri profughi", "fallacia": "Evading-the-burden-of-proof", "confidenza": 70.0},
    {"text": "e' boom", "fallacia": "Vagueness", "confidenza": 66.0},]
    testo = "AIDS, E’ BOOM DI MIGRANTI INFETTI NEI CENTRI PROFUGHI [URL]"
    print(genera_annotazione_BIO(prediction, testo))