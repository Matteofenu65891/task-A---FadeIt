import os
import Annotazione
import json
import logging
import re
import pandas as pd
from typing import Dict, List, Union
import csv

DATA_FOLDER=os.path.join(os.path.dirname(__file__), 'faina-v1.0_fadeit-shared-task','data')
FILE_PATH=os.path.join(DATA_FOLDER, 'subtask-b', 'train-dev.conll')

logging.basicConfig(
    filename="log_info.log",          # nome file di log
    filemode="a",                # "a" = append, "w" = overwrite
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO           # livello minimo da loggare
)

# Funzione per leggere i dati da un file
def readData(filePath=FILE_PATH):
    with open(filePath, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]

def main():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    posts = parse_conll(text)

def parse_conll(text, id: int = 0):
    posts = []
    current_post = None
    ann_A = []
    ann_B = []
    current_id = id
    for line in text.splitlines():
        line = line.strip()

        # Inizio di un nuovo post (metadati con "# ...")
        if line.startswith("# post_id"):
            # Se c'era un post precedente, salvalo

            if current_post:
                current_post["list_annotatore_A"] = ann_A
                current_post["list_annotatore_B"] = ann_B
                annotazione=Annotazione.Annotazione(current_post)
                posts.append(annotazione)

                if id != -1 and current_id == id:
                    return posts
                else:
                    current_id= int(line.split("=")[1].strip())
                    
            # Nuovo post
            current_post = {}
            ann_A = []
            ann_B = []

            current_post["post_id"] = int(line.split("=")[1].strip())

        elif line.startswith("# post_date"):
            current_post["post_date"] = line.split("=")[1].strip()

        elif line.startswith("# post_topic_keywords"):
            current_post["post_topic_keywords"] = line.split("=")[1].strip()

        elif line.startswith("# post_text"):
            current_post["post_text"] = line.split("=", 1)[1].strip()

        # Riga vuota → fine frase
        elif line == "":
            continue

        # Riga con token e annotazioni
        else:
            parts = line.split("\t")
            if len(parts) >= 4:
                token_id, token, annA, annB = parts
                ann_A.append((token, annA))
                ann_B.append((token, annB))            

    # Non dimenticare l’ultimo post
    if current_post:
        current_post["list_annotatore_A"] = ann_A
        current_post["list_annotatore_B"] = ann_B
        annotazione=Annotazione.Annotazione(current_post)
        posts.append(annotazione)
    return posts

OUTPUT_CSV = "fallacies.csv"

def _ensure_csv_writer(path):
    """Apre il CSV in append e scrive l'header se manca. Ritorna (file_handle, csv_writer)."""
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    f = open(path, "a", encoding="utf-8", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["text", "target"])
    return f, writer

def _rows_from_final_dict(final_dict):
    """Converte final_dict -> lista di tuple (text, target), rimuovendo '(i) ' dal testo."""
    rows = []
    for target, texts in final_dict.items():
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        for t in texts:
            if t is None:
                continue
            text = str(t).strip()
            text = re.sub(r"^\(\d+\)\s*", "", text)  # rimuove prefisso (indice)
            if text:
                rows.append((text, target))
    return rows

def process():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    posts = parse_conll(text)

    fcsv, writer = _ensure_csv_writer(OUTPUT_CSV)

    try:
        for p in posts:

            for l in [p.list_annotatore_A, p.list_annotatore_B]:
                sentences=split_sentences(l)

                dict=(build_dictionary_fallacies(sentences))

                dict=post_processing_dict(dict)
                logging.info(f"Post ID: {p.post_id}. frase: {p.post_text}")

                for k,v in dict.items():
                    logging.info(f"Fallacia: {k} -> Esempio: {v}")
                logging.info("--------------------------------------------------")

                # scrive le righe per questo post
                for text_val, target in _rows_from_final_dict(dict):
                    writer.writerow([text_val, target])
    finally:
        fcsv.close()

def split_sentences(annotations):
    token_o = "O"
    current_sentence=""
    setences=[]


    for token, label  in annotations:
        #i token con etichetta bio O vengono ignorati
        if label != token_o:
            current_sentence += token + "_TRG_"+  f"[{label}]" + "[SEP]"
        else: 
            if current_sentence!="":
                setences.append(current_sentence.strip())
                current_sentence=""
    
    if current_sentence:
        setences.append(current_sentence.strip())

    return setences

def build_dictionary_fallacies(sentences):
    fallacies_dict={}
    total_dict={}
    i=0
    for s in sentences:
        tokens=s.split("[SEP]")

        try:
            for t in tokens:
                if t.strip()=="":
                    continue
                word,target= t.split("_TRG_")[0], t.split("_TRG_")[1]
                target=target.replace("[","").replace("]","")

                fallacies=target.split("|")

                for f in fallacies:
                    #rimuovo il tag bio, primi due caratteri
                    f=f[2:]
                    if f not in fallacies_dict:
                        fallacies_dict[f]=[word]
                    else:
                        fallacies_dict[f].append(word)

        
            for k, v in fallacies_dict.items():
                total_dict.setdefault(i, {})[k] = v
        

            fallacies_dict={}
            i+=1
        except Exception as e:
            print(f"Errore: {type(e).__name__} - {e!r}")
            continue
        

    return total_dict

def post_processing_dict(dict_fallacies):
    final_dict = {}

    for i, inner_dict in dict_fallacies.items():   # i = indice frase, inner_dict = dizionario fallacie
        for fallacy, words in inner_dict.items():
            # rimuovo duplicati mantenendo l'ordine
            unique_words = list(dict.fromkeys(words))
            
            # accumulo i risultati se la fallacia appare in più frasi
            if fallacy not in final_dict:
                final_dict[fallacy] = []
            
            final_dict[fallacy].append(f"({i}) " + ' '.join(unique_words))

    return final_dict


def build_final_csv(final_dict: Dict[str, Union[str, List[str]]],
                    filepath: str = "final.csv") -> str:
    rows = []
    for target, texts in final_dict.items():
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        for t in texts:
            text = re.sub(r"^\(\d+\)\s*", "", str(t).strip())
            if text:
                rows.append({"text": text, "target": target})

    df = pd.DataFrame(rows).drop_duplicates(['text', 'target'])
    df.to_csv(filepath, index=False)
    return filepath

def clean_duplicates(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].str.strip()
    df["target"] = df["target"].str.strip()

    dedup = df.drop_duplicates(keep="first")
    dedup.to_csv("fallacies_clean.csv", index=False) 

def create_llm_dataset():
    input_path = os.path.join("dati", "fallacies_clean.csv")
    output_path = os.path.join("dati", "fallacies_clean_llm.csv")

    df = pd.read_csv(input_path)

    fallacies = sorted(set(df["target"]))

    llm_data = []

    for _, row in df.iterrows():
        text = row["text"]
        target = row["target"]

        instruction = (
            f"Classifica la seguente porzione di frase utilizzando una delle seguenti fallacie: "
            f"{', '.join(fallacies)}.\n\n"
            f"Testo: {text}"
        )

        llm_data.append({"instruction": instruction, "output": target})

    llm_df = pd.DataFrame(llm_data)

    #stampa visiva del df
    print(llm_df.head())

    llm_df.to_csv(output_path, index=False, encoding="utf-8")

def create_multilabel_dataset():
    input_path = os.path.join("dati", "fallacies_clean.csv")
    output_path = os.path.join("dati", "fallacies_clean_multilabel.csv")

    df = pd.read_csv(input_path)

    # Crea una colonna ID
    df["id"] = [f"{i}" for i in range(len(df))]

    # Converti le categorie in colonne (multi-hot encoding)
    df_onehot = pd.get_dummies(df["target"])

    # Unisci il testo e l'id alle nuove colonne binarie
    df_final = pd.concat([df[["id", "text"]], df_onehot], axis=1)

    # Rinomina le colonne per chiarezza
    df_final.rename(columns={"text": "comment_text"}, inplace=True)

    # Mostra le prime righe
    print(df_final.head())

    # Salva su CSV se vuoi
    df_final.to_csv(output_path, index=False)


if __name__ == "__main__":
    # process()
    # clean_duplicates(OUTPUT_CSV)
    create_multilabel_dataset()
