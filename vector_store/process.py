from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
import os
import re
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
import uuid
from datapizza.type import Chunk, DenseEmbedding
from datapizza.vectorstores.qdrant import QdrantVectorstore
from openai import OpenAI
from datapizza.core.vectorstore import Distance, VectorConfig
from datapizza.type import EmbeddingFormat
from datapizza.vectorstores.qdrant import QdrantVectorstore
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import sys
import pre_processing
import Annotazione
from pydantic import BaseModel, field_validator
from typing import List
import json
from typing import List, Dict, Any, Tuple
from annotator import genera_annotazione_BIO
from final_output_generator import find_rows_by_post_text,append_row_to_tsv
load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
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

client=OpenAIClient(api_key=api_key, model="text-embedding-3-small")
file_path = os.path.join("fallacies_clean.csv")


vectorstore = QdrantVectorstore(host=os.getenv("QDRANT_ENDPOINT"),
                               api_key=os.getenv("QDRANT_API_KEY"),
                               port=6333)

def CreateCollection(dim):
    vectorstore.create_collection(
        collection_name="documents",
            vector_config = [
        VectorConfig(
        name="text_embeddings",
        dimensions=dim,               
        format=EmbeddingFormat.DENSE,
        distance=Distance.COSINE,
    
    )
])

def emded_openai(texts):
    response = OpenAI().embeddings.create(model="text-embedding-3-small",input=texts)
    return [item.embedding for item in response.data]

def get_features(text):
    prompt=open("features_extraction.txt", "r", encoding="utf-8").read()
    prompt= prompt.replace("{commento_da_analizzare}", f"{text}")

    client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1-nano",  # Default model
    system_prompt=prompt,  # Optional
    temperature=0)  # Optional, controls randomness (0-2) )

    response = client.invoke(prompt)

    return response.text


def build_pipeline(df):
    chunks = []
    texts=[]
    labels1=[]
    labels2=[]
    annotazioni_complete_a=[]
    annotazioni_complete_b=[]
    features=[]
    print(f"Inizio elaborazione e embedding dei testi... len(df): {len(df)}")
    for i, row in enumerate(df.itertuples()):
        # ottieni il vettore di embedding (1D)
        text = light_clean(row.post_text)
        texts.append(text)
        labels1.append(row.labels_a1)
        labels2.append(row.labels_a2)

        annotazionia, annotazionib= crea_annotazioni_singole(row.post_id)
        annotazioni_complete_a.append(annotazionia)
        annotazioni_complete_b.append(annotazionib)
        feature=get_features(text)
        features.append(feature if feature else "")

        print(f"Processed {i+1}/{len(df)} text:{text} features: {feature}")

    embeddings = emded_openai(texts)
    CreateCollection(len(embeddings[0]))


    for text, embedding,label1,label2,annotazionia,annotazionib,feature in zip(texts, embeddings, labels1, labels2,annotazioni_complete_a,annotazioni_complete_b,features):
            chunk = Chunk(
            id=str(uuid.uuid4()),
            text=text,
            metadata={"label_a1": label1, "label_a2": label2,"annotazione_a": annotazionia,"annotazione_b": annotazionib,"features": feature},
            embeddings=[DenseEmbedding("text_embeddings", embedding)]
        )
            chunks.append(chunk)    

    vectorstore.add(chunks, collection_name="documents")


## Carica il prompt di sistema e prepara il client OpenAI
with open("prompt_classificatore.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-5",  # Default model
    system_prompt=system_prompt,  # Optional
    temperature=0,
)

class Prediction(BaseModel):
    text: str
    fallacia: str
    confidenza: float  # meglio numerico 0–1

    @field_validator("confidenza", mode="before")
    def normalize_confidenza(cls, v):
        if isinstance(v, str):
            s = v.strip().replace(",", ".")
            if s.endswith("%"):
                s = s[:-1].strip()
                return float(s) / 100.0
            return float(s)
        return float(v)

class Predictions(BaseModel):
    items: List[Prediction]

import math
from typing import Iterable

def run():
    file_path="train-dev.tsv"
    file_output_subtaskb="results_subtaskb.conll"
    file_output_corretto="output_corretto.tsv"
    file_output_predetto="output_test_GPT5.tsv"
    test_file="test.tsv"
    df = pd.read_csv(test_file, sep="\t", encoding="utf-8")

    #take 140 random row
    test_df = df.sample(n=100)
    train_df = df.drop(test_df.index)

    # build_pipeline(train_df)
    print("Inizio valutazione sul set di test...")
    y_pred = []
    y_true = []
    i=0
    for row in df.itertuples():
        text=row.post_text
        # text=light_clean(text)
        target=row.labels_a1 if pd.notna(row.labels_a1) else ""
        target+="|"+row.labels_a2 if pd.notna(row.labels_a2) else ""
        query_vector = client.embed([text], "text-embedding-3-small")[0]

        results = vectorstore.search(
        query_vector=query_vector,
        collection_name="documents",
        k=3
        )

        if light_clean(text) in [ch.text for ch in results]:
            continue
        
        if results:
            
            prompt=f"{system_prompt}\n\n Esempi di commenti simili con le loro fallacie:\n"

            for ch in results:
                prompt+=f"""- Commento: {ch.text} 
                Fallacia secondo annotatore 1: {ch.metadata['label_a1']}\n 
                Fallacia secondo annotatore 2: {ch.metadata['label_a2']} \n 
                Annotazioni completa 1: {ch.metadata['annotazione_a']}\n 
                Annotazioni completa 2: {ch.metadata['annotazione_b']} 
                Features estratte: {ch.metadata['features']}
                \n\n\n"""

            feature= get_features(text) 
            prompt+=f"Features estratte dal commento da analizzare: {feature}\n"
            prompt+=f"\nCommento da analizzare: {text} \nFallacie:"

            SYSTEM_PROMPT = (
                'Rispondi SOLO con un JSON valido della forma esatta: '
                '{"items":[{"text":"...","fallacia":"...","confidenza":0.0}, ...]}. '
                'Niente testo fuori dal JSON. "confidenza" è un numero tra 0 e 1.'
            )

            processed=False
            while not processed:
                try:
                    response = client.structured_response(
                        input=prompt,
                        output_cls=Predictions,
                        
                    )
                    processed=True
                except Exception as e:
                    print(f"Errore durante l'elaborazione del commento: {e}")
                    import time
                    time.sleep(2)
                    continue

            items = response.structured_data[0].items

            # Costruisci le etichette predette dal risultato strutturato
            predetta = [p.fallacia.replace("-","").replace(" ","").lower() for p in items if getattr(p, "fallacia", None) and p.confidenza >= 70]

            pred_norm = set((predetta))
            corr_norm = set([f.replace("-","").replace(" ","").lower() for f in ((target.split("|")) if isinstance(target, str) else [])])

            intersection = len(pred_norm & corr_norm)
            union = len(pred_norm | corr_norm)
            partial_accuracy = intersection / union if union > 0 else 0

            y_pred.append(list(pred_norm))
            y_true.append(list(corr_norm))

            # annotazione_bio= genera_annotazione_BIO([p for p in items if getattr(p, "fallacia", None) and p.confidenza >= 70], text)

            # with open(file_output_subtaskb, "a", encoding="utf-8") as f_out:
            #     f_out.write(annotazione_bio)
            #     f_out.write("\n\n")

            print(f"\n{i})Suggerimenti: {'\n'.join(['---'+ch.text[:30] for ch in results])}\n Testo: {text}\nPredetto: {pred_norm}\nReale: {corr_norm}\n")

            partial_accuracy = intersection / union if union > 0 else 0

            #recupero la riga nel train-dev.tsv
            # righe = find_rows_by_post_text("train-dev.tsv", text, exact_match=True)
            # try:
            #     if not righe:
            #         print("Nessuna riga trovata\n")
            #         print(text)
            #     else:
            #         riga_corretta = righe[0]
            #         riga_predetta = riga_corretta.copy() 

            #         # se vuoi un ordine deterministico delle etichette:
            #         labels_iter = predetta or []   
            #         lab_pred = "|".join(sorted(set(labels_iter)))

            #         riga_predetta["labels_a1"] = lab_pred
            #         riga_predetta["labels_a2"] = lab_pred

            #         append_row_to_tsv(file_output_corretto, riga_corretta)
            #         append_row_to_tsv(file_output_predetto, riga_predetta)
            #         print("Riga salvata\n")
            # except Exception as e:
            #     print(e)
            #     print(righe)
            #     print(riga_corretta)
            #     print(riga_predetta)
            labels_iter = predetta or []   
            lab_pred = "|".join(sorted(set(labels_iter)))
            riga_predetta = row._asdict()
            riga_predetta["labels_a1"] = lab_pred
            riga_predetta["labels_a2"] = lab_pred
            append_row_to_tsv(file_output_predetto, riga_predetta)
            print(f"Correttezza parziale: {partial_accuracy:.4f}\n------------------------\n")
        i+=1   
    return y_true, y_pred    


def crea_annotazioni_singole(id: int):
    with open("train-dev.conll", "r", encoding="utf-8") as f:
        file=f.read()
    
    res_annotatore_a=""
    res_annotatore_b=""
    posts=(pre_processing.parse_conll(file,id))

    for p in posts:
         for l in [p.list_annotatore_A, p.list_annotatore_B]:
                res=""
                sentences=pre_processing.split_sentences(l)

                dict=(pre_processing.build_dictionary_fallacies(sentences))

                dict=pre_processing.post_processing_dict(dict)

                for k,v in dict.items():

                    res+=(f"Fallacia: {k} -> {v}\n")

                if res_annotatore_a=="":
                    res_annotatore_a=res
                else:
                    res_annotatore_b=res
    
    return res_annotatore_a, res_annotatore_b


if __name__ == "__main__":
    y_true, y_pred = run()

    mlb = MultiLabelBinarizer()
    mlb.fit(y_true + y_pred)

    # 2️⃣ Trasforma in vettori binari
    Y_true = mlb.transform(y_true)
    Y_pred = mlb.transform(y_pred)

    # 3️⃣ Calcola le metriche globali
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="micro", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0
    )

    print(f"Micro Precision: {prec_micro:.4f}")
    print(f"Micro Recall:    {rec_micro:.4f}")
    print(f"Micro F1-Score:  {f1_micro:.4f}")
    print(f"Macro Precision: {prec_macro:.4f}")
    print(f"Macro Recall:    {rec_macro:.4f}")
    print(f"Macro F1-Score:  {f1_macro:.4f}")

    # 4️⃣ (Opzionale) Report più dettagliato per singola fallacia
    print("\n=== Classification report ===")
    print(classification_report(Y_true, Y_pred, target_names=mlb.classes_, zero_division=0))

