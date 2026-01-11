from datapizza.clients.openai import OpenAIClient
import os
import pandas as pd
from dotenv import load_dotenv
from datapizza.tracing import ContextTracing
import json

load_dotenv()

data_prompt = ""
with open("prompt_dataAug.txt", "r", encoding="utf-8") as f:
    data_prompt = f.read()

client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-5-nano",  # Default model
    system_prompt=data_prompt,  # Optional
    temperature=0,  # Optional, controls randomness (0-2) 
)

def retrieve_example(fallacy: str):
        file_path="train-dev.tsv"

        df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
        examples = []
        for row in df.itertuples():
              conatins=fallacy in str(row.labels_a1).split("|") or fallacy in str(row.labels_a2).split("|")

              if conatins:
                    examples.append(row.post_text)

        return examples

def copy_in_tsv(response_text: str, fallacy: str):
    json_output = json.loads(response_text)

    df_new = pd.DataFrame(json_output)
    df_new = df_new.rename(columns={"text": "post_text"})
    df_new["post_id"] = range(10000, 10000 + len(df_new))  # nuovi ID univoci
    df_new["post_date"] = "2025-10"
    df_new["post_topic_keywords"] = "general"
    df_new["labels_a1"] = fallacy
    df_new["labels_a2"] = fallacy

    df_new = df_new[["post_id", "post_date", "post_topic_keywords", "post_text", "labels_a1", "labels_a2"]]
    df_old = pd.read_csv("train-dev.tsv", sep="\t")
    df_final = pd.concat([df_old, df_new], ignore_index=True)
    df_final.to_csv("train-dev.tsv", sep="\t", index=False, encoding="utf-8")

    return df_new

if __name__ == "__main__":
    fallacy="Slippery-slope"
    examples = retrieve_example(fallacy)

    prompt=f"""Fallacia target: {fallacy}
                    Esempi da cui prendere spunto:
                {['- ' + ex + '\n\n' for ex in examples[:10]]}


                Restituisci il json con 50 esempi"""
    
    with ContextTracing().trace("trace_name"):
        response = client.invoke(prompt)

    if response and response.text:
         #convert in json
        json_output = response.text
        
        try:
            copy_in_tsv(json_output,fallacy)              
        except Exception as e:
            print(f"Errore nel parsing del JSON: {e}")

    print(response.text)
    
