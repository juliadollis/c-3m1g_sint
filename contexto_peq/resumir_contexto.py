# passa contexto pro gpt 4 e pede pra resumir
import os
import pandas as pd
import requests
import time
from tqdm import tqdm

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"

def call_llm_for_summary(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    prompt = (
        "Resuma o texto a seguir de forma direta e objetiva, em poucas linhas, reduzindo o tamanho mas mantendo o sentido central. "
        "Remova detalhes extensos, repetições, listas longas e partes muito específicas. "
        "Não explique, não interprete e não adicione informações; apenas produza uma versão condensada e fiel do texto original.\n\n"
        f"{text}"
    )
    body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "Você resume textos de forma objetiva, fiel ao conteúdo e sem interpretações."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 256
    }
    response = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=headers, json=body, timeout=60)
    data = response.json()

    if "error" in data:
        raise RuntimeError(f"Erro da API: {data['error'].get('message', data['error'])}")

    return data["choices"][0]["message"]["content"].strip()

def resumir_csv():
    input_path = "labelstudio_ready_completo_format7.csv"
    output_path = "labelstudio_ready_completo_format7_RESUMO_gpt4.csv"

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não está definida no ambiente.")

    print("Carregando CSV...")
    df = pd.read_csv(input_path)

    if "right_context" not in df.columns:
        raise KeyError("A coluna 'right_context' não foi encontrada no CSV.")

    summaries = []

    print(f"Iniciando resumo de {len(df)} entradas...\n")

    for idx, text in tqdm(enumerate(df["right_context"]), total=len(df),
                          desc="Resumindo", ncols=100):
        try:
            resumo = call_llm_for_summary(text)
        except Exception as e:
            resumo = ""
            print(f"\nErro na linha {idx}: {e}")
        summaries.append(resumo)
        print(f"\n===== Linha {idx} =====")
        print("Saída do modelo:")
        print(resumo)
        time.sleep(0.2)

    df["right_context_summary"] = summaries

    df.to_csv(output_path, index=False)

    print("\nResumo concluído.")
    print("Arquivo salvo em:", output_path)

resumir_csv()
