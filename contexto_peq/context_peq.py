#passa pergunta e responsta pra llm 
#llm retorna so o contexto necessario para responder

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL_NAME = "gpt-4.1-mini"

def call_llm_for_focus(question, context):
    if not isinstance(context, str) or not context.strip():
        return ""
    if not isinstance(question, str) or not question.strip():
        return ""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    system_msg = (
        "Você recebe uma pergunta e um contexto longo. "
        "Seu objetivo é extrair apenas o trecho MÍNIMO do contexto necessário para responder essa pergunta. "
        "Copie o texto literalmente do contexto original, sem reescrever, resumir ou explicar. "
        "Se possível, selecione poucas frases ou parágrafos curtos. "
        "Não inclua nada que não ajude diretamente a responder a pergunta."
    )

    user_msg = (
        "Pergunta:\n"
        f"{question}\n\n"
        "Contexto completo:\n"
        f"{context}\n\n"
        "Retorne apenas o trecho mínimo do contexto acima que é necessário para responder a pergunta. "
        "Não explique nada, não acrescente texto seu, apenas copie as partes relevantes do contexto."
    )

    body = {
        "model": CHAT_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=60
    )

    data = response.json()

    if "error" in data and data["error"]:
        raise RuntimeError(f"Erro da API: {data['error']}")

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Resposta inesperada da API: {data}")

    if not isinstance(content, str):
        raise RuntimeError(f"Conteúdo inesperado da API: {data}")

    return content.strip()

def process_csv_llm():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não está definida no ambiente.")

    input_path = "labelstudio_ready_completo_format7.csv"
    output_path = "labelstudio_ready_completo_format7_FOCUS_LLM.csv"

    print("Carregando CSV...")
    df = pd.read_csv(input_path)

    if "question" not in df.columns:
        raise KeyError("A coluna 'question' não existe no CSV.")
    if "right_context" not in df.columns:
        raise KeyError("A coluna 'right_context' não existe no CSV.")

    focuses = []

    print(f"Iniciando extração de contexto focado para {len(df)} linhas...\n")

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="LLM focus",
        ncols=100
    ):
        question = row["question"]
        context = row["right_context"]

        try:
            focus = call_llm_for_focus(question, context)
        except Exception as e:
            focus = ""
            print(f"\nErro na linha {idx}: {e}")

        focuses.append(focus)
        print(f"\n===== Linha {idx} =====")
        print("Trecho selecionado pela LLM:")
        print(focus)

        time.sleep(0.1)

    df["right_context_focus_llm"] = focuses
    df.to_csv(output_path, index=False)
    print("\nFINALIZADO!")
    print("Arquivo salvo em:", output_path)

if __name__ == "__main__":
    process_csv_llm()
