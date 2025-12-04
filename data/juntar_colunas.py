import pandas as pd

input_path = "labelstudio_ready_completo_format7_RESUMOpequeno.csv"
output_path = "LABEL_labelstudio_ready_completo_format7_RESUMOpequeno.csv"

df = pd.read_csv(input_path)

if "right_context_summary" not in df.columns:
    raise KeyError("A coluna 'right_context_summary' não existe no CSV.")
if "completa_format" not in df.columns:
    raise KeyError("A coluna 'completa_format' não existe no CSV.")

def juntar_resumo_contexto(row):
    resumo = "" if pd.isna(row["right_context_summary"]) else str(row["right_context_summary"])
    contexto = "" if pd.isna(row["completa_format"]) else str(row["completa_format"])
    if resumo and contexto:
        return "Resumo:\n" + resumo + "\n\n" + contexto
    elif resumo:
        return "Resumo:\n" + resumo
    else:
        return contexto

df["completa_format"] = df.apply(juntar_resumo_contexto, axis=1)

df.to_csv(output_path, index=False)

print("Arquivo salvo em:", output_path)