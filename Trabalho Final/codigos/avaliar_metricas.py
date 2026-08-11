#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
avaliar_metricas.py
ESZA019 - Visao Computacional - Trabalho Final (Deteccao de Uso de EPI)

Calcula as metricas objetivas exigidas no item 4 do manual do Trabalho Final.
Dois modos:

  # 1) Metricas de DETECCAO (mAP, precisao, revocacao) no conjunto de validacao,
  #    usando o proprio YOLO. Requer o modelo treinado e o data.yaml.
  python3 avaliar_metricas.py --modo mapa --modelo epi_yolo.pt --data data.yaml

  # 2) Metricas de CLASSIFICACAO da DECISAO de conformidade (matriz de confusao,
  #    precisao, revocacao, F1) a partir de um CSV anotado manualmente.
  python3 avaliar_metricas.py --modo confusao --csv anotacoes.csv

Formato do CSV (uma linha por quadro/pessoa avaliada; cabecalho obrigatorio):
    real,predito
    conforme,conforme
    nao_conforme,conforme
    ...
(valores aceitos: 'conforme' e 'nao_conforme')

Saidas (modo confusao):
    matriz_confusao.png  e  metricas impressas no terminal.
"""

import argparse
import csv


CLASSES = ["conforme", "nao_conforme"]


def metricas_confusao(reais, preditos):
    """Monta a matriz de confusao 2x2 e calcula P/R/F1 para a classe de seguranca
    ('nao_conforme' = positivo, pois deixar passar quem esta sem EPI e o pior erro)."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    M = [[0, 0], [0, 0]]   # linhas = real, colunas = predito
    for r, p in zip(reais, preditos):
        M[idx[r]][idx[p]] += 1

    # Positivo = 'nao_conforme' (indice 1).
    VP = M[1][1]; FN = M[1][0]; FP = M[0][1]; VN = M[0][0]
    precisao = VP / (VP + FP) if (VP + FP) else 0.0
    revocacao = VP / (VP + FN) if (VP + FN) else 0.0
    f1 = (2 * precisao * revocacao / (precisao + revocacao)
          if (precisao + revocacao) else 0.0)
    acuracia = (VP + VN) / max(1, (VP + VN + FP + FN))
    return M, precisao, revocacao, f1, acuracia


def salvar_matriz(M, caminho="matriz_confusao.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(4.5, 4))
    arr = np.array(M)
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusao — decisao de EPI")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                    color="black", fontsize=14)
    fig.tight_layout(); fig.savefig(caminho, dpi=120)
    print("Matriz salva em", caminho)


def modo_confusao(csv_path):
    reais, preditos = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r, p = row["real"].strip(), row["predito"].strip()
            if r in CLASSES and p in CLASSES:
                reais.append(r); preditos.append(p)
    if not reais:
        raise SystemExit("CSV vazio ou invalido. Use colunas 'real' e 'predito'.")
    M, prec, rev, f1, acc = metricas_confusao(reais, preditos)
    print("\n=== Matriz de confusao (linhas=real, colunas=predito) ===")
    print(f"{'':14s}{'conforme':>14s}{'nao_conforme':>16s}")
    for i, c in enumerate(CLASSES):
        print(f"{c:14s}{M[i][0]:14d}{M[i][1]:16d}")
    print(f"\nClasse positiva = 'nao_conforme' (prioridade de seguranca)")
    print(f"Precisao : {prec:.3f}")
    print(f"Revocacao: {rev:.3f}")
    print(f"F1-Score : {f1:.3f}")
    print(f"Acuracia : {acc:.3f}  (n={len(reais)} amostras)")
    salvar_matriz(M)


def modo_mapa(modelo_path, data_path):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics nao instalado. Rode: pip install ultralytics")
    modelo = YOLO(modelo_path)
    m = modelo.val(data=data_path)
    print("\n=== Metricas de deteccao (conjunto de validacao) ===")
    print("mAP@0.5      :", getattr(m.box, "map50", "n/d"))
    print("mAP@0.5:0.95 :", getattr(m.box, "map", "n/d"))
    print("Precisao (media):", getattr(m.box, "mp", "n/d"))
    print("Revocacao (media):", getattr(m.box, "mr", "n/d"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Metricas do Trabalho Final (EPI).")
    ap.add_argument("--modo", choices=["mapa", "confusao"], required=True)
    ap.add_argument("--modelo", default="epi_yolo.pt")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--csv", default="anotacoes.csv")
    args = ap.parse_args()
    if args.modo == "mapa":
        modo_mapa(args.modelo, args.data)
    else:
        modo_confusao(args.csv)
