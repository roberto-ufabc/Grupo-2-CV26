#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
epi_utils.py
ESZA019 - Visao Computacional - Trabalho Final (Deteccao de Uso de EPI)

Funcoes utilitarias compartilhadas pelos demais programas do sistema:
    - carregar a calibracao da camera e corrigir a distorcao dos quadros;
    - associar cada EPI (capacete/colete) a pessoa correspondente;
    - decidir a conformidade (CONFORME / NAO CONFORME);
    - desenhar o resultado sobre o quadro.

Nao executa nada sozinho; e importado por deteccao_epi.py e avaliar_metricas.py.
"""

import cv2
import numpy as np

# Nomes das classes do detector (mesma ordem do data.yaml).
CLASSES = {0: "pessoa", 1: "capacete", 2: "colete"}


# ----------------------------------------------------------------------
# CALIBRACAO (Requisito A do trabalho)
# ----------------------------------------------------------------------
def carregar_calibracao(caminho_xml):
    """Le K (matriz intrinseca) e dist (coeficientes de distorcao) de um .xml.

    Entrada: caminho do arquivo gerado por calibrar_camera.py.
    Saida:   (K, dist) ou (None, None) se o arquivo nao existir/estiver incompleto.
    """
    fs = cv2.FileStorage(caminho_xml, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    fs.release()
    return K, dist


def corrigir_distorcao(frame, K, dist):
    """Aplica cv2.undistort ao quadro, se houver calibracao; senao devolve o quadro."""
    if K is None or dist is None:
        return frame
    h, w = frame.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, K, dist, None, newK)


# ----------------------------------------------------------------------
# ASSOCIACAO ESPACIAL EPI <-> PESSOA
# ----------------------------------------------------------------------
def iou(a, b):
    """Interseccao sobre uniao entre duas caixas no formato (x1, y1, x2, y2)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    uniao = area_a + area_b - inter
    return inter / uniao if uniao > 0 else 0.0


def centro_dentro(epi, pessoa):
    """True se o CENTRO da caixa do EPI cai dentro da caixa da pessoa."""
    cx = (epi[0] + epi[2]) / 2.0
    cy = (epi[1] + epi[3]) / 2.0
    return pessoa[0] <= cx <= pessoa[2] and pessoa[1] <= cy <= pessoa[3]


def associar(pessoas, capacetes, coletes, iou_min=0.02):
    """Para cada pessoa, verifica se ha capacete e colete associados.

    Um EPI pertence a pessoa se seu centro esta dentro da caixa dela OU se ha
    sobreposicao (IoU) minima. Retorna lista de dicts com a caixa da pessoa e
    os booleanos de presenca de cada EPI.
    """
    resultados = []
    for p in pessoas:
        tem_cap = any(centro_dentro(c, p) or iou(c, p) > iou_min for c in capacetes)
        tem_col = any(centro_dentro(v, p) or iou(v, p) > iou_min for v in coletes)
        resultados.append({"box": p, "capacete": tem_cap, "colete": tem_col,
                           "conforme": tem_cap and tem_col})
    return resultados


# ----------------------------------------------------------------------
# DECISAO TEMPORAL (estabiliza a saida ao longo de N quadros)
# ----------------------------------------------------------------------
class DecisorTemporal:
    """Confirma o estado de cada pessoa (por id de rastreamento) apenas quando
    ele se repete em N quadros consecutivos, evitando que o rotulo 'pisque'."""

    def __init__(self, n_confirmacao=5):
        self.n = n_confirmacao
        self.historico = {}   # id -> lista dos ultimos estados (bool conforme)

    def atualizar(self, track_id, conforme):
        h = self.historico.setdefault(track_id, [])
        h.append(conforme)
        if len(h) > self.n:
            h.pop(0)
        # So confirma quando os ultimos N quadros concordam; senao mantem o mais frequente.
        if len(h) == self.n and len(set(h)) == 1:
            return h[0]
        return sum(h) >= len(h) / 2.0


# ----------------------------------------------------------------------
# VISUALIZACAO
# ----------------------------------------------------------------------
def desenhar(frame, resultados, fps=None, zona=None):
    """Desenha as caixas das pessoas, o rotulo de EPI e o status global."""
    algum_nao_conforme = False
    for r in resultados:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        cor = (0, 200, 0) if r["conforme"] else (0, 0, 255)
        if not r["conforme"]:
            algum_nao_conforme = True
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cap = "capacete OK" if r["capacete"] else "sem capacete"
        col = "colete OK" if r["colete"] else "sem colete"
        cv2.putText(frame, f"{cap} | {col}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)

    status = "NAO CONFORME" if algum_nao_conforme else "CONFORME"
    cor_status = (0, 0, 255) if algum_nao_conforme else (0, 200, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), cor_status, -1)
    cv2.putText(frame, f"STATUS: {status}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if fps is not None:
        cv2.putText(frame, f"{fps:.1f} FPS", (frame.shape[1] - 120, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
