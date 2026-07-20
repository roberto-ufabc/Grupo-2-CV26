#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disparidade_offline.py
ESZA019 - Visao Computacional - Laboratorio 6 (Depth Map)

Demonstracao do MAPA DE DISPARIDADE em imagens JA GRAVADAS (nao precisa de
webcams). Util para validar os algoritmos e gerar figuras para o relatorio.
Reproduz os exemplos lab6a_.py (StereoSGBM) e lab6b_.py (StereoBM).

Como usar:
    python3 disparidade_offline.py
    (rode a partir da pasta que contem as imagens de exemplo, ou ajuste PASTA)

Saidas:
    demo_disparidade_sgbm.png         (par im0/im1, StereoSGBM)
    demo_disparidade_bm_tsukuba.png   (par tsukuba, StereoBM)
"""

import cv2
import numpy as np

PASTA = "../lab6_exemplo/"      # onde estao im0.png, im1.png, tsukuba_l/r.png


def salvar_colorido(disp, nome):
    vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(nome, cv2.applyColorMap(vis, cv2.COLORMAP_JET))
    print("salvo:", nome)


# 1) StereoSGBM no par im0/im1 (reproduz lab6a_.py)
L = cv2.imread(PASTA + "im0.png", cv2.IMREAD_GRAYSCALE)
R = cv2.imread(PASTA + "im1.png", cv2.IMREAD_GRAYSCALE)
L = cv2.resize(L, (720, 492)); R = cv2.resize(R, (720, 492))
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=8,
                             disp12MaxDiff=1, uniquenessRatio=10,
                             speckleWindowSize=10, speckleRange=8)
salvar_colorido(sgbm.compute(L, R).astype(np.float32), "demo_disparidade_sgbm.png")

# 2) StereoBM no par tsukuba (reproduz lab6b_.py)
tl = cv2.imread(PASTA + "tsukuba_l.png", cv2.IMREAD_GRAYSCALE)
tr = cv2.imread(PASTA + "tsukuba_r.png", cv2.IMREAD_GRAYSCALE)
bm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
salvar_colorido(bm.compute(tl, tr).astype(np.float32), "demo_disparidade_bm_tsukuba.png")
