#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obstacle_avoidance.py
ESZA019 - Visao Computacional - Laboratorio 6 (Depth Map)

Sistema simples de deteccao/medida de distancia de OBSTACULOS a partir do mapa
de profundidade. Baseado na secao "Obstacle avoidance system" da referencia [3]
(LearnOpenCV), ADAPTADO para ler params_py.xml (Lab 5) e
depth_estimation_params_py.xml (parametros do StereoBM + M e C da calibracao
disparidade->profundidade).

Converte disparidade em profundidade com  Z = M*(1/d) + C  e destaca a regiao
mais proxima da cena, exibindo sua distancia estimada.

Como usar:
    python3 obstacle_avoidance.py
    - 'q' encerra.

Entradas:
    params_py.xml
    depth_estimation_params_py.xml  (com numDisparities, blockSize, M, C)
"""

import cv2
import numpy as np

PARAMS_XML = "params_py.xml"
DEPTH_XML = "depth_estimation_params_py.xml"
CamL_id, CamR_id = 0, 2
FRAME_W, FRAME_H = 640, 480


def abrir_camera(cam_id, w, h):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


# 1) Mapas de retificacao
fs = cv2.FileStorage(PARAMS_XML, cv2.FILE_STORAGE_READ)
Lx = fs.getNode("Left_Stereo_Map_x").mat(); Ly = fs.getNode("Left_Stereo_Map_y").mat()
Rx = fs.getNode("Right_Stereo_Map_x").mat(); Ry = fs.getNode("Right_Stereo_Map_y").mat()
fs.release()

# 2) Parametros do BM + coeficientes M, C
fs = cv2.FileStorage(DEPTH_XML, cv2.FILE_STORAGE_READ)
def getf(k, d):
    n = fs.getNode(k); return float(n.real()) if not n.empty() else d
numDisparities = int(getf("numDisparities", 128)); blockSize = int(getf("blockSize", 15))
M = getf("M", 1000.0); C = getf("C", 0.0)
fs.release()

stereo = cv2.StereoBM_create(numDisparities=max(16, numDisparities), blockSize=blockSize)

CamL = abrir_camera(CamL_id, FRAME_W, FRAME_H)
CamR = abrir_camera(CamR_id, FRAME_W, FRAME_H)

print("Deteccao de obstaculo mais proximo. 'q' encerra.")
while True:
    retL, imgL = CamL.read(); retR, imgR = CamR.read()
    if not (retL and retR):
        continue
    gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY); gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    Ln = cv2.remap(gL, Lx, Ly, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Rn = cv2.remap(gR, Rx, Ry, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    disparity = stereo.compute(Ln, Rn).astype(np.float32) / 16.0

    # Converte para profundidade (cm) onde a disparidade e valida (d > 0).
    valido = disparity > 0
    depth = np.zeros_like(disparity)
    depth[valido] = M * (1.0 / disparity[valido]) + C

    # Obstaculo = regiao mais proxima (menor profundidade positiva).
    prox = np.where((depth > 0), depth, np.inf)
    if np.isfinite(prox.min()):
        z_min = float(prox.min())
        mask = (prox < z_min * 1.15).astype(np.uint8) * 255      # regiao ~mais proxima
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        saida = cv2.cvtColor(Ln, cv2.COLOR_GRAY2BGR)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(saida, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(saida, f"{z_min:.0f} cm", (x, max(20, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if z_min < 50:
                cv2.putText(saida, "OBSTACULO PROXIMO!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("obstaculo", saida)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

CamL.release(); CamR.release(); cv2.destroyAllWindows()
