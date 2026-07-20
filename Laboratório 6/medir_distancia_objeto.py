#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medir_distancia_objeto.py
ESZA019 - Visao Computacional - Laboratorio 6 (Depth Map) - item (vi)

Programa COMPLETO que fornece a distancia da camera estereo a um OBJETO
ESPECIFICO. Basta clicar sobre o objeto: o programa retifica o par, calcula a
disparidade (StereoBM), converte em profundidade (Z = M*(1/d) + C) e exibe a
distancia estimada em cm.

Pensado para o tema do Trabalho Final da equipe (Deteccao de Uso de EPI):
o "objeto" pode ser a PESSOA ou o CAPACETE, permitindo checar se ela esta
dentro da zona monitorada (ex.: 1,5 a 3 m) antes de avaliar o EPI.

Como usar:
    python3 medir_distancia_objeto.py
    - Clique sobre o objeto -> a distancia aparece na tela e no terminal.
    - 'q' encerra.

Entradas:
    params_py.xml
    depth_estimation_params_py.xml  (numDisparities, blockSize, M, C)
"""

import cv2
import numpy as np

PARAMS_XML = "params_py.xml"
DEPTH_XML = "depth_estimation_params_py.xml"
CamL_id, CamR_id = 0, 2
FRAME_W, FRAME_H = 640, 480
ZONA_MIN_CM, ZONA_MAX_CM = 150, 300      # zona monitorada (tema EPI)


def abrir_camera(cam_id, w, h):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


# 1) Parametros de calibracao/retificacao e do BM
fs = cv2.FileStorage(PARAMS_XML, cv2.FILE_STORAGE_READ)
Lx = fs.getNode("Left_Stereo_Map_x").mat(); Ly = fs.getNode("Left_Stereo_Map_y").mat()
Rx = fs.getNode("Right_Stereo_Map_x").mat(); Ry = fs.getNode("Right_Stereo_Map_y").mat()
fs.release()

fs = cv2.FileStorage(DEPTH_XML, cv2.FILE_STORAGE_READ)
def getf(k, d):
    n = fs.getNode(k); return float(n.real()) if not n.empty() else d
numDisparities = int(getf("numDisparities", 128)); blockSize = int(getf("blockSize", 15))
M = getf("M", 1000.0); C = getf("C", 0.0)
fs.release()

stereo = cv2.StereoBM_create(numDisparities=max(16, numDisparities), blockSize=blockSize)

estado = {"disp": None, "texto": ""}


def ao_clicar(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and estado["disp"] is not None:
        jan = estado["disp"][max(0, y-3):y+3, max(0, x-3):x+3]
        d = np.mean(jan[jan > 0]) if np.any(jan > 0) else 0.0
        if d <= 0:
            estado["texto"] = "sem disparidade nesse ponto"; return
        z = M * (1.0 / d) + C
        dentro = ZONA_MIN_CM <= z <= ZONA_MAX_CM
        estado["texto"] = f"Distancia: {z:.0f} cm " + ("(na zona)" if dentro else "(fora da zona)")
        print(estado["texto"])


cv2.namedWindow("medir distancia")
cv2.setMouseCallback("medir distancia", ao_clicar)
CamL = abrir_camera(CamL_id, FRAME_W, FRAME_H)
CamR = abrir_camera(CamR_id, FRAME_W, FRAME_H)

print("Clique no objeto para medir a distancia. 'q' encerra.")
while True:
    retL, imgL = CamL.read(); retR, imgR = CamR.read()
    if not (retL and retR):
        continue
    gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY); gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    Ln = cv2.remap(gL, Lx, Ly, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Rn = cv2.remap(gR, Rx, Ry, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    estado["disp"] = stereo.compute(Ln, Rn).astype(np.float32) / 16.0

    saida = cv2.cvtColor(Ln, cv2.COLOR_GRAY2BGR)
    if estado["texto"]:
        cv2.putText(saida, estado["texto"], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("medir distancia", saida)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

CamL.release(); CamR.release(); cv2.destroyAllWindows()
