#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disparity_params_gui.py
ESZA019 - Visao Computacional - Laboratorio 6 (Depth Map)

Sintonia interativa do algoritmo de Block Matching (StereoBM) do OpenCV para
gerar o MAPA DE DISPARIDADE ao vivo, a partir da camera estereo construida no
Lab 5. Baseado na secao "Block Matching For Dense Stereo Correspondence" da
referencia [3] (LearnOpenCV - Depth Perception Using Stereo Camera), ADAPTADO
para ler os parametros de calibracao/retificacao do arquivo params_py.xml.

Como usar:
    python3 disparity_params_gui.py
    - Ajuste as barras deslizantes ate o mapa de disparidade ficar limpo.
    - Tecla 's' -> salva os parametros em depth_estimation_params_py.xml.
    - Tecla 'q' -> sai.

Entradas:
    params_py.xml  (mapas de retificacao Left_/Right_Stereo_Map_x/y do Lab 5)
    duas webcams   (CamL_id, CamR_id)
Saida:
    depth_estimation_params_py.xml  (parametros do StereoBM sintonizados)
"""

import cv2
import numpy as np

# ----------------------------------------------------------------------
# 1) CONFIGURACAO
# ----------------------------------------------------------------------
PARAMS_XML = "params_py.xml"                       # calibracao do Lab 5
SAIDA_XML = "depth_estimation_params_py.xml"       # onde salvar os parametros do BM
CamL_id, CamR_id = 0, 2                             # indices das webcams (Linux: /dev/video*)
FRAME_W, FRAME_H = 640, 480


def abrir_camera(cam_id, w, h):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


# ----------------------------------------------------------------------
# 2) LER OS MAPAS DE RETIFICACAO (params_py.xml do Lab 5)
# ----------------------------------------------------------------------
cv_file = cv2.FileStorage(PARAMS_XML, cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
if Left_Stereo_Map_x is None:
    raise SystemExit(f"{PARAMS_XML} nao encontrado/incompleto. "
                     "Copie o params_py.xml gerado no Lab 5 (calibrate_abc.py) para esta pasta.")

# ----------------------------------------------------------------------
# 3) JANELA COM BARRAS DESLIZANTES (trackbars) PARA O StereoBM
# ----------------------------------------------------------------------
cv2.namedWindow("disparidade", cv2.WINDOW_NORMAL)
cv2.resizeWindow("disparidade", 700, 700)


def nada(x):
    pass


# Os limites seguem as recomendacoes da referencia [3].
cv2.createTrackbar("numDisparities", "disparidade", 1, 17, nada)   # x16
cv2.createTrackbar("blockSize", "disparidade", 5, 50, nada)        # 2x+5 (impar)
cv2.createTrackbar("preFilterType", "disparidade", 1, 1, nada)
cv2.createTrackbar("preFilterSize", "disparidade", 2, 25, nada)    # 2x+5 (impar)
cv2.createTrackbar("preFilterCap", "disparidade", 5, 62, nada)
cv2.createTrackbar("textureThreshold", "disparidade", 10, 100, nada)
cv2.createTrackbar("uniquenessRatio", "disparidade", 15, 100, nada)
cv2.createTrackbar("speckleRange", "disparidade", 0, 100, nada)
cv2.createTrackbar("speckleWindowSize", "disparidade", 3, 25, nada)  # x2
cv2.createTrackbar("disp12MaxDiff", "disparidade", 5, 25, nada)
cv2.createTrackbar("minDisparity", "disparidade", 5, 25, nada)

stereo = cv2.StereoBM_create()

CamL = abrir_camera(CamL_id, FRAME_W, FRAME_H)
CamR = abrir_camera(CamR_id, FRAME_W, FRAME_H)

print("Ajuste as barras. 's' salva os parametros, 'q' sai.")
while True:
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()
    if not (retL and retR):
        continue

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Retificacao (linhas epipolares horizontais).
    Left_nice = cv2.remap(grayL, Left_Stereo_Map_x, Left_Stereo_Map_y,
                          cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(grayR, Right_Stereo_Map_x, Right_Stereo_Map_y,
                           cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Le os valores das barras e converte para os intervalos validos.
    numDisparities = cv2.getTrackbarPos("numDisparities", "disparidade") * 16
    blockSize = cv2.getTrackbarPos("blockSize", "disparidade") * 2 + 5
    preFilterType = cv2.getTrackbarPos("preFilterType", "disparidade")
    preFilterSize = cv2.getTrackbarPos("preFilterSize", "disparidade") * 2 + 5
    preFilterCap = cv2.getTrackbarPos("preFilterCap", "disparidade")
    textureThreshold = cv2.getTrackbarPos("textureThreshold", "disparidade")
    uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", "disparidade")
    speckleRange = cv2.getTrackbarPos("speckleRange", "disparidade")
    speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize", "disparidade") * 2
    disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "disparidade")
    minDisparity = cv2.getTrackbarPos("minDisparity", "disparidade")

    # Aplica os parametros no objeto StereoBM.
    stereo.setNumDisparities(max(16, numDisparities))
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(max(1, preFilterCap))
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # Calcula a disparidade. StereoBM retorna valores em ponto fixo (x16).
    disparity = stereo.compute(Left_nice, Right_nice).astype(np.float32) / 16.0

    # Normaliza para [0,1] apenas para VISUALIZACAO.
    disp_vis = (disparity - minDisparity) / max(1, numDisparities)
    cv2.imshow("disparidade", disp_vis)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    elif tecla == ord("s"):
        fs = cv2.FileStorage(SAIDA_XML, cv2.FILE_STORAGE_WRITE)
        fs.write("numDisparities", numDisparities)
        fs.write("blockSize", blockSize)
        fs.write("preFilterType", preFilterType)
        fs.write("preFilterSize", preFilterSize)
        fs.write("preFilterCap", preFilterCap)
        fs.write("textureThreshold", textureThreshold)
        fs.write("uniquenessRatio", uniquenessRatio)
        fs.write("speckleRange", speckleRange)
        fs.write("speckleWindowSize", speckleWindowSize)
        fs.write("disp12MaxDiff", disp12MaxDiff)
        fs.write("minDisparity", minDisparity)
        fs.release()
        print(f"Parametros salvos em {SAIDA_XML}")

CamL.release()
CamR.release()
cv2.destroyAllWindows()
