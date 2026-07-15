#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_abc.py
ESZA019 - Visao Computacional - Laboratorio 5 (Camera Estereo)

Calibracao ESTEREO a partir dos pares de imagens capturados por
capture_images_abc.py. Baseado no calibrate.py da learnopencv, adaptado
para (a) ler automaticamente todos os pares existentes e (b) SALVAR TODOS
os parametros no arquivo params_py.xml (nao so os mapas de retificacao),
porque o Laboratorio 6 vai precisar da matriz Q e das intrinsecas.

Como usar:
    python3 calibrate_abc.py

Saidas:
    data/params_py.xml   -> todos os parametros de calibracao/retificacao
    (janelas mostrando os cantos detectados em cada par)
"""

import glob
import numpy as np
import cv2

# ----------------------------------------------------------------------
# 1) PARAMETROS
# ----------------------------------------------------------------------
pathL = "./data/stereoL/"
pathR = "./data/stereoR/"

# Cantos internos do tabuleiro (o MESMO valor usado na captura).
CHESSBOARD = (8, 6)

# Tamanho REAL de cada quadrado do tabuleiro impresso, em milimetros.
# Meca com uma regua! Esse valor da ESCALA METRICA a calibracao,
# ou seja, e o que permite depois medir distancias em mm/cm no Lab 6.
SQUARE_SIZE_MM = 25.0

# Criterio de parada para refinar a posicao dos cantos (subpixel).
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ----------------------------------------------------------------------
# 2) PONTOS 3D DO PADRAO (o tabuleiro no seu proprio sistema de coordenadas)
# ----------------------------------------------------------------------
# Criamos as coordenadas (X, Y, Z=0) de cada canto interno do tabuleiro.
# Z=0 porque o tabuleiro e plano. Multiplicamos por SQUARE_SIZE_MM para
# que fique em milimetros (escala real).
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

img_ptsL = []   # cantos 2D detectados na camera esquerda
img_ptsR = []   # cantos 2D detectados na camera direita
obj_pts = []    # cantos 3D correspondentes (o mesmo objp para cada par)


# ----------------------------------------------------------------------
# 3) DETECCAO DOS CANTOS EM TODOS OS PARES
# ----------------------------------------------------------------------
# Descobre automaticamente quantos pares existem.
arquivosL = sorted(glob.glob(pathL + "img*.png"))
n_pares = len(arquivosL)
print(f"Encontrados {n_pares} pares de imagens. Detectando cantos...\n")

imgL_gray = None
usados = 0
for i in range(1, n_pares + 1):
    imgL = cv2.imread(pathL + "lucas%d.png" % i)
    imgR = cv2.imread(pathR + "lucas%d.png" % i)
    if imgL is None or imgR is None:
        continue

    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(imgL_gray, CHESSBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(imgR_gray, CHESSBOARD, None)

    # So usamos o par se o tabuleiro foi encontrado nas DUAS imagens.
    if retL and retR:
        obj_pts.append(objp)
        # Refina a posicao dos cantos com precisao subpixel (melhora muito).
        cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)
        usados += 1

        # Feedback visual (aperte uma tecla para passar para o proximo par).
        outL = imgL.copy(); outR = imgR.copy()
        cv2.drawChessboardCorners(outL, CHESSBOARD, cornersL, retL)
        cv2.drawChessboardCorners(outR, CHESSBOARD, cornersR, retR)
        cv2.imshow("cantos L | R", cv2.hconcat([outL, outR]))
        cv2.waitKey(300)  # 300 ms por par; troque por 0 para pausar em cada um.
    else:
        print(f"  par {i}: tabuleiro NAO detectado nas duas imagens (descartado)")

cv2.destroyAllWindows()
print(f"\nPares realmente usados na calibracao: {usados}")
if usados < 8:
    print("AVISO: poucos pares validos. Recomenda-se >= 10 para bom resultado.")

image_size = imgL_gray.shape[::-1]  # (largura, altura)


# ----------------------------------------------------------------------
# 4) CALIBRACAO INDIVIDUAL (MONO) DE CADA CAMERA
# ----------------------------------------------------------------------
# Descobrimos a matriz intrinseca (foco fx,fy e centro cx,cy) e os
# coeficientes de distorcao de cada camera separadamente.
print("\nCalibrando camera ESQUERDA...")
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    obj_pts, img_ptsL, image_size, None, None)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

print("Calibrando camera DIREITA...")
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    obj_pts, img_ptsR, image_size, None, None)
hR, wR = imgR_gray.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

print(f"  Erro de reprojecao (RMS) esquerda: {retL:.4f} px")
print(f"  Erro de reprojecao (RMS) direita : {retR:.4f} px")


# ----------------------------------------------------------------------
# 5) CALIBRACAO ESTEREO (relacao geometrica entre as duas cameras)
# ----------------------------------------------------------------------
# CALIB_FIX_INTRINSIC: mantemos as intrinsecas ja obtidas e calculamos
# apenas a Rotacao (R) e Translacao (T) entre as cameras, alem das
# matrizes Essencial (E) e Fundamental (F).
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print("\nCalibracao estereo (stereoCalibrate)...")
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
    obj_pts, img_ptsL, img_ptsR,
    new_mtxL, distL, new_mtxR, distR,
    image_size, criteria_stereo, flags)
print(f"  Erro de reprojecao estereo (RMS): {retS:.4f} px")

# A baseline (distancia entre as cameras) e o modulo do vetor de translacao.
baseline_mm = float(np.linalg.norm(Trns))
print(f"  Baseline estimada (|T|): {baseline_mm:.1f} mm")


# ----------------------------------------------------------------------
# 6) RETIFICACAO ESTEREO
# ----------------------------------------------------------------------
# Alinha as duas imagens de modo que as linhas epipolares fiquem
# HORIZONTAIS e coincidentes. Assim, pontos correspondentes ficam na
# mesma linha -> a busca de correspondencia (Lab 6) vira 1D.
rectify_scale = 1  # 0 = corta a imagem; 1 = mantem tudo (com bordas pretas)
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(
    new_mtxL, distL, new_mtxR, distR,
    image_size, Rot, Trns, rectify_scale, (0, 0))

# Mapas que transformam cada pixel da imagem original na imagem
# retificada+sem distorcao. Usados com cv2.remap().
Left_Stereo_Map = cv2.initUndistortRectifyMap(
    new_mtxL, distL, rect_l, proj_mat_l, image_size, cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(
    new_mtxR, distR, rect_r, proj_mat_r, image_size, cv2.CV_16SC2)


# ----------------------------------------------------------------------
# 7) SALVAR TODOS OS PARAMETROS EM params_py.xml
# ----------------------------------------------------------------------
# O exemplo original salvava SO os mapas. Aqui salvamos tambem as
# intrinsecas, R, T, E, F e Q, porque o Lab 6 (mapa de profundidade)
# precisa deles (em especial da matriz Q e da baseline).
print("\nSalvando parametros em data/params_py.xml ...")
cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x",  Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y",  Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.write("mtxL", mtxL)
cv_file.write("distL", distL)
cv_file.write("mtxR", mtxR)
cv_file.write("distR", distR)
cv_file.write("R", Rot)
cv_file.write("T", Trns)
cv_file.write("E", Emat)
cv_file.write("F", Fmat)
cv_file.write("Q", Q)
cv_file.write("baseline_mm", baseline_mm)
cv_file.write("image_width", image_size[0])
cv_file.write("image_height", image_size[1])
cv_file.release()


# ----------------------------------------------------------------------
# 8) IMPRIMIR OS PARAMETROS (para copiar no relatorio, item C do roteiro)
# ----------------------------------------------------------------------
np.set_printoptions(precision=3, suppress=True)
print("\n================ PARAMETROS OBTIDOS ================")
print("Matriz intrinseca ESQUERDA (mtxL):\n", mtxL)
print("Distorcao ESQUERDA (distL):\n", distL.ravel())
print("Matriz intrinseca DIREITA (mtxR):\n", mtxR)
print("Distorcao DIREITA (distR):\n", distR.ravel())
print("Rotacao entre cameras (R):\n", Rot)
print("Translacao entre cameras (T) [mm]:\n", Trns.ravel())
print("Matriz Essencial (E):\n", Emat)
print("Matriz Fundamental (F):\n", Fmat)
print("Matriz de reprojecao (Q):\n", Q)
print(f"Baseline |T| = {baseline_mm:.1f} mm")
print("====================================================")
print("Pronto. Use data/params_py.xml no movie3d_abc.py e no Lab 6.")
