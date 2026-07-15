#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
movie3d_abc.py
ESZA019 - Visao Computacional - Laboratorio 5 (Camera Estereo)

Gera AO VIVO uma imagem 3D no formato ANAGLIFO (vermelho/ciano), a partir
das DUAS webcams, usando os parametros de retificacao salvos em
data/params_py.xml pelo calibrate_abc.py.

Para ver o efeito 3D, use os oculos anaglifo:
    lente VERMELHA no olho esquerdo, lente CIANO no olho direito.

Como usar:
    python3 movie3d_abc.py
Tecla q -> sair.

Ideia do anaglifo: cada olho deve ver a imagem da camera correspondente.
Como a lente vermelha deixa passar o canal vermelho e a ciano deixa passar
verde+azul, montamos uma imagem onde:
    canal R  <- imagem da camera ESQUERDA
    canais G,B <- imagem da camera DIREITA
"""

import cv2

# Mesmos indices usados na captura.
CamL_id = 0
CamR_id = 2
FRAME_W, FRAME_H = 640, 480


def abrir_camera(cam_id, w, h):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


# ----------------------------------------------------------------------
# 1) LER OS PARAMETROS DE RETIFICACAO
# ----------------------------------------------------------------------
print("Lendo data/params_py.xml ...")
cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

if Left_Stereo_Map_x is None:
    raise SystemExit("params_py.xml nao encontrado ou incompleto. "
                     "Rode antes: python3 calibrate_abc.py")


# ----------------------------------------------------------------------
# 2) LOOP AO VIVO
# ----------------------------------------------------------------------
CamL = abrir_camera(CamL_id, FRAME_W, FRAME_H)
CamR = abrir_camera(CamR_id, FRAME_W, FRAME_H)

print("Rodando anaglifo ao vivo. Use os oculos vermelho/ciano. q = sair.")
while True:
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()
    if not (retL and retR):
        break

    # Aplica retificacao + correcao de distorcao nas duas imagens.
    Left_nice = cv2.remap(imgL, Left_Stereo_Map_x, Left_Stereo_Map_y,
                          cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(imgR, Right_Stereo_Map_x, Right_Stereo_Map_y,
                           cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Monta o anaglifo. OpenCV usa ordem de canais BGR:
    #   indice 0 = Blue, 1 = Green, 2 = Red.
    # Queremos: R (indice 2) da ESQUERDA; B e G (0 e 1) da DIREITA.
    output = Right_nice.copy()
    output[:, :, 0] = Right_nice[:, :, 0]  # Blue  <- direita
    output[:, :, 1] = Right_nice[:, :, 1]  # Green <- direita
    output[:, :, 2] = Left_nice[:, :, 2]   # Red   <- esquerda

    output = cv2.resize(output, (700, 700))
    cv2.namedWindow("3D movie", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("3D movie", 700, 700)
    cv2.imshow("3D movie", output)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

CamL.release()
CamR.release()
cv2.destroyAllWindows()
