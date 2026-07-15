#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
movie3d_abc_gravacao.py
ESZA019 - Visao Computacional - Laboratorio 5 (Camera Estereo)

Igual ao movie3d_abc.py, mas GRAVA um video 3D anaglifo de ~10 a 20 s.
O video e salvo em data/anaglifo.avi e, ao final, convertido para .mp4
usando ffmpeg (se disponivel no sistema).

Como usar:
    python3 movie3d_abc_gravacao.py
Teclas: q -> encerra antes do tempo.

Requisito do roteiro (item D): gravar 10-20 s e converter para mp4.
"""

import os
import time
import subprocess
import cv2

CamL_id = 0
CamR_id = 2
FRAME_W, FRAME_H = 640, 480

DURACAO_S = 15          # duracao alvo da gravacao (dentro da faixa 10-20 s)
FPS = 20                # quadros por segundo do arquivo gravado
SAIDA_AVI = "data/anaglifo.avi"
SAIDA_MP4 = "data/anaglifo.mp4"
TAM_SAIDA = (700, 700)  # resolucao do video gravado


def abrir_camera(cam_id, w, h):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


# 1) Ler parametros de retificacao
cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
if Left_Stereo_Map_x is None:
    raise SystemExit("params_py.xml ausente. Rode antes: python3 calibrate_abc.py")

# 2) Abrir cameras e o gravador de video
CamL = abrir_camera(CamL_id, FRAME_W, FRAME_H)
CamR = abrir_camera(CamR_id, FRAME_W, FRAME_H)

# MJPG dentro de um .avi e amplamente compativel; depois convertemos p/ mp4.
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(SAIDA_AVI, fourcc, FPS, TAM_SAIDA)

print(f"Gravando ~{DURACAO_S}s de anaglifo em {SAIDA_AVI} ... (q p/ parar)")
t0 = time.time()
while True:
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()
    if not (retL and retR):
        break

    Left_nice = cv2.remap(imgL, Left_Stereo_Map_x, Left_Stereo_Map_y,
                          cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(imgR, Right_Stereo_Map_x, Right_Stereo_Map_y,
                           cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    output = Right_nice.copy()
    output[:, :, 2] = Left_nice[:, :, 2]   # canal vermelho vem da esquerda
    output = cv2.resize(output, TAM_SAIDA)

    writer.write(output)                   # grava o frame no arquivo
    cv2.imshow("Gravando 3D (q p/ parar)", output)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break
    if time.time() - t0 >= DURACAO_S:
        break

CamL.release()
CamR.release()
writer.release()
cv2.destroyAllWindows()
print(f"Video AVI salvo: {SAIDA_AVI}")

# 3) Converter para mp4 com ffmpeg (se instalado)
if os.path.exists(SAIDA_AVI):
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", SAIDA_AVI,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", SAIDA_MP4],
            check=True)
        print(f"Convertido para mp4: {SAIDA_MP4}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg nao encontrado. Instale com: sudo apt install ffmpeg")
        print("Ou converta manualmente:  ffmpeg -i data/anaglifo.avi data/anaglifo.mp4")
