#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disparity2depth_calib.py
ESZA019 - Visao Computacional - Laboratorio 6 (Depth Map)

Do MAPA DE DISPARIDADE ao MAPA DE PROFUNDIDADE. Baseado na secao
"From disparity map to depth map" da referencia [3] (LearnOpenCV), ADAPTADO
para ler params_py.xml (Lab 5) e depth_estimation_params_py.xml (parametros do
StereoBM sintonizados no disparity_params_gui.py).

Modelo fisico: a profundidade Z e inversamente proporcional a disparidade d,
    Z = f * B / d,
que ajustamos empiricamente na forma
    Z = M * (1 / d) + C,
onde M e C sao obtidos por MINIMOS QUADRADOS a partir de amostras (d, Z_real)
coletadas colocando um objeto a distancias CONHECIDAS.

Como usar:
    python3 disparity2depth_calib.py
    - Posicione o objeto a uma distancia real conhecida.
    - Clique com o mouse sobre o objeto no mapa de disparidade para amostrar 'd'.
    - Digite no terminal a distancia real (em cm) daquela amostra.
    - Repita para >= 5 distancias diferentes.
    - Tecla 'q' -> encerra, ajusta M e C, salva no xml e plota Z vs d.

Saidas:
    depth_estimation_params_py.xml  (atualizado com M e C)
    profundidade_vs_disparidade.png (grafico exigido pelo roteiro)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

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


# 1) Mapas de retificacao (Lab 5)
fs = cv2.FileStorage(PARAMS_XML, cv2.FILE_STORAGE_READ)
Lx = fs.getNode("Left_Stereo_Map_x").mat(); Ly = fs.getNode("Left_Stereo_Map_y").mat()
Rx = fs.getNode("Right_Stereo_Map_x").mat(); Ry = fs.getNode("Right_Stereo_Map_y").mat()
fs.release()

# 2) Parametros do StereoBM (sintonizados no disparity_params_gui.py)
fs = cv2.FileStorage(DEPTH_XML, cv2.FILE_STORAGE_READ)
def geti(k, d):
    n = fs.getNode(k); return int(n.real()) if not n.empty() else d
numDisparities = geti("numDisparities", 128); blockSize = geti("blockSize", 15)
stereo = cv2.StereoBM_create(numDisparities=max(16, numDisparities), blockSize=blockSize)
stereo.setPreFilterType(geti("preFilterType", 1)); stereo.setPreFilterSize(geti("preFilterSize", 5))
stereo.setPreFilterCap(geti("preFilterCap", 31)); stereo.setTextureThreshold(geti("textureThreshold", 10))
stereo.setUniquenessRatio(geti("uniquenessRatio", 15)); stereo.setSpeckleRange(geti("speckleRange", 0))
stereo.setSpeckleWindowSize(geti("speckleWindowSize", 0)); stereo.setDisp12MaxDiff(geti("disp12MaxDiff", 5))
stereo.setMinDisparity(geti("minDisparity", 5))
fs.release()

# Amostras coletadas: listas de disparidade media e distancia real (cm).
amostras_disp, amostras_dist = [], []
disparity_atual = None


def ao_clicar(event, x, y, flags, param):
    """Ao clicar, mede a disparidade media numa pequena janela ao redor do ponto."""
    global disparity_atual
    if event == cv2.EVENT_LBUTTONDOWN and disparity_atual is not None:
        jan = disparity_atual[max(0, y-3):y+3, max(0, x-3):x+3]
        d = np.mean(jan[jan > 0]) if np.any(jan > 0) else 0.0
        if d <= 0:
            print("  disparidade invalida nesse ponto, tente outro."); return
        try:
            z = float(input(f"  disparidade={d:.2f}. Distancia REAL do objeto (cm)? "))
        except ValueError:
            print("  valor invalido, amostra ignorada."); return
        amostras_disp.append(d); amostras_dist.append(z)
        print(f"  amostra {len(amostras_disp)} registrada: d={d:.2f}, Z={z:.1f} cm")


cv2.namedWindow("disparidade")
cv2.setMouseCallback("disparidade", ao_clicar)
CamL = abrir_camera(CamL_id, FRAME_W, FRAME_H)
CamR = abrir_camera(CamR_id, FRAME_W, FRAME_H)

print("Clique no objeto para amostrar; informe a distancia real. 'q' encerra e calibra.")
while True:
    retL, imgL = CamL.read(); retR, imgR = CamR.read()
    if not (retL and retR):
        continue
    gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY); gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    Ln = cv2.remap(gL, Lx, Ly, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Rn = cv2.remap(gR, Rx, Ry, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    disparity_atual = stereo.compute(Ln, Rn).astype(np.float32) / 16.0
    vis = cv2.normalize(disparity_atual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("disparidade", cv2.applyColorMap(vis, cv2.COLORMAP_JET))
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

CamL.release(); CamR.release(); cv2.destroyAllWindows()

# 3) Ajuste por minimos quadrados: Z = M*(1/d) + C
if len(amostras_disp) < 2:
    raise SystemExit("Poucas amostras (>=2 necessarias). Repita a coleta.")
d = np.array(amostras_disp, np.float32)
z = np.array(amostras_dist, np.float32)
A = np.vstack([1.0 / d, np.ones_like(d)]).T          # colunas: [1/d , 1]
(M, C), *_ = np.linalg.lstsq(A, z, rcond=None)
print(f"\nModelo ajustado:  Z = {M:.2f} * (1/d) + {C:.2f}   (Z em cm)")

# 4) Salvar M e C no xml (atualiza o arquivo existente)
fs = cv2.FileStorage(DEPTH_XML, cv2.FILE_STORAGE_WRITE)
for k, v in [("numDisparities", numDisparities), ("blockSize", blockSize)]:
    fs.write(k, v)
fs.write("M", float(M)); fs.write("C", float(C))
fs.release()
print(f"M e C salvos em {DEPTH_XML}")

# 5) Grafico Profundidade vs Disparidade (exigido pelo roteiro)
dd = np.linspace(max(d.min(), 1), d.max(), 100)
plt.figure(figsize=(6, 4))
plt.scatter(d, z, c="red", label="amostras medidas")
plt.plot(dd, M*(1.0/dd)+C, "b-", label=f"Z = {M:.1f}/d + {C:.1f}")
plt.xlabel("Disparidade (px)"); plt.ylabel("Profundidade Z (cm)")
plt.title("Profundidade vs Disparidade"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig("profundidade_vs_disparidade.png", dpi=120)
print("Grafico salvo: profundidade_vs_disparidade.png")
