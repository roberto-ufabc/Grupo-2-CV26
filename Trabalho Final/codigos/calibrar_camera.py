#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrar_camera.py
ESZA019 - Visao Computacional - Trabalho Final (Deteccao de Uso de EPI)

Calibracao da camera do sistema (Requisito A, OBRIGATORIO): extrai os parametros
intrinsecos (K) e os coeficientes de distorcao (dist), demonstra o ERRO DE
REPROJECAO e salva tudo em calib_camera.xml, usado por deteccao_epi.py para
corrigir a distorcao dos quadros.

Reaproveita o metodo validado nos Laboratorios 4 e 5.

Dois modos de uso:
    # 1) capturar imagens do tabuleiro pela webcam ('s' salva, 'c' calibra, 'q' sai)
    python3 calibrar_camera.py --capturar --camera 0

    # 2) calibrar a partir de imagens ja gravadas numa pasta
    python3 calibrar_camera.py --imagens "calib_imgs/*.jpg"

Parametros do tabuleiro: --cols e --rows sao os CANTOS INTERNOS (nao os quadrados).
"""

import argparse
import glob
import cv2
import numpy as np

SAIDA_XML = "calib_camera.xml"
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def capturar(camera_id, pasta="calib_imgs"):
    """Captura imagens do tabuleiro pela webcam para posterior calibracao."""
    import os
    os.makedirs(pasta, exist_ok=True)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise SystemExit("Nao consegui abrir a camera.")
    i = 0
    print("'s' salva o quadro | 'q' encerra a captura")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("captura (s=salva, q=sai)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            nome = f"{pasta}/calib_{i:02d}.jpg"
            cv2.imwrite(nome, frame); i += 1
            print("salvo:", nome)
        elif k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()
    return f"{pasta}/*.jpg"


def calibrar(padrao_imgs, cols, rows, mostrar=True):
    """Calibra a partir de um glob de imagens do tabuleiro (cols x rows cantos internos)."""
    # Pontos 3D do padrao (Z=0, tabuleiro plano).
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    gray = None
    arquivos = sorted(glob.glob(padrao_imgs))
    if not arquivos:
        raise SystemExit(f"Nenhuma imagem encontrada em: {padrao_imgs}")

    for fn in arquivos:
        img = cv2.imread(fn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            if mostrar:
                cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
                cv2.imshow("cantos", img); cv2.waitKey(150)
        else:
            print("  tabuleiro NAO detectado em", fn)
    if mostrar:
        cv2.destroyAllWindows()

    if len(objpoints) < 5:
        raise SystemExit(f"Poucas imagens validas ({len(objpoints)}). Capture mais.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    # Erro de reprojecao (demonstracao exigida pelo Requisito A).
    erro_total = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        erro_total += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    erro_medio = erro_total / len(objpoints)

    print("\n=== RESULTADO DA CALIBRACAO ===")
    print("Imagens validas:", len(objpoints))
    print("Matriz intrinseca K:\n", K)
    print("Distorcao (k1,k2,p1,p2,k3):", dist.ravel())
    print(f"Erro de reprojecao (RMS): {erro_medio:.4f} px")

    fs = cv2.FileStorage(SAIDA_XML, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K); fs.write("dist", dist)
    fs.write("erro_reprojecao", float(erro_medio))
    fs.write("num_imagens", len(objpoints))
    fs.release()
    print("Parametros salvos em", SAIDA_XML)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calibracao de camera (Trabalho Final EPI).")
    ap.add_argument("--capturar", action="store_true", help="capturar imagens pela webcam")
    ap.add_argument("--camera", type=int, default=0, help="indice da webcam")
    ap.add_argument("--imagens", default="calib_imgs/*.jpg", help="glob das imagens do tabuleiro")
    ap.add_argument("--cols", type=int, default=8, help="cantos internos por linha")
    ap.add_argument("--rows", type=int, default=6, help="cantos internos por coluna")
    args = ap.parse_args()

    padrao = capturar(args.camera) if args.capturar else args.imagens
    calibrar(padrao, args.cols, args.rows)
