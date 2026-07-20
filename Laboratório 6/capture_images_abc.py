#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_images_abc.py
ESZA019 - Visao Computacional - Laboratorio 5 (Camera Estereo)

Captura pares de imagens (esquerda/direita) do padrao de calibracao
(tabuleiro de xadrez) usando DUAS webcams USB no Linux.

Como usar:
    python3 capture_images_abc.py

Teclas durante a execucao:
    barra de ESPACO -> salva o par atual (L e R)
    q               -> encerra o programa

Os arquivos sao salvos como:
    data/stereoL/img1.png, img2.png, ...
    data/stereoR/img1.png, img2.png, ...

Objetivo: obter de 10 a 15 pares BONS (com o tabuleiro inteiro visivel
nas DUAS cameras ao mesmo tempo, em posicoes/inclinacoes variadas).
"""

import os
import cv2

# ----------------------------------------------------------------------
# 1) PARAMETROS QUE VOCE PROVAVELMENTE VAI PRECISAR AJUSTAR
# ----------------------------------------------------------------------

# Indices das cameras no Linux. Normalmente /dev/video0 e /dev/video1.
# Se abrir a camera errada, troque os numeros (0, 1, 2, ...).
# Dica: rode "ls /dev/video*" ou "v4l2-ctl --list-devices" no terminal.
CamL_id = 0   # webcam da ESQUERDA
CamR_id = 2   # webcam da DIREITA (muitas webcams criam 2 nós; por isso 2)

# Nome do integrante da equipe (o roteiro pede para usar no nome do arquivo).
# Aqui usamos so para organizar; as imagens vao para as pastas stereoL/stereoR.
NOME_EQUIPE = "roberto"

# Dimensoes INTERNAS do tabuleiro = (cantos por linha, cantos por coluna).
# ATENCAO: sao os cantos INTERNOS, nao o numero de quadrados.
# Um tabuleiro com 10x7 quadrados tem 9x6 cantos internos.
CHESSBOARD = (8, 6)

# Resolucao desejada de captura (as duas cameras devem usar a MESMA).
FRAME_W, FRAME_H = 640, 480

# Pasta de saida
pathL = "./data/stereoL/"
pathR = "./data/stereoR/"


# ----------------------------------------------------------------------
# 2) FUNCAO AUXILIAR PARA ABRIR A CAMERA COM O BACKEND CORRETO NO LINUX
# ----------------------------------------------------------------------
def abrir_camera(cam_id, w, h):
    """
    Abre a camera usando o backend V4L2 (padrao do Linux para webcams USB).
    Forcar o backend evita travamentos e leituras lentas.
    """
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    # Define a resolucao. Nem toda webcam aceita qualquer valor;
    # 640x480 e universalmente suportado.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # MJPG costuma permitir FPS maior em USB do que YUYV.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


# ----------------------------------------------------------------------
# 3) PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------
def main():
    os.makedirs(pathL, exist_ok=True)
    os.makedirs(pathR, exist_ok=True)

    CamL = abrir_camera(CamL_id, FRAME_W, FRAME_H)
    CamR = abrir_camera(CamR_id, FRAME_W, FRAME_H)

    if not CamL.isOpened() or not CamR.isOpened():
        print("ERRO: nao consegui abrir uma das cameras.")
        print("Verifique os indices CamL_id/CamR_id e o cabo USB.")
        print("Rode no terminal: v4l2-ctl --list-devices")
        return

    print("Cameras abertas. ESPACO = salvar par | q = sair")
    contador = 1

    while True:
        retL, frameL = CamL.read()
        retR, frameR = CamR.read()

        if not (retL and retR):
            print("Falha ao ler frame de uma das cameras. Tentando de novo...")
            continue

        # Converte para cinza (a deteccao de cantos trabalha em tons de cinza).
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # Procura o tabuleiro nas duas imagens SO PARA DAR FEEDBACK VISUAL.
        # (A deteccao "de verdade" e refeita no calibrate_abc.py.)
        okL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD, None)
        okR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD, None)

        visL, visR = frameL.copy(), frameR.copy()
        if okL:
            cv2.drawChessboardCorners(visL, CHESSBOARD, cornersL, okL)
        if okR:
            cv2.drawChessboardCorners(visR, CHESSBOARD, cornersR, okR)

        # Aviso na tela: so vale a pena salvar quando os DOIS estao verdes.
        status = "OK: pode salvar (ESPACO)" if (okL and okR) else "ajuste o tabuleiro..."
        cor = (0, 255, 0) if (okL and okR) else (0, 0, 255)
        cv2.putText(visL, f"{status}  salvos={contador-1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        # Mostra as duas lado a lado.
        combinado = cv2.hconcat([visL, visR])
        cv2.imshow("ESQUERDA | DIREITA  (ESPACO=salvar, q=sair)", combinado)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("l"):
            if okL and okR:
                cv2.imwrite(pathL + "lucas%d.png" % contador, frameL)
                cv2.imwrite(pathR + "lucas%d.png" % contador, frameR)
                print(f"[{NOME_EQUIPE}] par {contador} salvo.")
                contador += 1
            else:
                print("Tabuleiro nao detectado nas DUAS cameras. Par NAO salvo.")

    CamL.release()
    CamR.release()
    cv2.destroyAllWindows()
    print(f"Fim. Total de pares salvos: {contador-1}")


if __name__ == "__main__":
    main()
