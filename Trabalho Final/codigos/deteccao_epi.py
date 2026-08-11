#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deteccao_epi.py
ESZA019 - Visao Computacional - Trabalho Final (Deteccao de Uso de EPI)

Programa PRINCIPAL, em TEMPO REAL. Implementa o pipeline definido na Modelagem
Funcional (Etapa 3):

    (1) aquisicao do quadro da webcam
    (2) correcao de distorcao (calibracao, Requisito A)
    (3) inferencia CNN + rastreamento (YOLOv8 / Ultralytics)  -> pessoa/capacete/colete
    (4) pos-processamento: limiar de confianca + NMS (feitos pelo YOLO)
    (5) associacao espacial EPI <-> pessoa (epi_utils.associar)
    (6) decisao temporal por pessoa (epi_utils.DecisorTemporal)
    (7) saida: caixas, rotulos, status global, alerta e FPS

Como usar:
    python3 deteccao_epi.py --modelo epi_yolo.pt --camera 0 --calib calib_camera.xml
    Teclas: 'q' encerra | 'g' liga/desliga a gravacao do video da sessao.

Entradas:
    epi_yolo.pt       (pesos treinados em treinar_modelo.py; classes pessoa/capacete/colete)
    calib_camera.xml  (calibracao gerada em calibrar_camera.py) - opcional
Saida:
    janela em tempo real; opcionalmente sessao_epi.mp4 (tecla 'g').
"""

import argparse
import time
import cv2
import numpy as np

import epi_utils as U


def main():
    ap = argparse.ArgumentParser(description="Deteccao de uso de EPI em tempo real (YOLOv8).")
    ap.add_argument("--modelo", default="epi_yolo.pt", help="pesos do detector de EPI")
    ap.add_argument("--camera", type=int, default=0, help="indice da webcam")
    ap.add_argument("--calib", default="calib_camera.xml", help="calibracao da camera (opcional)")
    ap.add_argument("--conf", type=float, default=0.35, help="limiar de confianca")
    ap.add_argument("--n_confirmacao", type=int, default=5, help="quadros p/ confirmar a decisao")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics nao instalado. Rode: pip install ultralytics")

    # Modelo CNN. Se o modelo proprio nao existir, avisa (pode-se usar yolov8n so p/ 'pessoa').
    import os
    if not os.path.exists(args.modelo):
        print(f"[aviso] '{args.modelo}' nao encontrado. Treine com treinar_modelo.py.")
        print("        Usando 'yolov8n.pt' (detecta apenas 'person' do COCO) como demonstracao.")
        modelo = YOLO("yolov8n.pt")
    else:
        modelo = YOLO(args.modelo)
    nomes = modelo.names   # dict id->nome das classes do modelo carregado

    # Calibracao (Requisito A). Se ausente, segue sem corrigir distorcao.
    K, dist = U.carregar_calibracao(args.calib)
    if K is None:
        print(f"[aviso] calibracao '{args.calib}' ausente; seguindo sem corrigir distorcao.")

    decisor = U.DecisorTemporal(n_confirmacao=args.n_confirmacao)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Nao consegui abrir a camera.")

    gravando = False
    writer = None
    t_ant = time.time()
    fps = 0.0
    print("Rodando. 'q' encerra | 'g' liga/desliga gravacao.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # (2) Correcao de distorcao.
        frame = U.corrigir_distorcao(frame, K, dist)

        # (3)+(4) Inferencia CNN + rastreamento (ByteTrack embutido) + NMS/conf.
        res = modelo.track(frame, persist=True, conf=args.conf, verbose=False)[0]

        # Separa as deteccoes por classe.
        pessoas, capacetes, coletes, ids = [], [], [], []
        if res.boxes is not None and res.boxes.xyxy is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            tid = (res.boxes.id.cpu().numpy().astype(int)
                   if res.boxes.id is not None else np.arange(len(xyxy)))
            for box, c, i in zip(xyxy, cls, tid):
                nome = nomes.get(int(c), str(c)).lower()
                # Classes 'NO-...' (ex.: NO-Hardhat, NO-Safety Vest) indicam AUSENCIA -> ignoradas
                # aqui; a ausencia e inferida por nao haver capacete/colete associado a pessoa.
                if nome.startswith("no"):
                    continue
                # Casamento por conteudo do nome (funciona com pessoa/person, capacete/helmet/hardhat,
                # colete/vest/safety vest, etc., em PT ou EN).
                if "person" in nome or "pessoa" in nome:
                    pessoas.append(box); ids.append(i)
                elif "hardhat" in nome or "helmet" in nome or "capacete" in nome:
                    capacetes.append(box)
                elif "vest" in nome or "colete" in nome:
                    coletes.append(box)

        # (5) Associacao EPI <-> pessoa.
        resultados = U.associar(pessoas, capacetes, coletes)

        # (6) Decisao temporal por id de rastreamento.
        for r, i in zip(resultados, ids):
            r["conforme"] = decisor.atualizar(int(i), r["conforme"])

        # FPS (media exponencial simples).
        agora = time.time()
        dt = agora - t_ant; t_ant = agora
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # (7) Saida.
        frame = U.desenhar(frame, resultados, fps=fps)
        cv2.imshow("Deteccao de EPI", frame)

        if gravando and writer is not None:
            writer.write(frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("g"):
            gravando = not gravando
            if gravando and writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter("sessao_epi.mp4",
                                         cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            print("gravacao:", "ligada" if gravando else "pausada")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
