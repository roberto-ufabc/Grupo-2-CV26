#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rastreamento_webcam.py
ESZA019 - Visao Computacional - Laboratorio 8 (Rastreamento de Objetos)
Equipe Ctrl+C, Ctrl+V e Fe

Experimento (2) do roteiro: modifica o (1) para ler a imagem da WEBCAM ao vivo.
Mostra uma janela ao vivo com a imagem e o resultado do rastreamento, e SALVA o
video resultante.

Uso (Linux):
    python3 rastreamento_webcam.py --camera 0 --metodo CSRT --saida webcam_rastreada.mp4

Fluxo:
  1) A webcam abre em pre-visualizacao; tecle 's' para congelar e SELECIONAR a ROI
     (desenhe o retangulo e ENTER/ESPACO). Tecle 'q' para sair a qualquer momento.
  2) Apos a selecao, o rastreamento roda ao vivo e o video vai sendo gravado.
  3) Tecle 'r' para re-selecionar outro objeto; 'q' para encerrar.
"""
import argparse
import cv2


def criar_tracker(nome):
    nome = nome.upper()
    fabricas = {
        "CSRT": ["TrackerCSRT_create"],
        "KCF": ["TrackerKCF_create"],
        "MOSSE": ["legacy.TrackerMOSSE_create", "TrackerMOSSE_create"],
        "MIL": ["TrackerMIL_create"],
        "GOTURN": ["TrackerGOTURN_create"],
    }
    for caminho in fabricas.get(nome, []) + [f"legacy.Tracker{nome}_create"]:
        try:
            obj = cv2
            for parte in caminho.split("."):
                obj = getattr(obj, parte)
            return obj()
        except AttributeError:
            continue
    raise RuntimeError(f"Tracker {nome} indisponivel (instale opencv-contrib-python).")


def abrir_camera(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)   # backend do Linux
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="indice da webcam")
    ap.add_argument("--metodo", default="CSRT")
    ap.add_argument("--saida", default="webcam_rastreada.mp4")
    args = ap.parse_args()

    cap = abrir_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit("Nao consegui abrir a webcam. Confira o indice --camera.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.saida, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))

    tracker = None
    print("'s' seleciona a ROI | 'r' re-seleciona | 'q' sai")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if tracker is not None:
            achou, bbox = tracker.update(frame)
            if achou:
                x, y, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame, "Rastreando", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "PERDIDO", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Tecle 's' para selecionar o objeto", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)                      # grava sempre (com ou sem tracking)
        cv2.imshow("Webcam - rastreamento (s/r/q)", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break
        elif tecla in (ord("s"), ord("r")):
            # Congela o quadro atual para selecionar a ROI.
            bbox = cv2.selectROI("Selecione o objeto", frame, showCrosshair=True)
            cv2.destroyWindow("Selecione o objeto")
            if bbox != (0, 0, 0, 0):
                tracker = criar_tracker(args.metodo)
                tracker.init(frame, bbox)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video salvo em: {args.saida}")


if __name__ == "__main__":
    main()
