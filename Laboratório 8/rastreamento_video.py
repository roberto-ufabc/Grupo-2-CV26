#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rastreamento_video.py
ESZA019 - Visao Computacional - Laboratorio 8 (Rastreamento de Objetos)
Equipe Ctrl+C, Ctrl+V e Fe

Experimento (1) do roteiro: le um VIDEO em arquivo (ex.: um video da equipe ou
do trabalho de video), permite SELECIONAR MANUALMENTE a ROI do objeto, rastreia
o objeto com um dos algoritmos do OpenCV e SALVA o video resultante com o
rastreamento desenhado.

Uso:
    python3 rastreamento_video.py --video entrada.mp4 --metodo CSRT --saida saida_rastreada.mp4

Metodos suportados: CSRT (preciso), KCF (rapido), MOSSE (muito rapido),
                    MIL, GOTURN (deep learning; requer os arquivos do modelo).

Ao rodar:
  1) o primeiro quadro e mostrado; desenhe um retangulo em volta do objeto com o
     mouse e tecle ENTER/ESPACO para confirmar (C para cancelar);
  2) o rastreamento roda ate o fim do video; tecle 'q' para encerrar antes.
"""
import argparse
import cv2


# ----------------------------------------------------------------------
# Cria o rastreador pedido, lidando com as diferentes versoes do OpenCV.
# Em versoes recentes os trackers "classicos" ficam em cv2.legacy.
# ----------------------------------------------------------------------
def criar_tracker(nome):
    nome = nome.upper()
    fabricas = {
        "CSRT":  ["TrackerCSRT_create"],
        "KCF":   ["TrackerKCF_create"],
        "MOSSE": ["legacy.TrackerMOSSE_create", "TrackerMOSSE_create"],
        "MIL":   ["TrackerMIL_create"],
        "BOOSTING": ["legacy.TrackerBoosting_create"],
        "MEDIANFLOW": ["legacy.TrackerMedianFlow_create"],
        "TLD":   ["legacy.TrackerTLD_create"],
        "GOTURN": ["TrackerGOTURN_create"],
    }
    if nome not in fabricas:
        raise ValueError(f"Metodo '{nome}' desconhecido. Use: {list(fabricas)}")

    ultimo_erro = None
    for caminho in fabricas[nome] + [f"legacy.Tracker{nome}_create"]:
        try:
            obj = cv2
            for parte in caminho.split("."):
                obj = getattr(obj, parte)
            return obj()
        except AttributeError as e:
            ultimo_erro = e
    raise RuntimeError(
        f"Nao foi possivel criar o tracker {nome}. "
        f"Instale opencv-contrib-python. Detalhe: {ultimo_erro}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="caminho do video de entrada")
    ap.add_argument("--metodo", default="CSRT", help="CSRT|KCF|MOSSE|MIL|GOTURN...")
    ap.add_argument("--saida", default="saida_rastreada.mp4", help="video de saida")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Nao consegui abrir o video: {args.video}")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Video vazio ou ilegivel.")

    # 1) Selecao MANUAL da ROI no primeiro quadro.
    print("Desenhe a ROI com o mouse e tecle ENTER/ESPACO (C cancela).")
    bbox = cv2.selectROI("Selecione o objeto", frame, showCrosshair=True)
    cv2.destroyWindow("Selecione o objeto")
    if bbox == (0, 0, 0, 0):
        raise SystemExit("Nenhuma ROI selecionada.")

    # 2) Inicializa o rastreador com a ROI escolhida.
    tracker = criar_tracker(args.metodo)
    tracker.init(frame, bbox)

    # Gravador do video de saida (mesma resolucao/FPS do original).
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.saida, fourcc, fps, (w, h))

    print(f"Rastreando com {args.metodo.upper()}... (q para sair)")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = cv2.getTickCount()
        achou, bbox = tracker.update(frame)      # atualiza a posicao do objeto
        fps_inst = cv2.getTickFrequency() / (cv2.getTickCount() - t0)

        if achou:
            x, y, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, "Rastreando", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "PERDIDO", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, f"{args.metodo.upper()}  {fps_inst:.0f} FPS", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        out.write(frame)                          # salva o quadro anotado
        cv2.imshow("Rastreamento (q=sair)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
        n += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Concluido. {n} quadros. Video salvo em: {args.saida}")


if __name__ == "__main__":
    main()
