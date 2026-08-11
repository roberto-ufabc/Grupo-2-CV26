#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
treinar_modelo.py
ESZA019 - Visao Computacional - Trabalho Final (Deteccao de Uso de EPI)

Treina/adapta um detector CNN de estagio unico (YOLOv8, via Ultralytics) para as
classes do problema: pessoa, capacete e colete. Parte de um modelo pre-treinado
(transfer learning) e faz o fine-tuning no dataset descrito em data.yaml.

Como usar:
    python3 treinar_modelo.py --data data.yaml --epocas 100 --imgsz 640

Saida:
    runs/detect/train/weights/best.pt  (pesos do melhor modelo)
    -> copie/renomeie para epi_yolo.pt para usar no deteccao_epi.py

Observacao: o treino se beneficia de GPU. Sem GPU, reduza --epocas/--imgsz ou use
um modelo menor (yolov8n). A inferencia em tempo real roda em CPU com yolov8n.
"""

import argparse


def main():
    ap = argparse.ArgumentParser(description="Treino do detector de EPI (YOLOv8).")
    ap.add_argument("--data", default="data.yaml", help="arquivo de configuracao do dataset")
    ap.add_argument("--base", default="yolov8n.pt", help="modelo pre-treinado de partida")
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--saida", default="epi_yolo.pt", help="nome do arquivo de pesos final")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics nao instalado. Rode: pip install ultralytics")

    # 1) Modelo pre-treinado (transfer learning a partir do COCO).
    modelo = YOLO(args.base)

    # 2) Fine-tuning nas classes de EPI descritas em data.yaml.
    modelo.train(data=args.data, epochs=args.epocas, imgsz=args.imgsz, batch=args.batch)

    # 3) Avaliacao no conjunto de validacao (mAP, precisao, revocacao).
    metricas = modelo.val()
    print("mAP@0.5:", getattr(metricas.box, "map50", "n/d"))
    print("mAP@0.5:0.95:", getattr(metricas.box, "map", "n/d"))

    # 4) Salva os pesos finais com um nome amigavel.
    modelo.save(args.saida)
    print(f"Pesos salvos em {args.saida} (e em runs/detect/train/weights/best.pt)")


if __name__ == "__main__":
    main()
