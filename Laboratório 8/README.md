# Laboratório 8 — Rastreamento de Objetos (ESZA019)

Equipe **Ctrl+C, Ctrl+V e Fé**. Rastreamento de objetos com OpenCV (seleção manual de ROI),
em vídeo e ao vivo pela webcam, com gravação dos vídeos resultantes.

## Ambiente (Linux)

```bash
cd "Laboratório 8"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`opencv-contrib-python` é obrigatório — os rastreadores CSRT/KCF/MOSSE/GOTURN estão no módulo `contrib`.

## Experimento 1 — rastreamento em vídeo (arquivo)

```bash
python3 rastreamento_video.py --video meu_video.mp4 --metodo CSRT --saida saida_rastreada.mp4
```

Use os vídeos da equipe (inclusive os do trabalho de vídeo). Desenhe a ROI no 1º quadro e confirme
com ENTER. O vídeo anotado é salvo em `--saida`.

## Experimento 2 — rastreamento ao vivo (webcam)

```bash
python3 rastreamento_webcam.py --camera 0 --metodo CSRT --saida webcam_rastreada.mp4
```

Tecle `s` para selecionar o objeto, `r` para re-selecionar, `q` para sair. Mostra a janela ao vivo
e grava o vídeo resultante.

## Métodos disponíveis

`CSRT` (mais preciso), `KCF` (equilibrado), `MOSSE` (mais rápido), `MIL`, `GOTURN` (deep learning).

## GOTURN (opcional, deep learning)

Para usar `--metodo GOTURN` são necessários dois arquivos **nesta mesma pasta**:

- `goturn.prototxt`
- `goturn.caffemodel` — baixe do link indicado no roteiro:
  https://drive.google.com/file/d/1nz0SPrvUXpw8G9DCEDR131kWZ_WiUUyo/view?usp=sharing

> O `goturn.caffemodel` (~350 MB) não pôde ser baixado automaticamente (link do Google Drive).
> Baixe-o manualmente e salve aqui. O `goturn.prototxt` acompanha o repositório do OpenCV
> (`samples/data`) ou o projeto GOTURN.

## Saídas para o relatório

Grave os vídeos resultantes (`saida_rastreada.mp4`, `webcam_rastreada.mp4`) e prints, e referencie-os
no `Relatório.ipynb`.
