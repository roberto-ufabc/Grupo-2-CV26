# Trabalho Final — Detecção de Uso de EPI em Ambiente Simples (códigos)

Equipe **Ctrl+C, Ctrl+V e Fé** — ESZA019 Visão Computacional 2026.2

Sistema de visão computacional em tempo real que verifica o uso de EPI (capacete e colete de alta
visibilidade) por uma pessoa, com **calibração de câmera** (obrigatória) e um **detector CNN** (YOLOv8).
Os códigos seguem a Modelagem Funcional da Etapa 3.

## Arquivos

| Arquivo | Função |
|---------|--------|
| `requirements.txt` | Dependências (`opencv-contrib-python`, `numpy`, `matplotlib`, `ultralytics`). |
| `data.yaml` | Configuração do dataset para treino (classes: pessoa, capacete, colete). |
| `calibrar_camera.py` | **Requisito A** — calibra a câmera, mostra o erro de reprojeção, salva `calib_camera.xml`. |
| `treinar_modelo.py` | Treina/adapta o YOLOv8 no dataset → `epi_yolo.pt`. |
| `epi_utils.py` | Utilitários: calibração, associação EPI↔pessoa, decisão temporal, desenho. |
| `deteccao_epi.py` | **Programa principal** em tempo real (aquisição→undistort→CNN→associação→decisão→saída). |
| `avaliar_metricas.py` | Métricas: mAP (detecção) e matriz de confusão + P/R/F1 (decisão). |

## Fluxo de execução

```
calibrar_camera.py  → calib_camera.xml ─┐
                                        ├→ deteccao_epi.py  (tempo real)
treinar_modelo.py   → epi_yolo.pt ──────┘
                                         → avaliar_metricas.py (validação)
```

## Passo a passo

### 0. Instalar dependências
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 1. Calibrar a câmera (Requisito A — obrigatório)
```bash
# capturar imagens do tabuleiro (tecla 's' salva, 'q' encerra) e calibrar
python3 calibrar_camera.py --capturar --camera 0 --cols 8 --rows 6
# ou, se já tiver imagens numa pasta:
python3 calibrar_camera.py --imagens "calib_imgs/*.jpg" --cols 8 --rows 6
```
Gera `calib_camera.xml` (K, dist e o **erro de reprojeção** — reporte-o no relatório).
Também é possível reaproveitar a calibração do Lab 4/5.

### 2. Preparar o dataset e treinar o detector
- Anote imagens (Roboflow/CVAT/LabelImg) nas 3 classes e exporte em formato YOLO.
- Ajuste os caminhos em `data.yaml`.
```bash
python3 treinar_modelo.py --data data.yaml --epocas 100 --imgsz 640
```
Gera `epi_yolo.pt`. (Com GPU o treino é bem mais rápido; sem GPU, reduza épocas/imgsz.)

### 3. Rodar a detecção em tempo real
```bash
python3 deteccao_epi.py --modelo epi_yolo.pt --camera 0 --calib calib_camera.xml
# 'q' encerra | 'g' liga/desliga a gravação (sessao_epi.mp4)
```
A tela mostra a caixa de cada pessoa, os rótulos de capacete/colete, o **status global**
(verde = CONFORME, vermelho = NÃO CONFORME) e os **FPS**.

> Se `epi_yolo.pt` ainda não existir, o programa usa `yolov8n.pt` (só detecta "person" do COCO)
> apenas para demonstrar o pipeline — os rótulos de EPI exigem o modelo treinado no passo 2.

### 4. Avaliar as métricas (item 4 do manual)
```bash
# mAP / precisão / revocação do detector no conjunto de validação
python3 avaliar_metricas.py --modo mapa --modelo epi_yolo.pt --data data.yaml

# matriz de confusão + P/R/F1 da decisão de conformidade, a partir de um CSV anotado
python3 avaliar_metricas.py --modo confusao --csv anotacoes.csv
```
O CSV tem cabeçalho `real,predito` com valores `conforme`/`nao_conforme`. Gera `matriz_confusao.png`.
A **latência/FPS** é lida diretamente na tela do `deteccao_epi.py`.

### 5. Testes com voluntários (Etapas 5–7)
Grave (celular) os voluntários usando o sistema nas quatro situações (com tudo / sem capacete /
sem colete / sem nada), anote o *ground truth* em `anotacoes.csv` e rode o passo 4 para obter as
métricas. Registre também o feedback (dificuldades, erros, sugestões).

## Observações

- Ambiente-alvo "simples": uma pessoa por vez, iluminação controlada, fundo neutro, câmera fixa.
- Prioriza-se **alta revocação** para a classe `nao_conforme` (é preferível um alarme falso a deixar
  passar alguém sem EPI) — critério para ajustar o limiar `--conf`.
- A câmera deve permanecer **fixa** após a calibração.
