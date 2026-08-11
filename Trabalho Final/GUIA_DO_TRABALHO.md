# Guia do Trabalho Final — Detecção de Uso de EPI em Ambiente Simples

Equipe **Ctrl+C, Ctrl+V e Fé** — ESZA019 Visão Computacional 2026.2

Este guia tem duas partes: **(A)** o passo a passo de cada etapa do trabalho (conforme o manual) e
**(B)** o que cada código em `codigos/` faz. A trilha técnica escolhida é **Deep Learning (CNN / YOLOv8)**
para detectar *pessoa*, *capacete* e *colete* e decidir a conformidade em tempo real, com **calibração de
câmera** obrigatória.

---

## Parte A — Passo a passo por etapa

### Mapa geral (etapas × entregáveis)

| Etapa | Período | O que fazer | Entregável associado | Situação |
|-------|---------|-------------|----------------------|----------|
| 1 | sem. 2–3 | Entrevistas empáticas | (compõe o Tema) | concluída |
| 2 | sem. 3–4 | Contexto e cenário de aplicação | (1) Tema do Trabalho | concluída |
| 3 | sem. 5–6 | Modelagem funcional geral | esboço (Jupyter) | **entregue** (`Modelagem Funcional Geral.ipynb`) |
| 4 | sem. 7–9 | Desenvolvimento (hardware/software) | (compõe o Relatório Técnico) | a fazer |
| 5 | sem. 10 | Roteiro de teste com voluntários | roteiro de testes | a fazer |
| 6 | sem. 11 | Realização dos testes | filmagens + dados | a fazer |
| 7 | sem. 11 | Análise dos resultados | (2) Relatório Técnico | a fazer |
| 8 | sem. 12 | Simpósio | (3) vídeo, (4) artigo, (5) pptx | a fazer |

### Etapa 1 — Entrevistas empáticas *(concluída)*
Cada integrante entrevistou pelo menos duas pessoas para identificar a "dor". Guarde as anotações; elas
justificam o tema (fiscalização manual e intermitente do uso de EPI).

### Etapa 2 — Contexto e cenário *(concluída)*
Definido o cenário: **monitoramento automático do uso de EPI na entrada de uma área controlada**, em
"ambiente simples" (uma pessoa por vez, iluminação e fundo controlados, câmera fixa). Já entregue como o
Tema do Trabalho.

### Etapa 3 — Modelagem funcional geral *(entregue)*
Documento de projeto com requisitos, especificações, funções, diagramas de blocos, método de calibração
e método de avaliação. Está em `Modelagem Funcional Geral.ipynb`. **Nada a refazer** — as etapas
seguintes implementam este projeto.

### Etapa 4 — Desenvolvimento do projeto *(a fazer — núcleo técnico)*
É aqui que os códigos entram. Ordem sugerida:

1. **Instalar dependências**
   ```bash
   cd codigos
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Calibrar a câmera (Requisito A, obrigatório)** — `calibrar_camera.py`
   ```bash
   python3 calibrar_camera.py --capturar --camera 0 --cols 8 --rows 6
   ```
   Capture 10–15 imagens do tabuleiro (tecla `s`), calibre (o programa faz ao final) e **anote o erro de
   reprojeção** impresso — ele vai para o relatório. Gera `calib_camera.xml`.
3. **Montar e anotar o dataset** — anote imagens nas 3 classes (pessoa, capacete, colete) no Roboflow /
   CVAT / LabelImg, exporte em formato YOLO e ajuste os caminhos em `data.yaml`. Quanto mais variado
   (pessoas, cores de colete, com/sem capacete), melhor.
4. **Treinar o detector** — `treinar_modelo.py`
   ```bash
   python3 treinar_modelo.py --data data.yaml --epocas 100 --imgsz 640
   ```
   Gera `epi_yolo.pt`. (Com GPU é bem mais rápido.)
5. **Rodar o sistema em tempo real** — `deteccao_epi.py`
   ```bash
   python3 deteccao_epi.py --modelo epi_yolo.pt --camera 0 --calib calib_camera.xml
   ```
   Ajuste `--conf` até o sistema acertar sem muitos falsos. Grave demonstrações com a tecla `g`.
6. *(Maquete/hardware)* Fixe a câmera (tripé/suporte) e prepare o local do "portão de acesso".

### Etapa 5 — Roteiro de teste com voluntários *(a fazer)*
Escreva uma lista de tarefas que o voluntário fará diante da câmera, cobrindo os quatro casos:
(a) com todos os EPIs, (b) sem capacete, (c) sem colete, (d) sem nenhum. Defina como anotar o resultado
esperado (*ground truth*) de cada situação. Este roteiro é um entregável próprio da Etapa 5.

### Etapa 6 — Realização dos testes *(a fazer)*
1. Grave (celular) os voluntários executando o roteiro — material para o **vídeo de demonstração**.
2. Para cada quadro/pessoa avaliada, registre em `anotacoes.csv` o par `real,predito`
   (`conforme` / `nao_conforme`). Colete também o **feedback** (dificuldades, erros do sistema, sugestões).

### Etapa 7 — Análise dos resultados *(a fazer → Relatório Técnico)*
1. **Métricas de detecção** — `avaliar_metricas.py --modo mapa` → mAP, precisão, revocação.
2. **Métricas da decisão** — `avaliar_metricas.py --modo confusao --csv anotacoes.csv` → matriz de
   confusão + precisão/revocação/F1 (gera `matriz_confusao.png`).
3. **Latência/FPS** — leia direto na tela do `deteccao_epi.py`.
4. Escreva o **Relatório Técnico** (Jupyter + PDF) com: detalhamento matemático da calibração (erro de
   reprojeção), arquitetura do algoritmo, resultados das métricas e análise dos testes com voluntários.

### Etapa 8 — Simpósio *(a fazer)*
- **Vídeo de demonstração** (mp4, 3–5 min): problema (entrevista) → solução em tempo real → teste com
  voluntário.
- **Artigo acadêmico** (PDF, formato SBC/IEEE): introdução, metodologia, resultados, conclusão.
- **Apresentação** (pptx, ~10 min) + demonstração ao vivo.

---

## Parte B — O que cada código faz

Todos estão em `codigos/`. O fluxo de dados entre eles:

```
calibrar_camera.py ─► calib_camera.xml ─┐
                                        ├─► deteccao_epi.py ─► (janela em tempo real / sessao_epi.mp4)
treinar_modelo.py  ─► epi_yolo.pt ──────┘
       │                                     ▲
       └── usa data.yaml + dataset           │ importa epi_utils.py
                                             │
anotacoes.csv ─► avaliar_metricas.py ─► matriz_confusao.png + métricas
```

| Código | O que faz | Entradas | Saídas |
|--------|-----------|----------|--------|
| `requirements.txt` | Lista as dependências do projeto. | — | — |
| `data.yaml` | Configura o dataset de treino (caminhos + classes pessoa/capacete/colete). | — | — |
| `calibrar_camera.py` | **Requisito A.** Detecta os cantos do tabuleiro, calibra a câmera, calcula o **erro de reprojeção** e salva os parâmetros. | imagens do tabuleiro (ou webcam) | `calib_camera.xml` (K, dist, erro) |
| `treinar_modelo.py` | Treina/adapta o **YOLOv8** (transfer learning) para as 3 classes; ao final valida (mAP). | `data.yaml` + dataset anotado | `epi_yolo.pt` |
| `epi_utils.py` | Biblioteca de apoio (não roda sozinha): carrega calibração e corrige distorção; associa cada EPI à pessoa (IoU/centro dentro da caixa); **decisão temporal** (confirma o estado em N quadros); desenha caixas, rótulos, status e FPS. | importado pelos outros | funções |
| `deteccao_epi.py` | **Programa principal, tempo real.** Executa o pipeline completo: aquisição → correção de distorção → inferência CNN + rastreamento → associação EPI↔pessoa → decisão → status/alerta + FPS. Grava a sessão com a tecla `g`. | `epi_yolo.pt`, `calib_camera.xml`, webcam | janela ao vivo / `sessao_epi.mp4` |
| `avaliar_metricas.py` | Calcula as métricas do item 4: **modo `mapa`** (mAP/precisão/revocação do detector) e **modo `confusao`** (matriz de confusão + P/R/F1 da decisão de conformidade). | modelo+`data.yaml` **ou** `anotacoes.csv` | métricas no terminal + `matriz_confusao.png` |
| `README.md` | Referência rápida de instalação e execução dos códigos. | — | — |

### Detalhe do pipeline em tempo real (`deteccao_epi.py`)
1. **Aquisição** — lê o quadro da webcam.
2. **Calibração** — `cv2.undistort` com `calib_camera.xml` (corrige a distorção da lente).
3. **Inferência CNN** — YOLOv8 (`model.track`) detecta *pessoa/capacete/colete* com rastreamento embutido
   (ByteTrack) e já aplica limiar de confiança e NMS.
4. **Associação** — cada capacete/colete é vinculado à pessoa cuja caixa o contém (via `epi_utils.associar`).
5. **Decisão temporal** — por id de rastreamento, o estado só é confirmado após N quadros iguais
   (`epi_utils.DecisorTemporal`), evitando que o rótulo "pisque".
6. **Saída** — desenha caixas, rótulos (capacete/colete OK ou ausente), o **status global** (verde =
   CONFORME, vermelho = NÃO CONFORME) e os **FPS**.

### Observações importantes
- Sem `epi_yolo.pt`, o `deteccao_epi.py` cai para `yolov8n.pt` (só detecta "person") apenas para demonstrar
  o pipeline — os rótulos de EPI exigem o modelo treinado (passo 4 da Etapa 4).
- Prioriza-se **alta revocação** para `nao_conforme` (melhor um alarme falso do que deixar passar alguém
  sem EPI); ajuste o limiar `--conf` com esse critério.
- A câmera deve ficar **fixa** após a calibração.
