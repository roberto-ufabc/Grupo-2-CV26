# Laboratório 6 — Depth Map: passo a passo de reprodução

Equipe **Ctrl+C, Ctrl+V e Fé** — ESZA019 Visão Computacional 2026.2

Guia para reproduzir os procedimentos da Parte 2 do roteiro, do mapa de disparidade até a medição de
distância. Os programas estão nesta pasta e leem o `params_py.xml` da câmera estéreo do **Laboratório 5**
(já presente aqui).

## 0. Pré-requisitos

- Linux, Python 3, `pip install opencv-contrib-python numpy matplotlib`.
- A **câmera estéreo do Lab 5** montada e **fixa** (as webcams não podem se mover após a calibração).
- `params_py.xml` nesta pasta (mapas de retificação, `Q` e `baseline`). Já incluído.
- Ajustar, se necessário, os índices das câmeras (`CamL_id = 0`, `CamR_id = 2`) no topo dos scripts,
  conforme `v4l2-ctl --list-devices`.

## 1. (Opcional) Validar os algoritmos sem webcam

```bash
python3 disparidade_offline.py
```

Gera `demo_disparidade_sgbm.png` e `demo_disparidade_bm_tsukuba.png` a partir de pares de imagens
gravados. Serve para conferir que o pipeline de correspondência funciona e como figura da fundamentação
teórica. (Requer as imagens de exemplo `im0/im1` e `tsukuba_l/r` na pasta indicada em `PASTA`.)

## 2. (Roteiro ii) Sintonizar o Block Matching → mapa de disparidade

```bash
python3 disparity_params_gui.py
```

- Ajuste as barras deslizantes; pressione **`s`** para salvar (`depth_estimation_params_py.xml`) e
  **`q`** para sair.

**Valores recomendados** (corrigidos após a análise do relatório — a primeira execução usou a faixa de
busca mínima e limitou a medição a distâncias acima de ~2,3 m):

| Parâmetro | Valor sugerido | Por quê |
|-----------|----------------|---------|
| `numDisparities` | **144 a 224** | define a **distância mínima** mensurável; 16 só mede além de ~2,3 m |
| `minDisparity` | **0** | libera as distâncias maiores |
| `blockSize` | **9 a 15** | menos ruído que 5, sem perder muito detalhe |
| `preFilterCap` | ~30 | realce das bordas antes do casamento |
| `uniquenessRatio` | 10–15 | subir só até o ruído sumir |
| `textureThreshold` | ~10 | descarta regiões lisas |
| `speckleWindowSize` / `speckleRange` | ~50 / ~2 | remove manchas isoladas |

> **O mapa NÃO deve ficar preto.** Preto total significa que todos os pixels foram invalidados (filtros
> agressivos demais, `minDisparity` alto ou câmeras esquerda/direita **invertidas**). O alvo é um mapa
> limpo em que ainda se enxerga o **relevo da cena**.

Referência rápida: com $f = 676{,}57$ px e $B = 6{,}729$ cm, vale $Z(\text{cm}) = 4553/d$. Para medir a
partir de 35 cm é preciso cobrir $d \approx 130$ px, logo `numDisparities` ≥ 144.

## 3. (Roteiro iii) Calibrar disparidade → profundidade

```bash
python3 disparity2depth_calib.py
```

- Posicione um objeto a uma **distância real conhecida** (trena).
- **Clique** sobre ele no mapa e **digite no terminal** a distância real (cm).
- Repita para **pelo menos 5 distâncias** dentro da faixa mensurável configurada no passo 2.
- **`q`** encerra: ajusta `Z = M·(1/d) + C`, salva `M` e `C` no xml e gera
  `profundidade_vs_disparidade.png`.

*Verificação de sanidade:* se a sintonia estiver correta, `M` deve ficar próximo de $f\cdot B \approx
4553$ e `C` próximo de 0. Valores muito distantes disso indicam que as disparidades medidas estão fora da
faixa de busca.

## 4. (Roteiro iv e v) Medir distâncias e montar a tabela de erros

```bash
python3 obstacle_avoidance.py
```

Destaca a região mais próxima e mostra a distância estimada. Meça **pelo menos três objetos distintos** a
distâncias reais conhecidas e preencha:

| Objeto | Distância real (cm) | Distância medida (cm) | Erro (cm) | Erro (%) |
|--------|---------------------|-----------------------|-----------|----------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

**Anote sempre a distância real com trena** — sem ela não é possível fechar a análise de erro do item (v).

## 5. (Roteiro vi) Programa completo: distância a um objeto específico

```bash
python3 medir_distancia_objeto.py
```

Clique no objeto para obter a distância. Ajustado ao tema do Trabalho Final (**Detecção de EPI**):
indica se o objeto está dentro da **zona monitorada** (1,5–3 m, editável em `ZONA_MIN_CM`/`ZONA_MAX_CM`).

## 6. (Roteiro vii) Registro para o relatório

Grave imagens e vídeos de cada etapa (mapa sintonizado, gráfico Z×d, medições) e insira no
`Relatório.ipynb` junto com a tabela de erros e a análise.

## Resumo dos arquivos

| Arquivo | Função | Entrada | Saída |
|---------|--------|---------|-------|
| `disparidade_offline.py` | Disparidade em imagens gravadas (sem webcam) | pares de exemplo | PNGs de disparidade |
| `disparity_params_gui.py` | Sintonia do StereoBM ao vivo (ii) | `params_py.xml` + 2 webcams | `depth_estimation_params_py.xml` |
| `disparity2depth_calib.py` | Calibra disparidade→profundidade (iii) | acima + amostras (d, Z) | `M`,`C` no xml + gráfico |
| `obstacle_avoidance.py` | Distância do obstáculo mais próximo (iv/v) | os dois xml + webcams | janela com distância |
| `medir_distancia_objeto.py` | Distância a um objeto por clique (vi) | os dois xml + webcams | distância na tela |

> **Fluxo:** `params_py.xml` → `disparity_params_gui.py` → `depth_estimation_params_py.xml` →
> `disparity2depth_calib.py` (adiciona `M`,`C`) → `obstacle_avoidance.py` / `medir_distancia_objeto.py`.
