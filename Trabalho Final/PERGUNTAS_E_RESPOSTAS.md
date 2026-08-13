# Roteiro de perguntas e respostas — Seminário

**Detecção de Uso de EPI em Ambiente Simples** · Equipe Ctrl+C, Ctrl+V e Fé · ESZA019 2026.2

Guia de apoio para a arguição. As respostas são curtas e faladas — adapte com naturalidade e sempre
com os **números reais** de vocês. Regra de ouro: se não souberem, digam "não medimos isso, mas o
esperado é..." em vez de inventar.

---

## A. Problema e escopo

**1. Por que esse tema?**
A fiscalização do uso de EPI ainda é manual e intermitente, sujeita a falha humana e sem registro
objetivo. Nas entrevistas empáticas isso apareceu como uma dor real, e é um problema onde a visão
computacional ajuda de forma simples.

**2. Por que "ambiente simples" (uma pessoa por vez, fundo neutro)?**
Para manter o projeto específico e preciso — "menos é mais, desde que seja preciso". Delimitar reduz a
variabilidade que o modelo precisa aprender e permite bons resultados com um modelo leve e um dataset
modesto, dentro do prazo da disciplina.

**3. Por que só capacete e colete, e não todos os EPIs?**
São os EPIs mais visíveis e de cores conspícuas, ideais para um primeiro sistema robusto. A arquitetura
é a mesma para adicionar óculos, luvas, etc. — bastaria treinar o modelo com essas classes.

## B. Técnica (deep learning e calibração)

**4. Por que deep learning (YOLO) e não métodos clássicos (cor/forma)?**
Cor e forma funcionam num cenário muito controlado, mas quebram com variação de iluminação, ângulo e
tipos de colete/capacete. A CNN generaliza muito melhor e detecta pessoa e EPIs no mesmo passo, com bom
custo-benefício.

**5. Como o sistema decide "conforme / não conforme"?**
O YOLO detecta pessoa, capacete e colete; associamos cada EPI à pessoa por sobreposição das caixas
(IoU / centro dentro da caixa). Se a pessoa tem capacete E colete, é conforme. A decisão só é confirmada
após alguns quadros seguidos, para não "piscar".

**6. Por que a calibração de câmera é necessária?**
É o Requisito A do trabalho e evita que a distorção da lente deforme as caixas e a associação. Usamos o
método dos Laboratórios 4 e 5 (tabuleiro de xadrez), corrigindo a distorção antes da detecção; a
qualidade é medida pelo erro de reprojeção. *(Digam o valor que obtiveram.)*

**7. Como treinaram o modelo? De onde vieram as imagens?**
Usamos transfer learning a partir do YOLOv8 e treinamos num dataset público de EPI já anotado
(Construction Site Safety, do Roboflow), rodando no Google Colab com GPU gratuita. Não anotamos milhares
de imagens à mão.

## C. Desempenho e métricas

**8. Qual o FPS? Roda mesmo em tempo real?**
Medimos cerca de 8 FPS em CPU, sem GPU — suficiente para o cenário de uma pessoa entrando numa área.
Dá para melhorar com GPU, um modelo ainda menor ou resolução menor.

**9. Quais métricas usaram para provar que funciona?**
Três frentes: FPS/latência (desempenho); mAP do detector (qualidade da detecção); e matriz de confusão
com precisão, revocação e F1 para a decisão de conformidade. *(Preencham com os números de vocês.)*

**10. Por que priorizar a revocação e não a precisão?**
Por segurança: o erro grave é liberar alguém sem EPI (falso "conforme"). Preferimos um alarme falso a
deixar passar um risco, então otimizamos para não perder casos de "não conforme".

## D. Testes com voluntários

**11. Como foi a validação com usuários?**
7 voluntários usaram o sistema (com e sem EPI) em 10/08/2026, preencheram um questionário de usabilidade
(escala SUS) e responderam perguntas abertas; a interação foi filmada.

**12. O que os voluntários acharam?**
Feedback positivo: destacaram a rapidez e a facilidade de uso, e confirmaram que o modelo reconheceu os
equipamentos. A média SUS dos 7 é [consolidar]. A sugestão principal foi mostrar a avaliação parcial —
indicar quando falta só um EPI.

**13. Só 7 pessoas não é pouco?**
É uma amostra pequena, coerente com o escopo de um trabalho de disciplina. Serve para uma avaliação
qualitativa de usabilidade e para identificar melhorias; não é uma validação estatística ampla, e
deixamos isso claro.

## E. Limitações e trabalhos futuros

**14. E se tiver mais de uma pessoa no quadro?**
Está fora do escopo "ambiente simples" desta versão. O rastreamento já lida com múltiplas caixas, mas a
avaliação simultânea de várias pessoas seria um passo seguinte.

**15. O modelo generaliza para o colete/capacete de vocês?**
Como foi treinado com dados públicos, pode não pegar perfeitamente a cor exata do nosso colete. A solução
é incluir algumas fotos nossas no treino (fine-tuning) — algo que planejamos.

**16. Como a câmera estéreo dos Labs 5 e 6 entra aqui?**
Como extensão: a profundidade permite medir a distância da pessoa e avaliar o EPI só de quem está na zona
monitorada (1,5–3 m), reduzindo falsos positivos de gente ao fundo.

**17. E questões de privacidade/uso real?**
Num uso real seria preciso consentimento e cuidado com dados de imagem. No nosso escopo acadêmico, o
processamento é local e em tempo real, sem armazenar identidade — só o status de conformidade.

## F. Perguntas "difíceis"

**18. O que acontece se a iluminação for ruim ou o fundo bagunçado?**
A detecção piora — por isso delimitamos o ambiente. É uma limitação assumida; robustez a esses fatores
exigiria mais dados de treino e controle de exposição.

**19. Qual foi a maior dificuldade do projeto?**
Integrar as partes (calibração + detector + decisão) e treinar o modelo sem GPU local — resolvido usando
o Colab. *(Ajustem ao que foi real para vocês.)*

**20. Se tivessem mais tempo, o que fariam?**
Fine-tuning com imagens próprias, avaliação parcial na tela, suporte a múltiplas pessoas e otimização
para mais FPS.

---

### Dicas rápidas de postura
- Tragam sempre os **números reais** (FPS, SUS médio, mAP/F1, erro de reprojeição) na ponta da língua.
- Ao citar limitações, mostrem que foram **escolhas conscientes** de escopo, não falhas.
- Se travarem numa pergunta, devolvam com honestidade: "não medimos, mas o esperado seria...".
