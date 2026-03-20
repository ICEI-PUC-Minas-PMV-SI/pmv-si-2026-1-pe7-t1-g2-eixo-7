# Introdução

O acesso ao crédito é um dos principais mecanismos que impulsionam o desenvolvimento econômico, permitindo que indivíduos e empresas realizem investimentos, adquiram bens e financiem atividades produtivas. Instituições financeiras, como bancos e fintechs, desempenham um papel fundamental nesse processo ao oferecer empréstimos e outros produtos de crédito para diferentes perfis de clientes.

Entretanto, a concessão de crédito envolve riscos, sendo a inadimplência um dos principais desafios enfrentados pelas instituições financeiras. A inadimplência ocorre quando um cliente não consegue cumprir suas obrigações de pagamento dentro do prazo estabelecido, gerando prejuízos financeiros e aumentando a incerteza no sistema de crédito.

## Problema

A concessão de crédito exige que instituições financeiras avaliem cuidadosamente o risco associado a cada cliente. Decidir se um empréstimo deve ou não ser concedido envolve analisar diversos fatores, como renda, histórico de crédito, valor solicitado e outras características financeiras do solicitante.

Quando o risco de inadimplência não é identificado corretamente, podem ocorrer dois tipos de problema. No primeiro caso, a instituição concede crédito a clientes com alto risco de inadimplência, o que pode gerar perdas financeiras. No segundo caso, clientes com bom perfil financeiro podem ter seu crédito negado, reduzindo o acesso ao sistema financeiro.

Neste projeto, o problema investigado consiste em identificar se é possível utilizar técnicas de aprendizado de máquina para prever a probabilidade de inadimplência em empréstimos com base em informações financeiras e demográficas dos clientes presentes em um dataset público. O estudo será realizado em um contexto experimental e acadêmico, utilizando ferramentas de análise de dados e modelagem de Machine Learning.

## Questão de pesquisa

Como técnicas de aprendizado de máquina podem ser utilizadas para prever a inadimplência em empréstimos financeiros a partir de dados demográficos e financeiros dos clientes?

Essa pergunta direciona a investigação realizada no projeto, buscando compreender se modelos de Machine Learning conseguem identificar padrões relevantes nos dados que permitam classificar clientes com maior risco de inadimplência.

Ao final da pesquisa, espera-se que seja possível responder a essa questão por meio da análise do desempenho dos modelos desenvolvidos e das evidências obtidas durante os experimentos.

## Objetivos preliminares

### Objetivo Geral

Experimentar e avaliar modelos de aprendizado de máquina capazes de prever a inadimplência em empréstimos financeiros utilizando dados históricos de clientes.

### Objetivos Específicos

#### Objetivo específico 1:
Preparar e estruturar o dataset para análise e modelagem, realizando o tratamento de dados ausentes, inconsistências e variáveis necessárias para o treinamento dos modelos.

#### Objetivo específico 2:
Analisar quais variáveis do dataset possuem maior influência na previsão de inadimplência, contribuindo para uma melhor compreensão do problema de risco de crédito.

## Justificativa

A inadimplência representa um dos principais desafios enfrentados pelo setor financeiro. De acordo com dados do Banco Central do Brasil, o nível de inadimplência no sistema financeiro pode impactar diretamente a estabilidade do mercado de crédito, afetando tanto instituições financeiras quanto consumidores.

Quando níveis elevados de inadimplência ocorrem, os bancos tendem a aumentar as taxas de juros ou restringir a concessão de crédito, o que pode dificultar o acesso ao financiamento para consumidores e empresas. Dessa forma, melhorar a capacidade de prever o risco de inadimplência pode contribuir para decisões de crédito mais eficientes e equilibradas.

O uso de técnicas de aprendizado de máquina tem se mostrado uma abordagem promissora nesse contexto, pois permite identificar padrões complexos em grandes volumes de dados que seriam difíceis de detectar por métodos tradicionais de análise estatística. Esses modelos podem apoiar o processo de análise de crédito, reduzindo riscos financeiros e aumentando a eficiência na concessão de empréstimos.

O dataset escolhido para este projeto contém diversas informações relevantes sobre solicitantes de empréstimos, como características demográficas, financeiras e histórico de crédito. Esse conjunto de dados permite investigar como diferentes variáveis podem influenciar a probabilidade de inadimplência e avaliar a eficácia de diferentes algoritmos de aprendizado de máquina na solução desse problema.

## Público-Alvo

O público-alvo deste projeto inclui instituições financeiras, como bancos, fintechs e cooperativas de crédito, que realizam processos de análise e concessão de empréstimos. Os resultados do estudo podem contribuir para a melhoria da avaliação de risco de crédito, auxiliando na identificação de clientes com maior probabilidade de inadimplência.

Além disso, o projeto também se destina a profissionais da área financeira, como analistas de crédito e gestores de risco, bem como pesquisadores e estudantes das áreas de ciência de dados, administração, economia e sistemas de informação, interessados na aplicação de técnicas de aprendizado de máquina na análise de dados financeiros.

# Estado da Arte

A predição de inadimplência em operações de crédito é um dos problemas mais relevantes do setor financeiro, com impacto direto na lucratividade das instituições, na concessão de crédito justo e na estabilidade econômica. A crescente disponibilidade de dados de solicitantes de empréstimo, aliada ao avanço de algoritmos de aprendizado de máquina, tem impulsionado pesquisas que buscam modelos preditivos mais precisos, interpretativos e robustos do que os sistemas tradicionais baseados em regras.

Esta seção documenta cinco trabalhos da literatura recente que tratam de problemas semelhantes ao proposto neste projeto, registrando para cada um: o problema e contexto, os dados utilizados, as abordagens e algoritmos empregados, as métricas de avaliação, os principais resultados obtidos e a correlação direta com o presente trabalho. Ao final, uma síntese crítica discute convergências, divergências, lacunas identificadas e o alinhamento com o projeto em desenvolvimento.
# Estado da arte

A predição de inadimplência em operações de crédito é um dos problemas mais relevantes do setor financeiro, com impacto direto na lucratividade das instituições, na concessão de crédito justo e na estabilidade econômica. A crescente disponibilidade de dados de solicitantes de empréstimo, aliada ao avanço de algoritmos de aprendizado de máquina, tem impulsionado pesquisas que buscam modelos preditivos mais precisos, interpretativos e robustos do que os sistemas tradicionais baseados em regras.

Esta seção documenta cinco trabalhos da literatura recente que tratam de problemas semelhantes ao proposto neste projeto, registrando para cada um: o problema e contexto, os dados utilizados, as abordagens e algoritmos empregados, as métricas de avaliação e os principais resultados obtidos. Ao final, uma síntese crítica discute convergências, divergências, lacunas identificadas e o alinhamento com o projeto em desenvolvimento.

---

## Estudos Levantados

### Estudo 1 — Akinjole et al. (2024)

AKINJOLE, A. et al. Ensemble-Based Machine Learning Algorithm for Loan Default Risk Prediction. *Mathematics*, v. 12, n. 21, p. 3423, out. 2024. DOI: 10.3390/math12213423.

| Campo | Descrição |
|---|---|
| **Problema e Contexto** | Predição de risco de inadimplência em empréstimos pessoais concedidos por plataformas P2P (peer-to-peer lending). O desafio central é identificar, dentre solicitantes com múltiplos atributos, aqueles com maior probabilidade de não honrar o compromisso financeiro, reduzindo perdas para credores. |
| **Dados (Dataset)** | Dados da plataforma LendingClub (EUA), período 2007–2015, com forte desequilíbrio de classes. Técnicas testadas: ROS, RUS, SMOTE, ADASYN, Tomek Links, SMOTE-Tomek e SMOTE+ENN. |
| **Abordagem / Algoritmos** | Comparação entre Random Forest, Decision Tree, SVM, XGBoost, ADABoost e MLP. Ensemble proposto: XGBoost + ADABoost. |
| **Métricas de Avaliação** | Acurácia, Precisão, Recall e AUC-ROC (com ênfase em Recall). |
| **Resultados** | Ensemble superou modelos individuais; SMOTE+ENN apresentou melhor desempenho geral. **Limitação:** restrição ao contexto P2P norte-americano. |
| **🔗 Relação com o Presente Trabalho** | **Semelhanças:** o problema central — predição de inadimplência com desequilíbrio de classes — é diretamente equivalente ao deste projeto. Os algoritmos de ensemble (XGBoost, Random Forest, ADABoost), as técnicas de balanceamento (SMOTE e variantes) e as métricas priorizadas (AUC-ROC, Recall, F1-score) são os mesmos. **Diferenças:** Akinjole et al. utilizam o dataset LendingClub (EUA, contexto P2P), enquanto o presente projeto utiliza o Loan Default Dataset do Kaggle (148.670 registros, 34 atributos), com escopo demográfico e financeiro mais amplo. O presente trabalho também inclui etapa explícita de interpretabilidade (SHAP/LIME), não abordada por Akinjole et al. |

---

### Estudo 2 — Lin et al. (2022)

LIN, C.; QIAO, N.; ZHANG, W.; LI, Y.; MA, S. Default risk prediction and feature extraction using a penalized deep neural network. *Statistics and Computing*, v. 32, n. 76, set. 2022. DOI: 10.1007/s11222-022-10140-z.

| Campo | Descrição |
|---|---|
| **Problema e Contexto** | Predição de inadimplência com modelagem temporal via análise de sobrevivência. |
| **Dados (Dataset)** | LendingClub, com tratamento de censura à direita e treinamento em larga escala via minibatch gradient descent. |
| **Abordagem / Algoritmos** | Rede neural profunda penalizada com L1, comparada a modelos de sobrevivência clássicos (Cox com Lasso). |
| **Métricas de Avaliação** | C-statistic, erro de Brier e calibração. |
| **Resultados** | Melhor desempenho e seleção automática de atributos. **Limitação:** baixa interpretabilidade e ausência de avaliação em cenários adversos. |
| **🔗 Relação com o Presente Trabalho** | **Semelhanças:** ambos os trabalhos lidam com predição de inadimplência em empréstimos e buscam identificar atributos relevantes para o default. A preocupação com seleção de atributos (via penalização L1 em Lin et al., e via importância de features e SHAP neste projeto) é um ponto de convergência metodológica. **Diferenças:** Lin et al. enquadram o problema como análise de sobrevivência (estimando probabilidade de default ao longo do tempo), enquanto o presente projeto trata o problema como classificação binária supervisionada — abordagem mais alinhada às práticas operacionais de concessão de crédito. O dataset utilizado (Loan Default Dataset, Kaggle, 148.670 registros) não possui estrutura temporal de censura. A interpretabilidade, ponto cego em Lin et al., é um objetivo explícito deste projeto. |
---

### Estudo 1 — Akinjole et al. (2024)

> AKINJOLE, A. et al. Ensemble-Based Machine Learning Algorithm for Loan Default Risk Prediction. *Mathematics*, v. 12, n. 21, p. 3423, out. 2024. DOI: [10.3390/math12213423](https://doi.org/10.3390/math12213423).

| Campo | Descrição |
|---|---|
| **Problema e Contexto** | Predição de risco de inadimplência em empréstimos pessoais concedidos por plataformas P2P (*peer-to-peer lending*). O desafio central é identificar, dentre solicitantes com múltiplos atributos, aqueles com maior probabilidade de não honrar o compromisso financeiro, reduzindo perdas para credores. O contexto é o de plataformas digitais de crédito, com dados heterogêneos e desequilíbrio severo de classes. |
| **Dados (Dataset)** | Dados da plataforma LendingClub (EUA), período 2007–2015. Atributos: idade, status de emprego, tempo de emprego, renda, finalidade do empréstimo, histórico de crédito e status do pagamento. O desequilíbrio de classes é marcante — a maioria dos registros corresponde a empréstimos pagos integralmente. Técnicas de reamostragem testadas: ROS, RUS, SMOTE, ADASYN, Tomek Links, SMOTE-Tomek e SMOTE+ENN. |
| **Abordagem / Algoritmos** | Avaliação comparativa de Random Forest, Decision Tree, SVM, XGBoost, ADABoost e MLP (Perceptron Multicamadas). Modelo ensemble proposto combinando XGBoost e ADABoost. Divisões treino/teste testadas de 60/40 a 80/20, com 80/20 selecionada como ótima. Seleção de atributos via importância de features no XGBoost. |
| **Métricas de Avaliação** | Acurácia, Precisão, Recall e AUC-ROC. O Recall foi destacado como métrica central, pois identificar inadimplentes (verdadeiros positivos) é mais crítico do que classificar corretamente quem paga. |
| **Resultados** | O ensemble (XGBoost + ADABoost + SMOTE+ENN) superou modelos individuais em Recall e AUC. MLP com três camadas ocultas atingiu 93,2% de acurácia com SMOTE, contra 77,1% da Regressão Logística. SMOTE+ENN foi a técnica de balanceamento com melhor desempenho geral. **Limitação:** resultado limitado ao contexto P2P norte-americano, sem validação em outros mercados. |

---

### Estudo 2 — Lin et al. (2022)

> LIN, C.; QIAO, N.; ZHANG, W.; LI, Y.; MA, S. Default risk prediction and feature extraction using a penalized deep neural network. *Statistics and Computing*, v. 32, n. 76, set. 2022. DOI: [10.1007/s11222-022-10140-z](https://doi.org/10.1007/s11222-022-10140-z).

| Campo | Descrição |
|---|---|
| **Problema e Contexto** | Predição de risco de inadimplência em plataformas P2P, com ênfase na modelagem da *probabilidade de default ao longo do tempo*, tratando o problema como análise de sobrevivência. O contexto é o das plataformas digitais de crédito, onde a assimetria de informação entre tomadores e credores representa o principal desafio para a avaliação de risco. |
| **Dados (Dataset)** | Dados reais da plataforma LendingClub, com grande volume de registros de empréstimos. Variáveis incluem histórico de crédito, dados demográficos, características do empréstimo e comportamento de pagamento. O pré-processamento incluiu tratamento de censura à direita (característica dos dados de sobrevivência) e uso de minibatch gradient descent para viabilizar o treinamento em larga escala. |
| **Abordagem / Algoritmos** | Rede neural profunda penalizada (*Penalized Deep Neural Network*) com camada extra *one-to-one* que incorpora penalização L1 na função objetivo, permitindo seleção de atributos e estimação simultâneas. Comparado com Cox Proportional Hazards com Lasso e métodos clássicos de ML. |
| **Métricas de Avaliação** | C-statistic (equivalente à AUC para dados de sobrevivência), erro de Brier e calibração das probabilidades previstas — métricas apropriadas para modelos que estimam probabilidades de evento ao longo do tempo. |
| **Resultados** | O modelo de rede neural penalizada superou os modelos de sobrevivência paramétrica e o Cox com Lasso em C-statistic e calibração. A penalização L1 produziu seleção automática de atributos relevantes, como histórico de inadimplência recente e tipo de propriedade. **Limitação:** redes neurais profundas continuam sendo modelos de difícil interpretação; o estudo não avalia o comportamento em períodos de crise econômica, limitando a robustez em cenários adversos. |

---

### Estudo 3 — Zhang et al. (2025)

ZHANG, X.; ZHANG, T.; HOU, L.; LIU, X.; GUO, Z.; TIAN, Y.; LIU, Y. Data-Driven Loan Default Prediction: A Machine Learning Approach for Enhancing Business Process Management. *Systems*, v. 13, n. 7, p. 581, jul. 2025. DOI: 10.3390/systems13070581.

| Campo | Descrição |
|---|---|
| **Problema e Contexto** | Uso de ML para melhorar processos de concessão de crédito e reduzir empréstimos não performados (NPLs). |
| **Dados (Dataset)** | Dataset sintético derivado do Credit Risk Dataset do Kaggle (`taweilo/loan-approval-classification-data`), enriquecido com variáveis adicionais. Atributos incluem renda, score de crédito, taxa de juros, finalidade do empréstimo e histórico de inadimplência. |
| **Abordagem / Algoritmos** | Pipeline completo com SMOTE, engenharia de atributos e tuning via GridSearchCV com validação cruzada 5-fold. Modelos: XGBoost, Gradient Boosting, Random Forest e LightGBM. |
| **Métricas de Avaliação** | Acurácia, F1-score, ROC AUC e matriz de confusão. |
| **Resultados** | Gradient Boosting com melhor desempenho geral (acurácia = 0,8887, F1 = 0,8084, Recall = 0,8021). **Limitação:** natureza black-box e sensibilidade ao desbalanceamento. |
| **🔗 Relação com o Presente Trabalho** | **Semelhanças:** o pipeline adotado — pré-processamento, engenharia de atributos, SMOTE, tuning via validação cruzada e comparação entre XGBoost, Gradient Boosting e Random Forest — é estruturalmente equivalente ao deste projeto. As métricas (F1-score, ROC AUC, Recall, matriz de confusão) são as mesmas. Ambos reconhecem a natureza black-box como limitação regulatória. **Diferenças:** Zhang et al. utilizam um dataset sintético derivado do Kaggle (`taweilo/loan-approval-classification-data`), com conjunto de variáveis mais enxuto, enquanto o presente projeto utiliza o Loan Default Dataset (148.670 registros, 34 atributos), com maior volume e cobertura de variáveis demográficas e financeiras. Este projeto também incorpora explicitamente ferramentas de XAI (SHAP/LIME), lacuna identificada por Zhang et al. como trabalho futuro. |
> ZHANG, X.; ZHANG, T.; HOU, L.; LIU, X.; GUO, Z.; TIAN, Y.; LIU, Y. Data-Driven Loan Default Prediction: A Machine Learning Approach for Enhancing Business Process Management. *Systems*, v. 13, n. 7, p. 581, jul. 2025. DOI: [10.3390/systems13070581](https://doi.org/10.3390/systems13070581).

| Campo | Descrição |
|---|---|
| **Problema e Contexto** | Predição de inadimplência como ferramenta de gestão de processos de negócio em instituições financeiras, com foco em como modelos de ML podem ser integrados ao pipeline de aprovação de crédito para reduzir empréstimos não performados (NPLs). |
| **Dados (Dataset)** | Dataset de risco de crédito com variáveis financeiras e atributos categóricos e contínuos relacionados ao cliente e ao empréstimo. O desbalanceamento de classes foi tratado com SMOTE e ponderação de classes (*class weighting*). |
| **Abordagem / Algoritmos** | Pipeline estruturado com pré-processamento, engenharia de atributos, balanceamento (SMOTE + class weighting), treinamento e tuning de hiperparâmetros via GridSearchCV com validação cruzada 5-fold. Modelos avaliados: XGBoost, Gradient Boosting, Random Forest e LightGBM. Para XGBoost, o tuning concentrou-se em learning rate, profundidade máxima da árvore, subsample ratio e número de estimadores. |
| **Métricas de Avaliação** | Acurácia, F1-score, ROC AUC, curvas precisão-recall e matrizes de confusão — combinação justificada pela necessidade de avaliar performance tanto na classe majoritária quanto na minoritária. |
| **Resultados** | Gradient Boosting obteve a melhor performance geral: acurácia = 0,8887, F1-score = 0,8084, Recall = 0,8021. XGBoost e LightGBM foram competitivos, com AUC próximos. **Limitação:** natureza *black-box* dos modelos representa desafio regulatório; o desequilíbrio de classes, mesmo com SMOTE, continuou afetando a sensibilidade do modelo. |

---

### Estudo 4 — Jin, Wu e Zhao (2022)

JIN, L.; WU, Z.; ZHAO, J. Prediction of Credit Card Defaulters Based on SMOTE-XGBoost Model. In: *WHICEB 2022 Proceedings*, 2022. Disponível em: https://aisel.aisnet.org/whiceb2022/64/.

| Campo | Descrição |
|---|---|
| **Problema e Contexto** | Inadimplência em cartões de crédito, com foco na superação das limitações de algoritmos tradicionais frente ao forte desequilíbrio de classes. |
| **Dados (Dataset)** | Dataset UCI Machine Learning Repository (Taiwan), 30.000 registros, com atributos demográficos e histórico de pagamentos. |
| **Abordagem / Algoritmos** | SMOTE + XGBoost, validado por 10-fold cross-validation. Comparado com XGBoost sem SMOTE e com Random Forest. |
| **Métricas de Avaliação** | Recall, Acurácia e AUC. |
| **Resultados** | SMOTE melhora significativamente o Recall. **Limitação:** domínio específico (cartão de crédito taiwanês) e ausência de validação externa. |
| **🔗 Relação com o Presente Trabalho** | **Semelhanças:** a combinação SMOTE + XGBoost é um dos pilares metodológicos deste projeto, e a conclusão de Jin et al. — de que o SMOTE é essencial para melhorar o Recall sobre a classe minoritária — corrobora diretamente as escolhas técnicas aqui feitas. As métricas (Recall, AUC, Acurácia) e o uso de validação cruzada são os mesmos. **Diferenças:** Jin et al. trabalham com inadimplência em cartão de crédito (domínio e dinâmica distintos dos empréstimos pessoais), com apenas 30.000 registros (UCI, Taiwan). O presente projeto utiliza o Loan Default Dataset (148.670 registros, 34 atributos), com maior volume e cobertura de variáveis. Jin et al. também não abordam interpretabilidade, lacuna que este projeto busca suprir. |
> JIN, L.; WU, Z.; ZHAO, J. Prediction of Credit Card Defaulters Based on SMOTE-XGBoost Model. In: *WHICEB 2022 Proceedings*, 2022. Disponível em: [https://aisel.aisnet.org/whiceb2022/64/](https://aisel.aisnet.org/whiceb2022/64/).

| Campo | Descrição |
|---|---|
| **Problema e Contexto** | Inadimplência em cartões de crédito, problema que expõe bancos comerciais a crises de negócio. O objetivo é propor um modelo (SMOTE-XGBoost) para superar as limitações dos algoritmos tradicionais frente ao forte desequilíbrio de classes típico desse contexto, identificando inadimplentes antes da ocorrência do default. |
| **Dados (Dataset)** | Dataset UCI Machine Learning Repository de inadimplência de cartão de crédito (Taiwan, 30.000 registros). Atributos: dados demográficos (sexo, escolaridade, estado civil, idade), histórico de pagamentos, valores de extratos e pagamentos anteriores. Pré-processamento com SMOTE para balanceamento da classe minoritária. |
| **Abordagem / Algoritmos** | Combinação de SMOTE (oversampling da classe minoritária) com XGBoost (gradient boosting). Comparado com XGBoost sem SMOTE e com Random Forest. Validação por 10-fold cross-validation. |
| **Métricas de Avaliação** | Recall, Acurácia e AUC. O Recall é central por medir a capacidade de identificar inadimplentes reais; a validação 10-fold garante que os resultados não sejam artefatos de uma única divisão dos dados. |
| **Resultados** | O modelo SMOTE-XGBoost superou XGBoost puro e Random Forest em Recall e AUC, demonstrando que o balanceamento via SMOTE é fundamental para melhorar a detecção da classe minoritária. **Limitação:** estudo restrito a dados de cartão de crédito taiwanês, sem avaliação em outros domínios ou períodos econômicos distintos. |

---

### Estudo 5 — Abbas, Ying e Ayoubi (2026)

ABBAS, G.; YING, Z.; AYOUBI, M. Consensus-driven feature selection for transparent and robust loan default prediction. *Scientific Reports*, v. 16, n. 1496, jan. 2026. DOI: 10.1038/s41598-025-31468-2.

| Campo | Descrição |
|---|---|
| **Problema e Contexto** | Predição de inadimplência com foco em interpretabilidade e seleção de atributos para conformidade regulatória. |
| **Dados (Dataset)** | Datasets de crédito de alta dimensionalidade com desequilíbrio de classes. SMOTE aplicado exclusivamente ao treino para evitar data leakage. |
| **Abordagem / Algoritmos** | Framework HRA-FS (Hybrid Rank-Aggregated Feature Selection) combinando ReliefF, RFE e ElasticNet via Borda count. XGBoost como classificador base. |
| **Métricas de Avaliação** | Acurácia, Precisão, Recall e F1-score, com avaliação qualitativa de interpretabilidade. |
| **Resultados** | Melhor desempenho e redução de redundância de features. **Limitação:** alta complexidade computacional e ausência de validação externa. |
| **🔗 Relação com o Presente Trabalho** | **Semelhanças:** este é o estudo com maior proximidade temática ao presente projeto. Ambos tratam predição de inadimplência em empréstimos, utilizam XGBoost como classificador central, aplicam SMOTE apenas ao treino (evitando data leakage), adotam as mesmas métricas (Recall, F1-score) e compartilham explicitamente a preocupação com XAI e conformidade regulatória. **Diferenças:** Abbas et al. propõem um novo framework de seleção de features (HRA-FS) como contribuição central, enquanto este projeto foca na comparação sistemática de modelos de ensemble com interpretabilidade via SHAP/LIME. O dataset utilizado por Abbas et al. não é o mesmo — o presente projeto utiliza o Loan Default Dataset (Kaggle, 148.670 registros, 34 atributos documentados). Adicionalmente, este projeto avalia questões de fairness e viés demográfico, ponto não coberto por Abbas et al. |
> ABBAS, G.; YING, Z.; AYOUBI, M. Consensus-driven feature selection for transparent and robust loan default prediction. *Scientific Reports*, v. 16, n. 1496, jan. 2026. DOI: [10.1038/s41598-025-31468-2](https://doi.org/10.1038/s41598-025-31468-2).

| Campo | Descrição |
|---|---|
| **Problema e Contexto** | Predição de inadimplência com ênfase em interpretabilidade e seleção de atributos. O problema central é que métodos tradicionais de seleção de features sofrem com redundância e instabilidade, gerando modelos subótimos e pouco interpretáveis — contexto de instituições financeiras que precisam de modelos auditáveis para conformidade regulatória. |
| **Dados (Dataset)** | Datasets de crédito com alto número de dimensões e acentuado desequilíbrio de classes. SMOTE aplicado exclusivamente ao conjunto de treino para evitar *data leakage*. Categorização estratégica de features utilizada para mitigar dominância de domínio. |
| **Abordagem / Algoritmos** | Framework HRA-FS (*Hybrid Rank-Aggregated Feature Selection*), integrando três métodos — ReliefF, Recursive Feature Elimination (RFE) e ElasticNet — combinados via agregação de Borda count. XGBoost utilizado como classificador base para avaliar os subconjuntos de atributos selecionados. |
| **Métricas de Avaliação** | Acurácia, Precisão, Recall e F1-score. A interpretabilidade dos atributos selecionados também é avaliada qualitativamente, em linha com exigências de XAI (*Explainable AI*) em finanças. |
| **Resultados** | O HRA-FS produziu subconjuntos de atributos mais concisos e não redundantes, com melhor desempenho em Recall e F1 em relação à seleção por métodos individuais. A categorização de features por domínio reduziu o viés de atributos numericamente dominantes. **Limitação:** o framework aumenta a complexidade computacional do pipeline; validação externa em outros datasets não foi realizada. |

---

## Síntese Crítica

**Convergências entre os estudos.**
Há consenso quanto à superioridade de métodos de ensemble (especialmente XGBoost e Gradient Boosting) sobre modelos mais simples como regressão logística e árvores de decisão individuais, resultado observado em diferentes contextos (P2P lending, cartões de crédito, empréstimos bancários). Todos os estudos reconhecem o desequilíbrio de classes como o principal desafio metodológico, com SMOTE como técnica predominante para mitigá-lo. Métricas como AUC-ROC, F1-score e Recall são consistentemente priorizadas sobre acurácia pura.

**Divergências e nuances.**
A principal variação está na formulação do problema: Lin et al. (2022) adotam análise de sobrevivência em vez de classificação binária, adicionando riqueza preditiva mas também complexidade metodológica. Questões de interpretabilidade são tratadas de forma desigual — Jin et al. (2022) não a discutem, enquanto Abbas et al. (2026) e Zhang et al. (2025) a reconhecem explicitamente como limitação regulatória e apontam para a necessidade de frameworks como SHAP ou LIME.

**Lacunas identificadas.**
Destacam-se a ausência de datasets de mercados diversos (especialmente fora de EUA e Ásia), pouca exploração de fairness e viés demográfico, e ausência de análise de *concept drift* ao longo do tempo, o que limita a aplicabilidade prática dos modelos em produção.

**Alinhamento com o projeto.**
O presente projeto utiliza o Loan Default Dataset (Kaggle, 148.670 registros, 34 atributos), com volume superior ao dataset UCI de Jin et al. (30.000 registros) e cobertura de variáveis mais ampla do que o dataset sintético de Zhang et al. A metodologia — baseada em ensembles, SMOTE e métricas robustas — está alinhada à literatura consolidada nos cinco estudos analisados. Como diferencial, o projeto incorpora análise de interpretabilidade (SHAP/LIME) e considera aspectos de fairness e viés demográfico, contribuindo para suprir lacunas identificadas no estado da arte.
Os cinco trabalhos compartilham um conjunto consistente de conclusões técnicas. Há consenso de que algoritmos de ensemble — especialmente XGBoost, Gradient Boosting e combinações com ADABoost ou Random Forest — superam modelos mais simples como regressão logística e árvores de decisão individuais na tarefa de predição de inadimplência, resultado observado em diferentes contextos (P2P lending, cartões de crédito, empréstimos bancários) e diferentes bases de dados. Todos os estudos também reconhecem o desequilíbrio de classes como o principal desafio metodológico: a proporção de inadimplentes é tipicamente muito menor do que a de bons pagadores, o que faz com que modelos treinados sem tratamento específico obtenham alta acurácia superficial mas baixo Recall para a classe de interesse. A técnica SMOTE emerge como a abordagem mais utilizada e recomendada para mitigar esse problema, sendo citada em quatro dos cinco estudos. Por fim, há concordância quanto às métricas relevantes: AUC-ROC, F1-score e Recall são consistentemente priorizados sobre acurácia pura, por serem mais informativos em contextos desequilibrados.

**Divergências e nuances.**
Apesar das convergências, existem divergências metodológicas relevantes. Lin et al. (2022) propõem uma abordagem distinta ao enquadrar o problema como análise de sobrevivência, estimando a probabilidade de default ao longo do tempo em vez de uma classificação binária — o que adiciona riqueza preditiva, mas também complexidade metodológica. Por outro lado, Zhang et al. (2025) e Akinjole et al. (2024) mostram que modelos de gradient boosting com engenharia de features e balanceamento cuidadoso atingem resultados competitivos (acurácia entre 0,87 e 0,93, F1 entre 0,80 e 0,93) com maior simplicidade operacional. Outra divergência diz respeito à interpretabilidade: Jin et al. (2022) não discutem explicabilidade dos modelos, ao passo que Abbas et al. (2026) e Zhang et al. (2025) reconhecem explicitamente a natureza *black-box* como limitação regulatória, apontando para a necessidade de frameworks como SHAP ou LIME.

**Lacunas identificadas na literatura.**
A análise dos estudos revela lacunas relevantes. Do ponto de vista dos dados, a maioria dos trabalhos utiliza datasets norte-americanos (LendingClub) ou asiáticos (Taiwan, China), não contemplando as especificidades do mercado de crédito brasileiro, cujas variáveis macroeconômicas, taxas de juros elevadas e perfil socioeconômico dos tomadores diferem substancialmente. Em relação às técnicas, poucos estudos realizam uma avaliação sistemática e comparativa de múltiplas etapas do pipeline de forma integrada. Do ponto de vista ético e regulatório, apenas Abbas et al. (2026) aborda de forma substantiva a interpretabilidade como requisito não funcional, sendo que questões de *fairness* (viés em atributos demográficos) e privacidade de dados permanecem pouco exploradas. Por fim, a maioria dos estudos trabalha com snapshots estáticos dos dados, sem considerar a deriva de conceito (*concept drift*) ao longo do tempo, o que limita a aplicabilidade prática dos modelos em produção.

**Alinhamento com o projeto.**
O presente projeto se alinha diretamente com o estado da arte levantado ao utilizar o Loan Default Dataset (Kaggle), que conta com 148.670 registros e 34 atributos cobrindo informações demográficas, financeiras e histórico de crédito do tomador — escopo similar ao dos datasets de LendingClub utilizados nos estudos de referência. A abordagem metodológica adotada — comparação de algoritmos de ensemble (incluindo XGBoost e Random Forest), tratamento do desequilíbrio de classes com SMOTE e avaliação por métricas como AUC, F1-score e Recall — está em consonância com as práticas consolidadas na literatura. Adicionalmente, o projeto busca contribuir para suprir lacunas identificadas ao adotar um pipeline sistemático que cobre desde o pré-processamento até a interpretabilidade dos modelos, considerando tanto a performance preditiva quanto a explicabilidade das decisões, aspecto de crescente relevância para a adoção responsável de modelos de crédito em contextos regulados.

# Descrição do _dataset_ selecionado

* Loan Default Dataset, https://www.kaggle.com/datasets/yasserh/loan-default-dataset, kaggle.com
* 149 mil registros com 34 colunas no periodo de 2019 contemplando valores e caracteristicas do emprestimo como possibilidade de amortização, inadimplencia, dados do devedor como sexo, garantia, motivo do emprestimo, entre outros.<br>
* Colunas do Dataset:

| titulo | descrição | dados faltantes |
|----------|----------|----------|
| ID | Identificador único da solicitação de empréstimo | 0 |
| year | Ano em que o empréstimo foi solicitado | 0 |
| loan_limit | Indica se o empréstimo segue normas padrão (cf) ou não (ncf) | 3344 |
| Gender | Gênero do solicitante (masculino, feminino, conjunto, não informado). | 0 |
| approv_in_adv | Indica se houve pré-aprovação do empréstimo (pre) ou não (nopre). | 908 |
| loan_type | Categoria do tipo de empréstimo (type1, 2, 3). | 0 |
| loan_purpose | Motivo do empréstimo (p1, p2, p3, p4). | 134 |
| Credit_Worthiness | Solvência/idoneidade de crédito (l1, l2). | 0 |
| open_credit | Indica se o solicitante possui contas de crédito abertas (opc) ou não (nopc). | 0 |
| business_or_commercial | Finalidade do empréstimo: Comercial (ob/c) ou pessoal (nob/c). | 0 |
| loan_amount | Valor solicitado no empréstimo. | 0 |
| rate_of_interest | Taxa de juros aplicada ao empréstimo. | 36.4K |
| Interest_rate_spread | Diferença entre a taxa de juros do empréstimo e uma taxa de referência (benchmark). | 36.6K |
| Upfront_charges | Taxas iniciais cobradas para liberar o crédito. | 39.6K |
| term | Prazo para pagamento do empréstimo (em meses). | 41 |
| Neg_ammortization | Indica se permite amortização negativa. | 121 |
| interest_only | Indica se há opção de pagar apenas juros. | 0 |
| lump_sum_payment | Indica se exige pagamento de uma parcela única final. | 0 |
| property_value | Valor do imóvel usado como garantia. | 15.1K |
| construction_type | Tipo de construção (sb, mh). | 0 |
| occupancy_type | Tipo de ocupação (principal, secundária, investimento). | 0 |
| Secured_by | Tipo de garantia real. | 0 |
| total_units | Quantidade de unidades no imóvel financiado. | 0 |
| income | Renda anual do solicitante. | 9150 |
| credit_type | Bureau de crédito consultado. | 0 |
| Credit_Score | Pontuação de crédito do solicitante. | 0 |
| co-applicant_credit_type | Bureau de crédito do co-requerente. | 0 |
| age | Idade do solicitante. | 200 |
| submission_of_application | Forma de submissão. | 0 |
| LTV | Relação entre valor do empréstimo e valor do imóvel. | 15.1K |
| Region | Região geográfica do imóvel. | 0 |
| Security_Type | Tipo de segurança/garantia. | 0 |
| Status | Em inadimplência (1) ou adimplente (0). | 0 |
| dtir1 | Debt-to-Income Ratio. | 24.1K |
# Canvas analítico

<img width="2000" height="1414" alt="Software-Analtics-Canvas-v1 0 pdf" src="https://github.com/user-attachments/assets/6a330ec8-6648-4f30-a634-c4adf0214b11" />



> **Links Úteis**:
> - [Modelo do Canvas Analítico](https://github.com/ICEI-PUC-Minas-PMV-SI/PesquisaExperimentacao-Template/blob/main/help/Software-Analtics-Canvas-v1.0.pdf)

# Vídeo de apresentação da Etapa 01

Nesta etapa, o grupo deverá produzir um vídeo de 5 a 8 minutos apresentando o trabalho realizado, no qual cada integrante deve dizer seu nome e apresentar uma parte do conteúdo desenvolvido, garantindo que todos participem ativamente da gravação. A ausência de participação de qualquer membro resultará em penalização na nota final desta etapa. Recomenda-se que o grupo elabore previamente um roteiro para organizar a ordem das falas, distribuir o tempo de forma equilibrada e assegurar que todos os tópicos relevantes sejam apresentados de maneira clara e objetiva.

# Referências

ABBAS, G.; YING, Z.; AYOUBI, M. Consensus-driven feature selection for transparent and robust loan default prediction. *Scientific Reports*, v. 16, n. 1496, jan. 2026. DOI: [10.1038/s41598-025-31468-2](https://doi.org/10.1038/s41598-025-31468-2).

AKINJOLE, A.; SHOBAYO, O.; POPOOLA, J.; OKOYEIGBO, O.; OGUNLEYE, B. Ensemble-Based Machine Learning Algorithm for Loan Default Risk Prediction. *Mathematics*, v. 12, n. 21, p. 3423, out. 2024. DOI: [10.3390/math12213423](https://doi.org/10.3390/math12213423).

JIN, L.; WU, Z.; ZHAO, J. Prediction of Credit Card Defaulters Based on SMOTE-XGBoost Model. In: *WHICEB 2022 Proceedings*, 2022. Disponível em: [https://aisel.aisnet.org/whiceb2022/64/](https://aisel.aisnet.org/whiceb2022/64/).

LIN, C.; QIAO, N.; ZHANG, W.; LI, Y.; MA, S. Default risk prediction and feature extraction using a penalized deep neural network. *Statistics and Computing*, v. 32, n. 76, set. 2022. DOI: [10.1007/s11222-022-10140-z](https://doi.org/10.1007/s11222-022-10140-z).

ZHANG, X.; ZHANG, T.; HOU, L.; LIU, X.; GUO, Z.; TIAN, Y.; LIU, Y. Data-Driven Loan Default Prediction: A Machine Learning Approach for Enhancing Business Process Management. *Systems*, v. 13, n. 7, p. 581, jul. 2025. DOI: [10.3390/systems13070581](https://doi.org/10.3390/systems13070581).
