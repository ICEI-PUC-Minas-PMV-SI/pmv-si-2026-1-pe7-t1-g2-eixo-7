# Relatório Técnico: Modelagem Preditiva de Risco de Crédito (Loan Default)

Este documento detalha o ciclo completo de desenvolvimento, validação e comparação de modelos preditivos aplicados à detecção de inadimplência de crédito, utilizando dados históricos da operação de 2019.

---

## 1. Preparação dos Dados (Data Preparation)

A etapa de tratamento de dados foi estruturada em duas fases complementares: um saneamento básico inicial e um fluxo automatizado encapsulado via pipelines de Machine Learning para evitar o vazamento de dados (*data leakage*).

### Remoção de Outliers e Dados Corrompidos
Antes de qualquer partição amostral, aplicaram-se filtros de consistência de negócios para mitigar ruídos e incoerências que comprometeriam o aprendizado dos estimadores:
* **Renda Positiva:** Filtrou-se a base para garantir apenas rendas estritamente positivas (`income > 0`), eliminando registros zerados ou negativos.
* **Validação de Endividamento:** Assegurou-se que o índice dívida/renda fosse válido (`dtir1 >= 0`), preservando os valores nulos nesta fase para tratamento estatístico posterior.

### Tratamento de Valores Ausentes (Imputação)
Para preservar o volume amostral e evitar vieses por descarte de linhas, adotou-se uma abordagem híbrida de imputação:
* **Imputação Estática Pré-Pipeline:** Atributos cujo preenchimento direto não distorceria as distribuições populacionais receberam tratamento fixo:
    * **Moda (Variáveis Categóricas/Discretas):** Aplicada em `loan_limit`, `approv_in_adv`, `loan_purpose`, `Neg_ammortization` e `submission_of_application`.
    * **Mediana (Variáveis Numéricas):** Aplicada em `term` e `age` para neutralizar a influência de valores extremos.
* **Imputação Avançada Iterativa (MICE):** Para variáveis de alto impacto preditivo (`income`, `dtir1` e `Credit_Score`), integrou-se o `IterativeImputer` acoplado a um estimador `RandomForestRegressor(n_estimators=10)` ao pipeline principal. O método estima os valores ausentes com base no comportamento cruzado das demais features, mantendo a consistência estatística interna do dataset.

---

## 2. Engenharia de Atributos e Arquitetura dos Pipelines

Para garantir uma comparação justa e aderente aos padrões de mercado, o conjunto de dados foi dividido estritamente na proporção de **80% para treinamento** e **20% para teste definitivo**. A criação de uma partição secundária estática de validação foi descartada, optando-se por utilizar a Validação Cruzada (*Cross-Validation*) rotativa durante a busca de hiperparâmetros, maximizando a volumetria disponível para o aprendizado dos algoritmos.

### 2.1. Pipeline da Regressão Logística (Logistic Regression)
Atuando como o *baseline* estatístico clássico do projeto, este modelo linear exige transformações rigorosas de escala e codificação:
* **Padronização Numérica:** Aplicação de `StandardScaler()` em `income`, `dtir1` e `Credit_Score` para garantir média zero e variância unitária, permitindo a convergência estável do otimizador geométrico (`solver='saga'`).
* **Codificação Categórica:** Uso de `OneHotEncoder(drop='first')` para neutralizar a armadilha da multicolinearidade perfeita (*dummy variable trap*) através do descarte da primeira categoria.
* **Seleção de Atributos:** Remoção manual de identificadores (`ID`, `year`) e de variáveis que gerariam *data leakage* por revelarem o andamento posterior do crédito (`rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`, `property_value`, `LTV`, `processo_interrompido`).
* **Sintonia via GridSearchCV:** O espaço de busca priorizou a otimização da métrica PR-AUC sob validação cruzada em 3 dobras (`cv=3`), atingindo os seguintes parâmetros ótimos: `C: 10.0`, `penalty: 'l1'` (Lasso) e `class_weight: None`.

### 2.2. Pipeline do CatBoost Classifier
Implementado como o representante dos modelos de *Gradient Boosting* sequencial, este algoritmo foi desenhado para processar estruturas tabulares complexas:
* **Tratamento de Strings:** Fora do pipeline, as variáveis categóricas foram convertidas explicitamente para o tipo `string` e seus valores ausentes unificados sob a tag `'missing'`.
* **Mapeamento Nativo por Índices:** O pipeline foi configurado para rastrear a posição posicional das colunas pós-processamento, alimentando o parâmetro `cat_features` com a lista de índices (3 a 25) para que o estimador aplicasse seu algoritmo interno de *Target Encoding* ordenado, dispensando o One-Hot Encoding.
* **Sintonia via RandomizedSearchCV:** Conduzida sob 50 iterações aleatórias com `cv=3` focando em PR-AUC. A combinação campeã resultou em: `iterations: 300`, `learning_rate: 0.05`, `depth: 8`, `l2_leaf_reg: 5` e `auto_class_weights: 'SqrtBalanced'`.

---

## 3. Avaliação de Resultados e Métricas de Desempenho

### 3.1. Justificativa para a Escolha da Métrica Principal: PR-AUC
O conjunto de teste deste projeto exibe um desbalanceamento nativo de classes, composto por 22.405 registros de bons pagadores (`Classe 0`) e 7.077 registros de inadimplentes (`Classe 1`) — fixando a taxa de risco basal em aproximadamente 24%.

Em cenários desbalanceados, a Acurácia Geral torna-se uma métrica enviesada, pois um classificador ingênuo que aprovasse todas as propostas obteria 76% de acerto, falhando completamente na identificação do risco. A curva ROC-AUC também pode projetar um otimismo artificial inflado pela abundância de verdadeiros negativos. 

Por isso, a **Área sob a Curva Precisão-Recall (PR-AUC)** foi definida como a métrica soberana. Ela avalia exclusivamente o desempenho do algoritmo sobre a classe minoritária de risco, ponderando a Precisão (certeza ao acusar um calote) e o Recall (capacidade de capturar os caloteiros) ao longo de todos os limiares de decisão possíveis.

### 3.2. Tabela Comparativa de Desempenho (Conjunto de Teste)

| Métrica de Avaliação | Regressão Logística | Random Forest | CatBoost Classifier |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Métrica Alvo)** | 0.7397 | 0.7948 | **0.8030** |
| **F1-Score (Classe 1)** | 0.6200 | 0.6800 | **0.7100** |
| **Recall (Classe 1)** | 0.4700 | 0.5300 | **0.6000** |
| **Precision (Classe 1)** | **0.9300** | **0.9300** | 0.8500 |
| **Acurácia Geral** | 0.8600 | **0.8800** | **0.8800** |

---

## 4. Análise Comparativa: Desempenho de Treino versus Teste

A tabela abaixo confronta as métricas obtidas no conjunto de treinamento (117.928 registros) contra o conjunto de teste isolado (29.482 registros), permitindo auditar a capacidade de generalização e a presença de sobreajuste (*overfitting*).

| Algoritmo | Base | PR-AUC | F1-Score | Recall | Precision | Acurácia Geral |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística** | Treino<br>Teste | 0.7435<br>0.7397 | 0.6200<br>0.6200 | 0.4700<br>0.4700 | 0.9400<br>0.9300 | 0.8600<br>0.8600 |
| **Random Forest** | Treino<br>Teste | 0.8905<br>0.7948 | 0.7300<br>0.6800 | 0.5900<br>0.5300 | 0.9800<br>0.9300 | 0.9000<br>0.8800 |
| **CatBoost Classifier** | Treino<br>Teste | 0.8181<br>0.8030 | 0.7200<br>0.7100 | 0.6100<br>0.6000 | 0.8700<br>0.8500 | 0.8900<br>0.8800 |

### Diagnóstico Técnico de Consistência
* **Regressão Logística (Convergência Estável):** Apresentou um gap absoluto de PR-AUC de apenas 0.0038. A estabilidade total indica que o modelo atingiu seu limite de representação linear estável (*underfitting* estrutural relativo), operando de forma previsível, mas incapaz de capturar interações complexas.
* **Random Forest (Evidência de Overfitting):** Registrou a maior degradação de performance entre as bases, com uma queda de 0.0957 na PR-AUC e recuo de 6 pontos em Recall. Sua arquitetura de árvores independentes profundas (`max_depth=20`) tendeu a memorizar ruídos específicos da base de treino, reduzindo sua eficiência em dados inéditos.
* **CatBoost Classifier (Generalização Ideal):** Demonstrou excelente equilíbrio com um gap contido de PR-AUC de 0.0151 e variações marginais de F1-Score e Recall. Os resultados comprovam que as estruturas de regularização L2 e o uso de árvores simétricas contiveram a variância de forma eficaz.

---

## 5. Análise de Importância de Variáveis: Individual e Conjunta

### Random Forest Classifier
<img width="1101" height="785" alt="image" src="https://github.com/user-attachments/assets/92519349-1573-4173-a5a2-8a2ba91799ab" />

* **Comportamento:** O modelo distribui seu aprendizado concentrando **47.63%** da importância na variável categórica isolada `credit_type_EQUI`. Os pesos secundários dividem-se entre o prazo do contrato (`term`: 9.12%), o valor financiado (`loan_amount`: 5.79%) e os indicadores do cliente (`Credit_Score`: 4.11%; `income`: 3.35%).

### Regressão Logística (Coeficientes)
<img width="1142" height="807" alt="image" src="https://github.com/user-attachments/assets/f78d9a46-a67b-4e0a-b7ef-09d4a129da69" />

* **Comportamento:** O modelo linear explicita a direção do risco. A categoria `credit_type_EQUI` desponta com o maior coeficiente positivo (**+10.665**), seguida por `Secured_by_land` (+3.074), atuando como os principais impulsionadores de calote. Em contrapartida, as variáveis `lump_sum_payment_not_lpsm` (-2.629) e `Neg_ammortization_not_neg` (-1.108) agem como os fatores protetores mais intensos. A regularização L1 minimizou a força linear dos atributos numéricos contínuos.

### CatBoost Classifier
<img width="1107" height="797" alt="image" src="https://github.com/user-attachments/assets/ee7a6842-819e-4e33-aac6-2a9195dcd230" />

* **Comportamento:** Ao avaliar os atributos sem fracionamento macro, o CatBoost concentra **71.26%** da relevância preditiva na variável mãe `credit_type`. O modelo utiliza o índice de endividamento (`dtir1`: 4.49%), o tipo de amortização (`Neg_ammortization`: 3.19%) e a renda (`income`: 3.01%) apenas como fatores de calibração periférica.

### Síntese de Negócio da Importância de Variáveis
* **Consenso Absoluto:** Independentemente da abordagem matemática adotada (linear, bagging ou boosting), a modalidade de crédito do tipo **EQUI** é identificada de forma unânime como o principal gatilho de risco e inadimplência da carteira.
* **Mapeamento Contínuo:** Os modelos de árvore conferem maior relevância a variáveis como `income` e `dtir1` do que a Regressão Logística. Isso decorre da capacidade dos modelos não-lineares de extrair valor dessas métricas por meio de cortes repetidos e interações dinâmicas, enquanto o modelo linear fica engessado a uma taxa de impacto fixa.

---

## 6. Análise Gráfica 

### 6.1. Análise da Curva Precisão-Recall Combinada
<img width="1187" height="708" alt="image" src="https://github.com/user-attachments/assets/d0e708ab-df84-4de6-b6ff-97193d1c35d5" />

A avaliação visual da curva PR ratifica a distribuição e consistência dos modelos:
* **Convergência Inicial (Recall de 0.0 a 0.42):** Todos os estimadores operam com Precisão de 1.0 (100%), capturando apenas os inadimplentes de risco óbvio e incontestável.
* **Quebra de Linearidade (Recall > 0.42):** A Regressão Logística sofre o decaimento mais acentuado do gráfico devido à incapacidade geométrica de contornar subgrupos ambíguos.
* **Região Operacional Crítica (Recall de 0.45 a 0.85):** O **CatBoost estabelece-se como o envelope superior do gráfico**. Na marca de 0.60 de Recall, por exemplo, o CatBoost sustenta 85% de precisão, superando o Random Forest (80%) e a Regressão Logística (63%). A maior integral de área (PR-AUC = 0.8030) valida estatisticamente o melhor ordenamento das probabilidades do modelo de *boosting*.

### 6.2. Análise Técnica do Gráfico de Ganho Cumulativo
<img width="1137" height="698" alt="image" src="https://github.com/user-attachments/assets/cc8c755a-0809-4271-9959-981f8a85d6ad" />

O gráfico de ganho traduz o poder preditivo em eficiência financeira de carteira:
* **Decil de Risco de 20%:** Ao isolar a quinta parte mais perigosa indicada pelos scores, o CatBoost e o Random Forest conseguem reter **65% de toda a inadimplência real** da base (contra apenas 20% de uma escolha aleatória).
* **Decil de Risco de 40%:** Ao atingir os 40% de maior risco, o CatBoost estabiliza a captura em **82% dos inadimplentes totais**. 
* **Otimização Operacional:** A curvatura acentuada do CatBoost prova uma alta densidade de ordenamento, permitindo que a mesa de crédito automatizada concentre seus esforços de restrição ou auditoria em menos da metade da base amostral, mitigando mais de quatro quintos das perdas financeiras por inadimplência.

---

## 7. Seleção e Justificativa do Modelo Campeão: CatBoost

Considerando os requisitos de implementação técnica e governança de risco de crédito, o **CatBoost Classifier foi selecionado como o modelo campeão do projeto**.

### Justificativa Técnica e Estatística
Sob a métrica alvo de PR-AUC em ambiente de teste, o CatBoost obteve o patamar de **0.8030**, superando o Random Forest (0.7948) e a Regressão Logística (0.7397). Além disso, a auditoria de consistência provou que o CatBoost possui a maior estabilidade estatística do experimento, mantendo um gap quase nulo de performance entre treino e teste (0.0151), o que blinda a operação contra surpresas de degradação preditiva em produção (*overfitting*).

### Justificativa de Negócio e Gestão de Risco
A escolha consolida-se na gestão do *trade-off* financeiro entre Precisão e Recall. A Regressão Logística e o Random Forest priorizaram uma precisão extrema de 93% no teste, porém ao custo de deixar passar, respectivamente, 53% e 47% de calotes reais (baixo Recall). 

No mercado de crédito, o custo financeiro de um **Falso Negativo** (aprovar um proponente inadimplente) é severamente mais destrutivo do que o de um **Falso Positivo** (negar crédito a um bom pagador), pois o calote consome diretamente o capital principal do caixa da instituição. 

Ao aceitar recuar estrategicamente a precisão para 85%, o CatBoost expandiu a capacidade de captura para **60% de Recall** (7% a mais que o Random Forest e 13% a mais que a Regressão Logística). Esse comportamento focado na contenção de perdas maximiza o ROI da operação de crédito, convertendo a inteligência de dados em proteção direta ao patrimônio financeiro do banco.

# Revisão do pipeline de pesquisa e análise de dados

Este documento apresenta uma avaliação crítica da evolução do nosso pipeline de ciência de dados, contrastando a arquitetura inicial (Etapa 03) com o pipeline otimizado atual. O objetivo é demonstrar como os ajustes estruturais implementados mitigaram riscos estatísticos graves e tornaram a esteira de modelagem modular, escalável e generalizável para qualquer problema de aprendizado de máquina tabular.

---

## 1. Resumo Comparativo das Alterações Arquiteturais

A tabela abaixo sintetiza as transformações realizadas no fluxo de engenharia e modelagem:

| Fase do Pipeline | Arquitetura Antiga (Etapa 03) | Nova Arquitetura Otimizada | Motivo da Mudança |
| :--- | :--- | :--- | :--- |
| **Saneamento de Variáveis** | Remoção reativa de colunas ao longo do script. | Exclusão global e centralizada na entrada dos dados. | Prevenção de anomalias de execução e organização do fluxo. |
| **Tratamento de Nulos** | Imputação estática e MICE executados globalmente antes do split. | Encapsulamento do MICE em um `ColumnTransformer` pós-split. | Eliminação completa de vazamento de dados (*Data Leakage*). |
| **Codificação de Categóricas** | `pd.get_dummies` aplicado na base inteira de forma isolada. | Processamento integrado ao fluxo ou tratamento estruturado de strings. | Reprodutibilidade e compatibilidade com novos fluxos de entrada. |
| **Estratégia de Validação** | Divisão simples treino/teste (*Holdout*) sem validação interna. | Validação Cruzada Estratificada de 3 dobras (`cv=3`). | Redução da variância da métrica alvo e maior estabilidade estatística. |
| **Ajuste de Parâmetros** | Treinamento com hiperparâmetros fixos (padrão de fábrica). | Otimização automatizada via `RandomizedSearchCV` (30 iterações). | Busca sistemática pela melhor capacidade de generalização do modelo. |

<img width="490" height="294" alt="image" src="https://github.com/user-attachments/assets/fae731d1-410b-4bb4-ba98-8dcb0fcc3225" />
Pipeline atualizado

---

## 2. Análise Crítica por Fase do Pipeline

### 2.1. Preparação de Dados e Mitigação de Vazamento (*Data Leakage*)

#### O Problema Estrutural Antigo:
No pipeline original, o método `IterativeImputer(MICE)` era instanciado e aplicado através de um `.fit_transform()` diretamente sobre a base de dados unificada (`df_clean`), antes da execução do `train_test_split`. 

* **Impacto Estatístico:** Esta prática introduz uma falha metodológica grave conhecida como *Data Leakage* (Vazamento de Dados). Ao calcular as relações estatísticas e preencher os valores nulos utilizando o dataset inteiro, o estimador do MICE "observou" as médias, variâncias e distribuições que pertenciam exclusivamente ao conjunto de teste. Consequentemente, o conjunto de treino foi corrompido com informações do futuro, gerando métricas de performance falsamente otimistas no desenvolvimento.

#### A Solução Implementada:
O novo pipeline realiza a separação estrita dos dados em treino (80%) e teste (20%) na largada do script. O `IterativeImputer` foi encapsulado dentro do objeto `ColumnTransformer`. 

* **Justificativa:** Dessa forma, o aprendizado da imputação (`.fit()`) ocorre estritamente sobre a partição de treino de cada dobra da validação cruzada. Os dados de teste permanecem completamente isolados (sala escura), recebendo apenas a aplicação das regras aprendidas (`.transform()`), garantindo a legitimidade científica dos resultados.

### 2.2. Gestão de Atributos Futuros e Visão de Negócio

#### O Problema Estrutural Antigo:
O descarte de variáveis com vazamento ou irrelevantes era feito de forma pulverizada e tardia. Colunas como `processo_interrompido` eram mantidas durante a imputação e removidas manualmente momentos antes do ajuste do estimador.

#### A Solução Implementada:
A nova arquitetura institui uma lista de exclusão global na primeira etapa do script (`colunas_para_remover`), eliminando de imediato atributos como `rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`, `property_value` e `LTV`.

* **Justificativa:** Em um cenário de produção real em uma instituição financeira, essas informações não estão disponíveis no momento em que o proponente preenche o cadastro na mesa de crédito (são atributos gerados após a análise ou andamento da proposta). Mantê-los no treinamento geraria um modelo incapaz de operar na triagem inicial de risco.

### 2.3. Transição do Modelo Estático para a Experimentação Automatizada

#### O Problema Estrutural Antigo:
O modelo anterior ajustava um único estimador `RandomForestClassifier` com parâmetros fixos, avaliando-o uma única vez contra a base de teste. O fluxo era rígido, impossibilitando testar novas combinações ou garantir que o algoritmo não estivesse operando em subotimização.

#### A Solução Implementada:
Substituiu-se o ajuste direto pela integração com o `RandomizedSearchCV`, avaliando 30 combinações distintas de parâmetros livres (como `n_estimators`, `max_depth`, `min_samples_split` e `class_weight`) sob o critério da métrica **PR-AUC** (`scoring='average_precision'`).

* **Justificativa:** A busca sistemática e automatizada transforma o pipeline em um arcabouço flexível. Se o volume de dados aumentar ou se o dataset for substituído por outra aplicação de negócios, o pipeline se reajusta sozinho para encontrar a nova fronteira de decisão ótima através da rotação da validação cruzada.

---

## 3. Generalização do Arcabouço para Outros Cenários

Com as mudanças realizadas, o novo pipeline tornou-se **agnóstico ao problema**, transformando-se em uma estrutura modular que atende aos requisitos de generalização exigidos:

1. **Modularidade via Pipelines e Transformers:** Toda a inteligência de processamento está amarrada aos objetos `Pipeline` e `ColumnTransformer`. Se o tipo de dado mudar de numérico para texto estruturado, basta substituir o `IterativeImputer` por um codificador adequado (como `OneHotEncoder` ou `TargetEncoder`), mantendo o restante da estrutura de busca e avaliação intacta.
2. **Independência de Algoritmo:** A arquitetura aceita a substituição do estimador final de forma transparente. O mesmo fluxo desenvolvido para o Random Forest foi replicado para a Regressão Logística e para o CatBoost Classifier, alterando apenas o dicionário de parâmetros de busca (`param_dist`).
3. **Robustez Métrica Confiável:** Ao fixar a validação cruzada avaliando simultaneamente o comportamento de Treino e Teste (conforme documentado na nossa análise de consistência preditiva), o pipeline valida a si mesmo de forma contínua, emitindo alertas numéricos de *overfitting* através do monitoramento do gap absoluto de performance.



