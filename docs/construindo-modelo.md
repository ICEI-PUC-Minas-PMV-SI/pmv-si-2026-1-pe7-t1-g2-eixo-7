# Preparação dos dados

O tratamento dos dados visou construir uma base que fosse limpa, consistente e adequada ao algoritmo de machine learning que seria utilizado. As escolhas feitas foram guiadas pelo diagnóstico prévio realizado nos dados, principalmente no que diz respeito aos valores ausentes, que revelaram padrões significativos para a questão da inadimplência.

## 1. Diagnóstico inicial e análise dos dados ausentes

Antes de iniciar as transformações, foi feito um diagnóstico aprofundado da qualidade dos dados, utilizando `df.isnull().sum()` e a biblioteca `missingno` para visualizar o padrão de ausência. Em vez de tratar os nulos de forma genérica, optou-se por entender se eles **carregavam significado** em relação à variável-alvo (`Status`).

Foram conduzidas as seguintes análises:

- **Casos críticos:** identificou-se quantos registros inadimplentes (`Status=1`) possuíam mais de cinco colunas faltantes, revelando um padrão estrutural de ausência associada à inadimplência.
- **Cruzamento de nulos com inadimplência:** para colunas suspeitas (`rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`, `property_value`, `LTV`, `dtir1`, `income`), calculou-se o percentual de inadimplentes entre os registros com valor nulo, indicando que a falta de informação é, em si, um sinal preditivo.
- **Interseção de ausências:** verificou-se que `rate_of_interest`, `property_value` e `LTV` tendem a estar ausentes simultaneamente, o que sugere que esses nulos representam **processos de crédito interrompidos** antes da formalização do contrato.

Esse diagnóstico foi determinante para definir a estratégia de tratamento dos nulos, evitando descartá-los indiscriminadamente.

## 2. Limpeza de dados

A limpeza foi realizada em duas frentes:

**Filtros de coerência:** registros com `income <= 0` ou `dtir1 < 0` foram considerados inconsistentes (renda ou comprometimento de renda negativos não fazem sentido no contexto de crédito) e removidos. Os valores ausentes nessas colunas foram preservados nesta etapa, pois seriam imputados posteriormente:

```python
df_clean = df[
    ((df['income'] > 0) | (df['income'].isnull())) &
    ((df['dtir1'] >= 0) | (df['dtir1'].isnull()))
].copy()
```

**Outliers:** os boxplots e histogramas das variáveis numéricas (`loan_amount`, `LTV`, `Credit_Score`, `dtir1`) revelaram a presença de valores extremos. **Optou-se por preservar os outliers**, pois o algoritmo Random Forest é robusto a valores extremos por operar com divisões binárias baseadas em valores de corte, e não em distância. Remover outliers poderia descartar casos legítimos de risco elevado, justamente os mais informativos para um modelo de inadimplência.

## 3. Feature Engineering

Foi criada uma variável binária derivada chamada `processo_interrompido`, que sinaliza registros nos quais qualquer uma das colunas-chave do contrato (`rate_of_interest`, `property_value`, `Interest_rate_spread`, `LTV`, `Upfront_charges`) estava ausente:

```python
df_clean['processo_interrompido'] = np.where(
    (df_clean['rate_of_interest'].isnull()) |
    (df_clean['property_value'].isnull()) |
    (df_clean['Interest_rate_spread'].isnull()) |
    (df_clean['LTV'].isnull()) |
    (df_clean['Upfront_charges'].isnull()),
    1, 0
)
```

A motivação da feature foi capturar o sinal informativo da própria ausência. Posteriormente, na fase de modelagem, essa variável foi reavaliada e removida por configurar potencial _data leakage_ (informação posterior ao momento da concessão do crédito), conforme detalhado na seção de modelagem.

Adicionalmente, a variável `age`, originalmente categórica em faixas (`<25`, `25-34`, ..., `>74`), foi mapeada para uma versão numérica auxiliar (`age_num`), utilizando o ponto médio de cada faixa, para viabilizar sua participação no processo de imputação multivariada:

```python
mapeamento_idade = {
    '25-34': 30, '35-44': 40, '45-54': 50, '55-64': 60,
    '65-74': 70, '>74': 80, '<25': 20
}
```

## 4. Tratamento de valores ausentes

Foi adotada uma estratégia híbrida, com diferentes técnicas conforme a natureza e a importância da variável.

### 4.1 Imputação simples (mediana e moda)

Para variáveis com baixa proporção de valores ausentes e papel secundário no modelo, aplicou-se imputação direta:

- **Mediana** para variáveis numéricas (`term`) — escolhida em vez da média por sua robustez a outliers.
- **Moda** para variáveis categóricas (`loan_limit`, `approv_in_adv`, `loan_purpose`, `Neg_ammortization`, `age`, `submission_of_application`).

### 4.2 Imputação multivariada com MICE

Para as variáveis numéricas com maior peso preditivo e maior quantidade de nulos (`loan_amount`, `income`, `dtir1`, `Credit_Score`), foi aplicada a técnica **MICE (Multiple Imputation by Chained Equations)** através do `IterativeImputer` do scikit-learn, utilizando um `RandomForestRegressor` como estimador base:

```python
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=42),
    max_iter=8,
    random_state=42
)
```

A escolha do MICE com Random Forest foi justificada por três motivos:

1. **Preserva a estrutura multivariada dos dados:** ao imputar uma variável com base nas demais, mantém-se a coerência entre `income`, `loan_amount`, `Credit_Score` e `dtir1`, evitando o achatamento da variância gerado por imputações simples.
2. **Captura relações não lineares:** o uso do Random Forest como estimador permite modelar interações complexas entre variáveis, mais aderentes ao comportamento real dos dados financeiros.
3. **Coerência metodológica:** alinha-se ao algoritmo escolhido para a classificação final.

### 4.3 Validação pós-imputação

Após o MICE, foi feita uma análise de variância para garantir que a imputação **não tivesse achatado a distribuição** das variáveis críticas. O Coeficiente de Variação (CV) de `income` e `dtir1` foi calculado e comparado com as estatísticas pré-imputação:

```python
cv_income = (df_final['income'].std() / df_final['income'].mean()) * 100
cv_dtir = (df_final['dtir1'].std() / df_final['dtir1'].mean()) * 100
```

Adicionalmente, gráficos de densidade (KDE) por status (adimplente vs. inadimplente) foram gerados para confirmar que a imputação preservou a separação entre as classes.

## 5. Codificação de variáveis categóricas

As variáveis categóricas (`Gender`, `Region`, `loan_purpose`, `loan_type`, `occupancy_type`, `credit_type`, entre outras) foram convertidas para representação numérica via **One-Hot Encoding**, utilizando `pd.get_dummies`:

```python
X = pd.get_dummies(X)
```

Em iterações posteriores, optou-se pelo parâmetro `drop_first=True` para evitar a armadilha da multicolinearidade (variável fictícia redundante):

```python
X_encoded = pd.get_dummies(X_puro, drop_first=True)
```

A escolha do One-Hot foi adequada porque as variáveis categóricas do dataset são, em sua maioria, **nominais** (sem ordem natural), e o Random Forest não exige ordinalidade nem normalização das colunas resultantes.

## 6. Sobre normalização e padronização

**Não foi aplicada normalização ou padronização** das variáveis numéricas. Essa decisão é metodologicamente correta: algoritmos baseados em árvores, como o Random Forest, são **invariantes a transformações monotônicas de escala**. Como as divisões dos nós são feitas com base em pontos de corte de cada variável individualmente, escalar `income` ou `loan_amount` não altera as previsões. Aplicar normalização aqui seria um passo desnecessário, que apenas adicionaria complexidade ao pipeline.

## 7. Separação dos dados em treino e teste

A base final foi dividida em conjuntos de treino e teste utilizando `train_test_split` do scikit-learn, com as seguintes configurações:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Justificativa dos parâmetros:

- **`test_size=0.2`:** divisão clássica 80/20, equilibrando volume de treino e representatividade do teste.
- **`random_state=42`:** garante reprodutibilidade entre execuções.
- **`stratify=y`:** parâmetro **fundamental no contexto de inadimplência**, pois preserva a proporção original entre adimplentes e inadimplentes nos dois subconjuntos. Sem essa estratificação, há risco de o conjunto de teste ficar com proporção distinta da classe minoritária, comprometendo a avaliação.

## 8. Seleção de features

A seleção de features foi conduzida de forma iterativa, baseada na análise de `feature_importances_` do Random Forest. A partir dos resultados do modelo baseline, uma variável foram identificadas como problemáticas e removidas em testes subsequentes:

- **`processo_interrompido`:** apesar de criada com intenção legítima na engenharia de atributos, representava informação posterior ao momento da concessão do crédito, configurando vazamento temporal, informações essas de features ausentes como rate_of_interest, property_value e LTV que tinha sua totalidade de omissão relacionada ao alvo em questão .


## Resumo do pipeline de preparação

O fluxo final de preparação dos dados pode ser resumido nas seguintes etapas sequenciais:

1. Diagnóstico de nulos e análise de seu significado em relação à inadimplência;
2. Filtragem de inconsistências (`income <= 0`, `dtir1 < 0`);
3. Criação da feature `processo_interrompido` e mapeamento numérico de `age`;
4. Imputação simples (mediana/moda) para variáveis secundárias;
5. Imputação multivariada (MICE com Random Forest Regressor) para variáveis críticas;
6. Validação da preservação da variância pós-imputação;
7. Codificação one-hot das variáveis categóricas;
8. Separação treino/teste estratificada (80/20);
9. Seleção iterativa de features baseada em feature importance.



# Descrição do modelo

O algoritmo escolhido para a criação do modelo foi o random forest. Ele opera construindo um conjunto de árvores de decisão durante o treinamento e combina cada previsão para gerar uma previsão final mais robusta. Para problemas de classificação, como o de inadimplência, a saída do modelo corresponde à classe mais votada pelas árvores.

### Justificativa

A escolha foi motivada pelos seguintes pontos:

**Adequação ao tipo de dado:** o dataset de empréstimos é composto por dados tabulares mistos, tendo variáveis numéricas como loan_amount e income e variáveis categóricas, como region e gender. Esse algoritmo lida bem com essa heterogeneidade, sem necessidade de normalizar ou padronizar as variáveis numéricas.

**Robustez a outliers e ruído:** as análises exploratórias feitas usando boxplots de LTV, DTI e valor de empréstimo revelaram a existência de outliers, presente principalmente em variáveis financeiras. Tendo em vista que o random forest opera por divisões binárias baseadas em valores de corte, ele lida melhor com valores extremos do que modelos lineares.

**Capacidade de capturar relações não lineares e interações:** a análise de correlação de Pearson realizada mostrou que as variáveis individualmente apresentam correlação fraca com o status (a maior foi o LTV com r ≈ 0,13). Isso sugere que a inadimplência depende de fatores mais complexas entre variáveis (alto LTV combinado com baixo Credit Score e alto DTI, por exemplo), padrões que árvores conseguem capturar naturalmente.

**Interpretabilidade via feature importance:** mesmo sendo um modelo _ensemble_, ele fornece o atributo feature*importances*, que foi importante para identificar variáveis com sinais de vazamento de informação, como o rate_of_interest, property_value e LTV.

### Configuração e ajuste dos hiperparâmetros

O modelo foi configurado utilizando como base de testes o RandomizedSearchCV com uma grade definida afim de encontrar a otimização entre estabilidade preditiva e custo computacional com os seguintes parâmetros base:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', 'balanced_subsample', None]
}
```
Após a definição do grid de testes foi realizado a configuração da busca pelo melhores parâmetros focando na melhora do f1-score buscando um equilíbrio entre precision e recall

```python
rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=20,           
    cv=3,                
    scoring='f1',       
    verbose=2,
    random_state=42,
    n_jobs=-1
)
```
como resultado dessa busca foi encontrado os parâmetros mais otimizados para um modelo que visa o equilíbrio sendo eles: 'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 30, 'criterion': 'gini', 'class_weight': 'balanced_subsample'

- **`n_estimators=200`**: define o número de árvores na floresta. O valor 200 pode oferecer maior estabilidade preditiva e redução do erro em um dataset com um grande volume de dados pode trazer um equilíbrio centre estabilidade das previsões e custo computacional.
- **`max_depth: 30`**: Determina a profundidade máxima que cada árvore pode crescer. O limite de 30 níveis permite que o modelo aprenda relações financeiras complexas entre variáveis (como renda e taxa de dívida), mas impede que a árvore cresça indefinidamente. Isso é fundamental para evitar o overfitting
- **`min_samples_split: 10`**: É o número mínimo de registros necessários em um nó para que o modelo decida realizar uma nova divisão. Ao exigir pelo menos 10 clientes para criar uma regra, o modelo ignora ruídos e variações pequenas de dados. Isso força a floresta a focar em padrões estatísticos reais e robustos.
- **`class_weight: 'balanced_subsample'`**: Ajusta o peso atribuído a cada classe para lidar com dados desbalanceados como é o caso deste dataset que há menos inadimplentes do que pagadores
- **`min_samples_leaf: 1'`**:O número mínimo de amostras que deve sobrar em cada ponto final da árvore. O valor 1 na folha permite que o modelo aproveite ao máximo a profundidade de 30 níveis para refinar a classificação final.

### Testes realizados

Foram executados quatro experimentos com configurações distintas do conjunto de variáveis preditoras, mantendo os mesmos hiperparâmetros do modelo. O objetivo foi avaliar o impacto da remoção de variáveis suspeitas de causar _data leakage_ (vazamento de informação do alvo para as features).

| Experimento                                 | Variáveis/Parâmetros alterados                             |
| ------------------------------------------- | ------------------------------------------------ | 
| 1 — Modelo inicial            | Apenas `Status`,'processo interrompido' (alvo) removido                          |                                                                                                                                                                                                                          
| 2 — Modelo focado no F1-Score     | `Status`, 'processo interrompido' sendo removido, com os hiperparâmetros sendo n_estimators=200, min_samples_split=10, min_samples_leaf=1, max_depth=30, criterion='gini', class_weight='balanced_subsample'.                   |
| 3 —Modelo focado na precisão | `Status`, `processo_interrompido` sendo removido,  _estimators=200, min_samples_split=10, min_samples_leaf=2, max_depth=10, criterion='gini', class_weight=None,               |
| 4 — Modelo focado no Recall                              |  `Status`, 'processo interrompido' sendo removido, com os hiperparâmetros sendo n_estimators=200, min_samples_split=5, min_samples_leaf=1, max_depth=10, criterion='gini', class_weight='balanced_subsample'.      |

Analisando a necessidade do banco de manter uma boa relação entre a captação de clientes pagantes e a detecção de clientes inadimplentes (que geram prejuízo), escolheu-se o modelo focado no equilíbrio, o Modelo 2, orientado pelo F1-Score. A escolha justifica-se pois, ao focar exclusivamente no parâmetro Recall, embora ocorra um aumento na detecção de inadimplentes, o alto índice de falsos positivos gera insegurança e pode afastar bons clientes do banco. Por outro lado, caso fosse escolhido um modelo focado apenas em Precisão, o banco acabaria aceitando muitos clientes inadimplentes, o que traria um prejuízo financeiro superior à vantagem de atrair novos clientes pagantes.

### Vantagens observadas

- Flexibilidade no ajuste de hiperparâmetros: o baseline já entregou desempenho competitivo e respondeu bem ao tuning via randomizedSearchCV, permitindo otimizar o modelo para diferentes objetivos (F1, recall ou precisão)
- Treinamento paralelizável (`n_jobs=-1`), aproveitando múltiplos núcleos de CPU;
- Geração nativa de ranking de importância das variáveis;
- Resistência ao overfitting comparado a uma árvore de decisão isolada.

### Desvantagens observadas

- **Tendência inicial a favorecer a classe majoritária**: No baseline, o modelo apresentava recall reduzido para a classe minoritária (inadimplentes), justamente a de maior interesse no negócio. A limitação foi mitigada com class_weight='balanced_subsample', que elevou o recall da classe 1 para 0,60, embora ainda haja trade-off com a precisão.
- **Custo computacional:** com 100 árvores e o volume de dados após o MICE, o tempo de treinamento é mais alto que o de modelos mais simples.
- **Menor interpretabilidade individual:** apesar do `feature_importances_`, não é possível extrair regras explícitas como em uma única árvore de decisão.
- **Viés em variáveis categóricas com muitos níveis:** após pd.get_dummies, variáveis com muitos níveis tendem a inflar sua importância. O efeito foi confirmado no modelo final: credit_type_EQUI apareceu como variável mais importante (27,86%), valor desproporcional frente a dtir1 (11,56%) e income (9,85%).


# Avaliação dos modelos criados

## Métricas utilizadas

A avaliação do modelo foi conduzida utilizando um conjunto de métricas extraídas da Matriz de Confusão, fundamentais para problemas de classificação, especialmente em cenários onde as classes podem apresentar desbalanceamento (típico em bases de crédito). As métricas escolhidas e suas justificativas são

* **Precisão (Precision):** Mede a proporção de verdadeiros positivos em relação a todas as predições positivas feitas pelo modelo. Foi uma escolha essencial porque, no contexto do negócio, o custo de um Falso Positivo (classificar um bom cliente como inadimplente) é altíssimo, podendo gerar atritos, cobranças indevidas e a perda do cliente.

* **Recall / Revocação (Sensibilidade):** Avalia a proporção de casos positivos reais que o modelo conseguiu identificar corretamente. Esta métrica foi crucial para acompanhar a classe dos Adimplentes (Status 0), garantindo que a base de bons clientes fosse corretamente rastreada e preservada.

* **F1-Score:** Sendo a média harmônica entre a Precisão e o Recall, esta métrica foi utilizada para fornecer uma visão única do desempenho do modelo em cada classe, penalizando modelos que tenham uma disparidade muito grande entre Precisão e Recall.

* **Acurácia (Accuracy):** Utilizada como uma métrica de apoio para entender o percentual de acertos globais do modelo (soma dos verdadeiros positivos e verdadeiros negativos sobre o total), embora a tomada de decisão principal tenha se apoiado na Precisão e no Recall.


## Discussão dos resultados obtidos

Os resultados obtidos pelo modelo final demonstram um alinhamento com os objetivos de negócio, adotando uma postura de equilíbrio entre a proteção do capital e a experiência do cliente. O modelo, otimizado via RandomizedSearchCV com foco no F1-Score, estabelece uma régua de crédito eficiente para um banco conservador.

* **Análise das Métricas**

* Recall de Inadimplentes (0.60): O modelo demonstra uma capacidade moderada de detectar inadimplência, identificando e bloqueando 60% dos inadimplentes reais. Em termos absolutos, para um universo de 150.000 clientes, isso representa a detecção eficaz de 21.600 potenciais calotes.

* Precisão de Inadimplentes (0.84): A assertividade do modelo é alta; em 84% das vezes que o sistema emite um alerta de risco, o cliente é de fato um caso de inadimplência. 

* Recall de Adimplentes (0.96): O sistema atua com rigor na proteção dos bons pagadores, identificando corretamente 96% da base adimplente. Isso cumpre o objetivo estratégico de evitar que clientes pagantes sejam injustamente barrados por erros do sistema.

* F1-Score (0.70): Esta métrica consolida o sucesso do modelo em harmonizar a necessidade de capturar inadimplentes com a precisão das decisões.

* **Importância das Variáveis**:

* credit_type_EQUI (27.87%): Esta variável destaca-se como o principal indicador de risco. Sua alta importância sugere que o bureau de crédito Equifax atua como um filtro crítico na segmentação de perfis com maior propensão histórica à inadimplência.

* Fundamentos Financeiros (dtir1 e income): A relação dívida/renda (dtir1) com 11.56% e a renda (income) com 9.85% aparecem logo em seguida. Isso demonstra que o modelo toma decisões baseadas na capacidade real de pagamento do cliente, um pilar fundamental para o banco.

* Histórico e Montante (Credit_Score e loan_amount): O score de crédito (7.88%) e o valor do empréstimo (7.67%) completam o topo da pirâmide de decisão, garantindo que o comportamento passado e a exposição financeira atual sejam pesados de forma equilibrada.

É possível que a ausência de dados em colunas críticas tenha impactado negativamente o desempenho do modelo, especialmente na identificação de inadimplentes. Variáveis como rate_of_interest, property_value e LTV apresentam uma concentração de valores nulos extremamente alta entre os casos de inadimplência (Status 1), chegando a 100% e 99,99% em alguns campos. Como essas colunas são fundamentais para o cálculo do risco financeiro, a falta dessas informações limita a capacidade do algoritmo de aprender os padrões específicos dos clientes que deixam de pagar. Na prática, essa lacuna de dados reflete-se diretamente no parâmetro de Recall do Status 1, que poderia ser superior caso o modelo tivesse acesso a esses indicadores financeiros para mapear com maior precisão o perfil de risco desses clientes.



# Pipeline de pesquisa e análise de dados

Em pesquisa e experimentação em sistemas de informação, um pipeline de pesquisa e análise de dados refere-se a um conjunto organizado de processos e etapas que um profissional segue para realizar a coleta, preparação, análise e interpretação de dados durante a fase de pesquisa e desenvolvimento de modelos. Esse pipeline é essencial para extrair _insights_ significativos, entender a natureza dos dados e, construir modelos de aprendizado de máquina eficazes. 

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".
