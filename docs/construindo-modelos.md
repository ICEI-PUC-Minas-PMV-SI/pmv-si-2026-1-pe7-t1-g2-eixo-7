# Preparação dos dados

A etapa de limpeza foi dividida em duas fases estrategicamente posicionadas: uma etapa prévia de saneamento básico e uma etapa automatizada integrada diretamente aos pipelines de Machine Learning.

### Remoção de Outliers e Dados Corrompidos
Antes de qualquer divisão de dados, aplicamos filtros de consistência de negócios para remover ruídos e registros corrompidos que prejudicariam o aprendizado do modelo linear:
* Filtramos a base para garantir apenas rendas estritamente positivas (`income > 0`), eliminando registros zerados ou negativos incoerentes.
* Garantimos que o índice de divida/renda fosse válido (`dtir1 >= 0`), preservando valores nulos para tratamento posterior via MICE.

### Tratamento de Valores Ausentes (Imputação)
O dataset apresentava uma quantidade significativa de dados faltantes em colunas cruciais. Para não perder registros valiosos descartando linhas, adotamos uma abordagem híbrida:

* **Imputação Estática Pré-Pipeline:** Para atributos cujo preenchimento simples não distorceria a distribuição, aplicamos regras diretas:
    * **Moda (Variáveis Categóricas/Discretas):** Aplicada em `loan_limit`, `approv_in_adv`, `loan_purpose`, `Neg_ammortization` e `submission_of_application` para preencher as lacunas com a ocorrência mais frequente do banco de dados.
    * **Mediana (Variáveis Numéricas):** Aplicada em `term` e `age` para mitigar o impacto de possíveis distorções causadas por valores extremos.
* **Imputação Avançada Iterativa (MICE):** Para as variáveis numéricas de alto impacto comercial (`income`, `dtir1` e `Credit_Score`), integramos o `IterativeImputer` com um estimador `RandomForestRegressor(n_estimators=10)` ao pipeline. Esse método prevê os valores nulos de uma coluna baseando-se no comportamento das outras variáveis, mantendo a consistência estatística interna dos dados.


---
# Regressão Logística

## 1. Transformação de Dados

A Regressão Logística baseia-se em pesos matemáticos aplicados a matrizes e, por isso, é extremamente sensível à escala dos dados e à presença de textos brutas. Duas transformações críticas foram encapsuladas no `ColumnTransformer`:

### Padronização de Atributos Numéricos
As variáveis numéricas (`income`, `dtir1`, `Credit_Score`) possuem ordens de grandeza massivamente diferentes (ex: score de crédito vai até 900, enquanto a renda pode atingir milhares). 
* Aplicamos o `StandardScaler()`, que centraliza os dados na média zero e escala para variância unitária. 
* Essa etapa garantiu que o otimizador geométrico (`solver='saga'`) convergisse sem quebrar e que as penalizações normativas (`l1` e `l2`) agissem de forma justa sobre todos os coeficientes.

### Codificação de Variáveis Categóricas
Para converter os atributos textuais originais em representações numéricas computáveis pelo modelo, adotamos o `OneHotEncoder(drop='first')`:
* A configuração `drop='first'` remove a primeira categoria gerada de cada variável (*dummy variable trap*), eliminando problemas críticos de **multicolinearidade perfeita**, que invalidariam os coeficientes de uma Regressão Logística.
* O uso do parâmetro `handle_unknown='ignore'` blinda o pipeline em ambiente de produção contra possíveis categorias inéditas que apareçam em dados futuros.

---

## 2. Feature Engineering e Seleção de Atributos

A seleção de características focou rigorosamente na eliminação de **Data Leakage** (vazamento de dados do futuro) e no descarte de variáveis irrelevantes para a tomada de decisão de risco.

### Descarte de Atributos Irrelevantes ou Vazados
Removemos da matriz de recursos $X$ os seguintes componentes:
* `ID` e `year`: Identificadores únicos e marcas temporais fixas que não possuem poder preditivo comportamental.
* `rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`, `property_value`, `LTV` e `processo_interrompido`: Colunas que só são preenchidas após a aprovação/andamento efetivo do crédito ou que revelam diretamente o encerramento do contrato. Mantê-las causaria um vazamento artificial de performance, tornando o modelo inútil para prever o risco na mesa de entrada.

A seleção final concentrou-se estritamente nas informações cadastrais, de perfil financeiro e na política interna solicitada pelo cliente na contratação do financiamento.

---

## 3. Tratamento de Dados Desbalanceados

O problema de *Loan Default* apresenta um desbalanceamento natural (a classe de inadimplentes `1` é substancialmente menor que a classe de bons pagadores `0`). Se ignorado, o modelo focaria em maximizar a acurácia geral simplesmente ignorando os caloteiros.

Para solucionar este comportamento sem a necessidade de criar dados sintéticos artificiais fora do pipeline, adotamos a estratégia de **Pesos de Classe Nativos**:
* Injetamos o parâmetro `class_weight='balanced'` diretamente no motor do `LogisticRegression`. 
* O algoritmo calcula automaticamente penalidades inversamente proporcionais às frequências das classes. Na prática, errar um cliente inadimplente custa muito mais caro para a função de perda do modelo do que errar um bom pagador, forçando a Regressão Logística a encontrar o equilíbrio ideal.
* Toda a validação de hiperparâmetros no `GridSearchCV` foi configurada para maximizar a **PR-AUC** (`scoring='average_precision'`) através de uma **Validação Cruzada de 3 dobras (cv=3)**, garantindo que o ajuste fino dos parâmetros `C` e `penalty` priorizasse a precisão e o recall da classe de risco.
* A divisão entre os dados de treino e teste foi fixada estritamente na proporção de 80/20. A criação de uma terceira partição isolada para validação foi descartada por se tornar redundante devido à implementação da Validação Cruzada (Cross-Validation) durante o ajuste de hiperparâmetros. Julgou-se que expandir o volume de dados disponível para o aprendizado (garantindo 80% da base para o treino) traria maior robustez estatística aos modelos do que reduzir essa amostragem para sustentar um lote de validação estático.



# Descrição dos modelo Regressão Logística (Logistic Regression)

A Regressão Logística foi selecionada como o primeiro modelo preditivo deste projeto, atuando como o nosso pilar estatístico clássico e *baseline* fundamental de comparação para os modelos baseados em árvores.

### Conceitos Fundamentais e Princípio de Funcionamento
Ao contrário da Regressão Linear, que tenta prever valores contínuos, a Regressão Logística é desenhada para problemas de classificação binária (onde o resultado $Y$ assume valor 0 ou 1). O algoritmo funciona estabelecendo uma relação linear entre as variáveis de entrada e o **logaritmo da chance** (*log-odds*) do evento de interesse ocorrer (neste caso, a inadimplência, onde $Status = 1$).

O modelo estabelece uma fronteira de decisão linear na matriz espacial dos dados, separando os proponentes de crédito de acordo com a probabilidade calculada.

---

### Vantagens e Limitações

#### Vantagens:
* **Alta Interpretabilidade:** Cada variável possui um coeficiente associado. Isso permite que a diretoria e os auditores do banco entendam exatamente o peso e o impacto direto de cada atributo (como o aumento do DTI) no risco final do cliente.
* **Saída Probabilística Direta:** O modelo não entrega apenas um veredito "sim" ou "não", mas sim uma probabilidade contínua, permitindo a perfeita implementação do nosso **Sistema de Farol**.
* **Eficiência Computacional:** É extremamente leve e rápida para treinar e executar em ambientes de produção.

#### Limitações:
* **Premissa de Linearidade:** Assume que a relação entre as variáveis independentes e o log-odds do risco é linear, falhando em capturar interações complexas e cruzamentos de dados não-lineares sem engenharia de recursos prévia.
* **Sensibilidade a Outliers e Escala:** Valores discrepantes ou escalas massivamente diferentes distorcem completamente os coeficientes, exigindo o uso obrigatório de padronização (`StandardScaler`).

---

### Justificativa da Escolha do Algoritmo
No mercado financeiro e no setor bancário, a Regressão Logística é o **padrão ouro regulatório** exigido por órgãos de fiscalização (como os comitês de Basileia). A exigência de transparência em modelos de crédito (*explainable AI*) torna a Regressão Logística indispensável. Ninguém pode ter o crédito negado por uma "caixa preta" sem uma justificativa clara. 

Portanto, a escolha do grupo baseou-se em garantir conformidade com as práticas reais do mercado de risco de crédito, servindo como o benchmark ideal de interpretabilidade.

---

### Ajuste de Parâmetros Livres e Registro de Experimentos

Para extrair a máxima eficiência do modelo linear, estruturamos um pipeline integrado ao `GridSearchCV` executando uma **Validação Cruzada de 3 dobras (cv=3)** focada exclusivamente em otimizar a **PR-AUC** (`average_precision`). 

O espaço de busca foi configurado com as seguintes variações estruturais:
* **`modelo__C` (Inverso da força de regularização):** Testamos `[0.01, 0.1, 1.0, 10.0]`. Valores menores adicionam uma penalização severa para evitar Overfitting; valores maiores dão mais liberdade ao modelo.
* **`modelo__penalty` (Tipo de norma de penalização):** Testamos `['l1', 'l2']`. 
    * O `l2` (Ridge) encolhe os coeficientes sem zerá-los.
    * O `l1` (Lasso) força coeficientes de variáveis irrelevantes a zero, realizando uma seleção natural de atributos.
* **`modelo__class_weight` (Peso das classes):** Testamos `['balanced', None]` para avaliar se a compensação matemática do desbalanceamento ajudaria ou distorceria a métrica de foco (PR-AUC).
* **Configuração de Suporte:** Fixamos o otimizador como `solver='saga'` (obrigatório para suportar a penalidade `l1` em conjuntos de dados volumosos) e `max_iter=1000` para garantir a convergência dos gradientes.

---

### Análise do Resultado e Parâmetros Campeões

Após a varredura completa do espaço de busca, a combinação vencedora estabelecida pelo Grid Search foi:

* **C:** `10.0`
* **penalty:** `'l1'`
* **class_weight:** `None`
* **Melhor PR-AUC Média na Validação Cruzada:** `0.7430`

#### Justificativa dos Resultados:
1. **O Triunfo da Penalidade L1 (Lasso):** A escolha do `l1` pelo Grid Search indica que o nosso dataset possui ruídos ou colunas de texto (geradas pelo `OneHotEncoder`) redundantes. O Lasso agiu de forma cirúrgica, zerando os coeficientes das variáveis irrelevantes e mantendo apenas as features que realmente explicam o calote.
2. **C = 10.0 (Regularização Fraca):** O modelo optou por um valor alto de $C$, indicando que após o tratamento correto com o MICE e o Scaler no pipeline, as variáveis preditivas apresentavam um sinal limpo e confiável, não exigindo um freio severo de regularização para evitar overfitting.
3. **class_weight = None:** Como a nossa métrica alvo foi a **PR-AUC** (que avalia a precisão e o recall de forma contínua através de thresholds), o modelo preferiu manter a distribuição de probabilidade natural da base. Forçar o peso `balanced` deslocaria a curva de calibração, prejudicando a precisão pura exigida pelo cálculo da área.

Com uma PR-AUC de **0.7430** na validação cruzada, a Regressão Logística provou que, quando implementado um pipeline com MICE, escalonamento adequado e remoção de data leakage, ela consegue atingir um patamar de performance altamente competitivo e resiliente.

# CatBoost Classifier

O CatBoost Classifier foi implementado como o modelo baseado em *Gradient Boosting* sobre árvores de decisão, projetado para mapear interações não-lineares e processar o conjunto de variáveis categóricas de forma otimizada.

### Mapeamento Estrutural e Separação de Atributos
Antes da introdução dos dados no pipeline preditivo, o conjunto de características foi segregado e tratado estruturalmente para atender às exigências de entrada do algoritmo:

1. **Atributos Numéricos:** `income`, `dtir1` e `Credit_Score` foram isolados para o tratamento de dados ausentes via imputação iterativa.
2. **Atributos Categóricos:** Identificados automaticamente via tipo de dado (`object` e `category`). Devido à restrição do CatBoost em lidar com valores nulos textuais ou tipos ambíguos na inicialização, aplicou-se um tratamento prévio fora do pipeline. Todas as variáveis categóricas foram convertidas explicitamente para o tipo `string` e os valores ausentes foram preenchidos com a categoria unificada `'missing'`.

---

### Conceitos Fundamentais e Princípio de Funcionamento
O CatBoost opera sob o princípio de *ensemble* sequencial, onde árvores de decisão são construídas de forma sucessiva para corrigir os resíduos (erros) cometidos pelas árvores anteriores. O algoritmo se diferencia por duas características fundamentais:

* **Árvores Simétricas (*Oblivious Trees*):** O modelo adota estruturas de árvores onde os mesmos critérios de divisão são replicados em todos os nós de um mesmo nível. Essa simetria atua como uma barreira natural contra o sobreajuste (*overfitting*) e acelera o tempo de processamento na fase de teste.
* **Mapeamento Nativo de Categóricas por Índices:** Ao contrário de modelos tradicionais que exigem a expansão da matriz via *One-Hot Encoding* antes do treinamento, o CatBoost processa os textos nativamente através de técnicas internas de *Target Encoding*. Para isso, o pipeline foi configurado para rastrear a posição exata dessas colunas após a reordenação promovida pelo pré-processamento, passando a lista de índices numéricos diretamente para o parâmetro `cat_features`.

---

### Integração do Pré-processamento e Estrutura do Pipeline
Para garantir a robustez metodológica, desenhou-se um pipeline autocontido que gerencia o fluxo dos dados:

* **Via Numérica:** Aplica o `IterativeImputer` (MICE) utilizando o estimador `RandomForestRegressor(n_estimators=10)` para estimar probabilisticamente os valores nulos de `income`, `dtir1` e `Credit_Score`. Como o CatBoost é invariante à escala dos dados nas suas divisões, a padronização foi dispensada nesta esteira.
* **Via Categórica:** Configurada com o parâmetro `remainder='passthrough'`, permitindo que as strings limpas passem diretamente pelo `ColumnTransformer` na forma de um DataFrame do Pandas, preservando sua estrutura original para o processamento nativo do estimador.

---

### Ajuste de Parâmetros Livres e Registro de Experimentos

O refinamento dos hiperparâmetros foi executado através do `RandomizedSearchCV`, configurado com uma Validação Cruzada de 3 dobras (`cv=3`) e um limite de 50 iterações aleatórias (`n_iter=50`). O objetivo da busca foi maximizar a área sob a curva Precisão-Recall (**PR-AUC**). O espaço amostral explorou as seguintes configurações:

* **`iterations`:** Avaliado entre 100 e 400 árvores para determinar o ponto de convergência.
* **`learning_rate`:** Testado nos valores `[0.01, 0.05, 0.1, 0.2]` para controlar a velocidade de aprendizado.
* **`depth`:** Variado entre 4 e 10 níveis de profundidade das árvores simétricas.
* **`l2_leaf_reg`:** Parâmetro de regularização L2 avaliado em `[1, 3, 5, 10]` para penalizar a complexidade das folhas.
* **`auto_class_weights`:** Testado com as opções `Balanced`, `SqrtBalanced` e `None` para balancear o impacto da classe minoritária de inadimplentes.

---

### Análise do Resultado e Parâmetros Campeões

Após a execução das 50 combinações experimentais, os parâmetros ótimos selecionados pelo algoritmo foram:

* **iterations:** 300
* **learning_rate:** 0.05
* **depth:** 8
* **l2_leaf_reg:** 5
* **auto_class_weights:** `'SqrtBalanced'`
* **cat_features:** Índices mapeados de 3 a 25.

#### Avaliação de Consistência Estatística:
* **Melhor PR-AUC Média na Validação Cruzada:** `0.8045`

O modelo final convergiu para uma estrutura de 300 árvores com profundidade moderada (8), combinada a uma taxa de aprendizado controlada (0.05) e regularização L2 fixada em 5. A escolha do peso `'SqrtBalanced'` confirmou a necessidade de penalizar os erros da classe inadimplente de forma suavizada (pela raiz quadrada da frequência inversa), o que permitiu estabilizar a oscilação entre a precisão e o recall durante as dobras da validação cruzada.


# Análise de Resultados e Comparativo de Modelos

## 1. Justificativa para a Escolha da Métrica Principal: PR-AUC

A avaliação de modelos de concessão de crédito exige a escolha de métricas alinhadas à distribuição estatística do dataset e ao impacto financeiro do negócio. O conjunto de teste deste projeto apresenta um desbalanceamento nativo de classes, composto por **22.405 registros da classe 0** (bons pagadores) e **7.077 registros da classe 1** (inadimplentes) — o que significa que aproximadamente **24%** da base representa o evento de risco.

Em cenários de desbalanceamento, métricas tradicionais como a Acurácia tornam-se enviesadas, pois um classificador ingênuo que simplesmente aprovasse todas as propostas obteria cerca de 76% de acerto, falhando completamente em identificar o risco. A curva ROC-AUC também pode apresentar um otimismo artificial, pois é influenciada pelo grande volume de verdadeiros negativos da classe majoritária.

Por essa razão, a **Área sob a Curva Precisão-Recall (PR-AUC)** foi definida como a métrica soberana para a seleção do modelo campeão. A PR-AUC avalia exclusivamente o desempenho do algoritmo na classe minoritária (Inadimplentes), ponderando a **Precisão** (a certeza do modelo ao acusar um calote) e o **Recall** (a capacidade de capturar o maior número possível de caloteiros potenciais) ao longo de todos os limiares de decisão possíveis, sem ser distorcida pela abundância de clientes regulares.

---

## 2. Análise Técnica Individual dos Modelos

### 2.1. Random Forest Classifier
O modelo baseado em *Bagging* apresentou uma **PR-AUC de 0.7948** no conjunto de teste, demonstrando uma sólida capacidade de discriminação de risco. 
* **Precisão (Classe 1):** Atingiu **93%**, o patamar mais elevado do experimento. Isto significa que quando o modelo rotula um cliente como inadimplente, a probabilidade de acerto é muito alta, minimizando a ocorrência de falsos positivos (recusar um bom cliente por engano).
* **Recall (Classe 1):** Limitou-se a **53%**. O algoritmo demonstrou um comportamento conservador, deixando de detectar quase metade (47%) dos inadimplentes reais presentes na base de teste.
* **F1-Score (Classe 1):** O equilíbrio harmônico fixou-se em **0.68**.

### 2.2. Regressão Logística
Após a adequada estruturação do pipeline — englobando a codificação por *One-Hot Encoding*, imputação via MICE e padronização por *StandardScaler* —, o modelo linear estabeleceu uma **PR-AUC de 0.7397**.
* **Precisão (Classe 1):** Alinhou-se ao Random Forest, alcançando os mesmos **93%**.
* **Recall (Classe 1):** Registrou o menor desempenho do experimento, com **47%**. A rigidez geométrica da fronteira linear impossibilitou o algoritmo de mapear os bolsões de risco não-lineares, resultando na perda de 53% dos calotes reais.
* **F1-Score (Classe 1):** Fixou-se em **0.62**, refletindo a menor eficiência global no tratamento da classe minoritária.

### 2.3. CatBoost Classifier
O algoritmo de *Gradient Boosting* sequencial obteve a maior **PR-AUC do experimento, atingindo 0.8030** no conjunto de teste.
* **Precisão (Classe 1):** Registrou **85%**. Houve uma redução controlada na certeza em comparação aos 93% dos concorrentes, aumentando sutilmente a taxa de falsos positivos.
* **Recall (Classe 1):** Expandiu-se para **60%**, o maior índice apurado no projeto. O modelo foi capaz de localizar e reter uma fatia significativamente maior do prejuízo oculto na base.
* **F1-Score (Classe 1):** Alcançou o topo da análise com **0.71**, confirmando o melhor equilíbrio entre as forças de penalização.

---

## 3. Tabela Comparativa de Desempenho

A matriz abaixo consolida os resultados obtidos pelos três estimadores no conjunto de teste definitivo:

| Métrica de Avaliação | Regressão Logística | Random Forest | CatBoost Classifier |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Métrica Alvo)** | 0.7397 | 0.7948 | **0.8030** |
| **F1-Score (Classe 1)** | 0.6200 | 0.6800 | **0.7100** |
| **Recall (Classe 1)** | 0.4700 | 0.5300 | **0.6000** |
| **Precision (Classe 1)** | **0.9300** | **0.9300** | 0.8500 |
| **Acurácia Geral** | 0.8600 | **0.8800** | **0.8800** |

---

## 4. Escolha e Justificativa da escolha do Modelo: CatBoost

Considerando a premissa de selecionar **apenas um único modelo** para integrar a esteira de produção do banco, o **CatBoost Classifier foi o escolhido como o modelo campeão do projeto**.

### Justificativa Técnica
Sob o critério estatístico estrito da PR-AUC (0.8030), o CatBoost demonstrou superioridade matemática sobre o Random Forest (0.7948) e a Regressão Logística (0.7397). O ganho na área prova que a distribuição de probabilidades gerada pelas árvores simétricas do CatBoost é mais estável e robusta ao longo de diferentes pontos de corte, permitindo uma calibração mais eficiente para o negócio.

### Justificativa de Negócio e Gestão de Risco
A decisão apoia-se na análise do *trade-off* entre Precisão e Recall aplicado à saúde financeira de uma instituição de crédito. 

A Regressão Logística e o Random Forest priorizaram uma precisão extrema (93%), mas ao custo de ignorar, respectivamente, 53% e 47% dos calotes. No cenário bancário, o custo financeiro de um **Falso Negativo** (conceder crédito a um cliente que se tornará inadimplente) é massivamente superior ao custo de um **Falso Positivo** (negar crédito a um bom pagador por suspeita de risco). O calote consome diretamente o capital principal do banco, enquanto o alarme falso representa apenas um custo de oportunidade marginal.

O CatBoost sacrificou estrategicamente 8% de precisão (reduzindo de 93% para 85%), mas converteu esse recuo em um **ganho de 7% em Recall sobre o Random Forest** e **13% sobre a Regressão Logística**. Na prática do portfólio, ao capturar 60% de toda a inadimplência real da base de teste, o CatBoost barra a saída de milhões de reais em crédito podre, mitigando a inadimplência sistêmica e maximizando o Retorno sobre o Investimento (ROI) da operação de crédito de forma significativamente mais eficaz do que os demais modelos testados.



# Preparação dos dados

Nesta etapa, deverão ser descritas todas as técnicas utilizadas para pré-processamento/tratamento dos dados.

Algumas das etapas podem estar relacionadas à:

* Limpeza de Dados: trate valores ausentes: decida como lidar com dados faltantes, seja removendo linhas, preenchendo com médias, medianas ou usando métodos mais avançados; remova _outliers_: identifique e trate valores que se desviam significativamente da maioria dos dados.

* Transformação de Dados: normalize/padronize: torne os dados comparáveis, normalizando ou padronizando os valores para uma escala específica; codifique variáveis categóricas: converta variáveis categóricas em uma forma numérica, usando técnicas como _one-hot encoding_.

* _Feature Engineering_: crie novos atributos que possam ser mais informativos para o modelo; selecione características relevantes e descarte as menos importantes.

* Tratamento de dados desbalanceados: se as classes de interesse forem desbalanceadas, considere técnicas como _oversampling_, _undersampling_ ou o uso de algoritmos que lidam naturalmente com desbalanceamento.

* Separação de dados: divida os dados em conjuntos de treinamento, validação e teste para avaliar o desempenho do modelo de maneira adequada.
  
* Manuseio de Dados Temporais: se lidar com dados temporais, considere a ordenação adequada e técnicas específicas para esse tipo de dado.
  
* Redução de Dimensionalidade: aplique técnicas como PCA (Análise de Componentes Principais) se a dimensionalidade dos dados for muito alta.

* Validação Cruzada: utilize validação cruzada para avaliar o desempenho do modelo de forma mais robusta.

* Monitoramento Contínuo: atualize e adapte o pré-processamento conforme necessário ao longo do tempo, especialmente se os dados ou as condições do problema mudarem.

* Entre outras....

Avalie quais etapas são importantes para o contexto dos dados que você está trabalhando, pois a qualidade dos dados e a eficácia do pré-processamento desempenham um papel fundamental no sucesso de modelo(s) de aprendizado de máquina. É importante entender o contexto do problema e ajustar as etapas de preparação de dados de acordo com as necessidades específicas de cada projeto.

# Descrição dos modelos

Nesta seção, conhecendo os dados e de posse dos dados preparados, é hora de descrever os outros dois algoritmos de aprendizado de máquina selecionados para a construção dos modelos propostos. Inclua informações abrangentes sobre cada algoritmo implementado, aborde conceitos fundamentais, princípios de funcionamento, vantagens/limitações e justifique a escolha de cada um dos algoritmos. 

Explore aspectos específicos, como o ajuste dos parâmetros livres de cada algoritmo. Lembre-se de experimentar parâmetros diferentes e principalmente, de justificar as escolhas realizadas e registrar todos os experimentos realizados.

# Avaliação dos modelos criados

## Métricas utilizadas

Nesta seção, as métricas utilizadas para avaliar os modelos desenvolvidos deverão ser apresentadas (p. ex.: acurácia, precisão, recall, F1-Score, MSE etc.). A escolha de cada métrica deverá ser justificada, pois esta escolha é essencial para avaliar de forma mais assertiva a qualidade do modelo construído. 

## Discussão dos resultados obtidos

Nesta seção, discuta os resultados obtidos por cada um dos modelos construídos na Etapa 03 e na Etapa 04, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade de cada um deles. Lembre-se de relacionar os resultados obtidos ao problema identificado, a questão de pesquisa levantada e estabelecer relação com os objetivos previamente propostos. Não deixe de comparar os resultados obtidos por cada modelo com os demais.

# Revisão do pipeline de pesquisa e análise de dados

Nesta etapa, os alunos devem revisar o pipeline de pesquisa e análise de dados proposto na Etapa 03, avaliando criticamente cada uma de suas etapas, fluxos e decisões. O objetivo agora é identificar possíveis ajustes, melhorias ou generalizações que tornem o pipeline mais abrangente e adaptável, de forma que ele seja capaz de representar qualquer processo de construção de sistemas de aprendizado de máquina – independentemente da área de aplicação, tipo de dado ou técnica utilizada.

Lembre-se de que um pipeline bem estruturado deve contemplar, de forma flexível e modular, as principais fases da pesquisa e experimentação em ciência de dados e aprendizado de máquina, incluindo (mas não se limitando a): formulação do problema, coleta e preparação dos dados, análise exploratória, definição de métricas, seleção e validação de modelos, interpretação dos resultados e documentação.

O resultado desta etapa deverá ser um pipeline revisado e justificado, acompanhado de uma breve descrição das alterações realizadas e dos motivos que levaram a cada mudança.

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".
