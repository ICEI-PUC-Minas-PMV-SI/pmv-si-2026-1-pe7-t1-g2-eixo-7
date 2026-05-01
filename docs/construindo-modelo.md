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

# Descrição do modelo

# Descrição do modelo

O algoritmo escolhido para a criação do modelo foi o random forest. Ele opera construindo um conjunto de árvores de decisão durante o treinamento e combina cada previsão para gerar uma previsão final mais robusta. Para problemas de classificação, como o de inadimplência, a saída do modelo corresponde à classe mais votada pelas árvores.

## Justificativa

A escolha foi motivada pelos seguintes pontos:

**Adequação ao tipo de dado:** o dataset de empréstimos é composto por dados tabulares mistos, tendo variáveis numéricas como loan_amount e income e variáveis categóricas, como region e gender. Esse algoritmo lida bem com essa heterogeneidade, sem necessidade de normalizar ou padronizar as variáveis numéricas.

**Robustez a outliers e ruído:** as análises exploratórias feitas usando boxplots de LTV, DTI e valor de empréstimo revelaram a existência de outliers, presente principalmente em variáveis financeiras. Tendo em vista que o random forest opera por divisões binárias baseadas em valores de corte, ele lida melhor com valores extremos do que modelos lineares.

**Capacidade de capturar relações não lineares e interações:** a análise de correlação de Pearson realizada mostrou que as variáveis individualmente apresentam correlação fraca com o status (a maior foi o LTV com r ≈ 0,13). Isso sugere que a inadimplência depende de fatores mais complexas entre variáveis (alto LTV combinado com baixo Credit Score e alto DTI, por exemplo), padrões que árvores conseguem capturar naturalmente.

**Interpretabilidade via feature importance:** mesmo sendo um modelo _ensemble_, ele fornece o atributo feature*importances*, que foi importante para identificar variáveis com sinais de vazamento de informação, como o credit_type.

## Configuração e ajuste dos hiperparâmetros

O modelo foi configurado com os seguintes parâmetros base:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

- **`n_estimators=100`**: define o número de árvores na floresta. O valor 100 representa um equilíbrio consolidado na literatura entre estabilidade das previsões e custo computacional. Um número maior tende a estabilizar a métrica, mas com retorno decrescente.
- **`random_state=42`**: fixa a semente aleatória para garantir reprodutibilidade dos resultados entre execuções, condição fundamental para comparar diferentes configurações experimentais.
- **`n_jobs=-1`**: utiliza todos os núcleos disponíveis do processador para paralelizar o treinamento das árvores, reduzindo o tempo de execução.

Os demais hiperparâmetros foram mantidos no padrão do scikit-learn (`max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features='sqrt'`, `criterion='gini'`), permitindo que as árvores cresçam até alcançar pureza máxima nos nós folha. Essa configuração maximiza a capacidade de ajuste, ficando a regularização por conta da diversidade entre as árvores.

## Registro dos testes realizados

Foram executados quatro experimentos com configurações distintas do conjunto de variáveis preditoras, mantendo os mesmos hiperparâmetros do modelo. O objetivo foi avaliar o impacto da remoção de variáveis suspeitas de causar _data leakage_ (vazamento de informação do alvo para as features).

| Experimento                                 | Variáveis removidas                              | Justificativa                                                                                                                                                                                                                                                        |
| ------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 — Modelo completo (`rf_model`)            | Apenas `Status` (alvo)                           | Baseline com todas as features disponíveis.                                                                                                                                                                                                                          |
| 2 — Sem `credit_type` (`rf_realista`)       | `Status`, `credit_type`                          | A análise de feature importance no baseline revelou que `credit_type` dominava a previsão de forma desproporcional, sugerindo que esta variável carregava informação derivada do próprio histórico de inadimplência (_data leakage_).                                |
| 3 — Sem `processo_interrompido` (`rf_puro`) | `Status`, `processo_interrompido`                | A variável foi criada artificialmente durante o pré-processamento como flag de "processo com dados faltantes". Por estar fortemente correlacionada com inadimplência observada nas análises iniciais, configurava-se como informação posterior à decisão de crédito. |
| 4 — Sem ambas                               | `Status`, `credit_type`, `processo_interrompido` | Avaliação do desempenho real do modelo apenas com variáveis legitimamente disponíveis no momento da concessão do crédito.                                                                                                                                            |

A comparação dos relatórios de classificação (`classification_report`) entre os experimentos permitiu observar a queda esperada de métricas como precisão e recall ao remover as variáveis com leakage, o que confirma que parte do desempenho do baseline era artificial. O modelo do Experimento 4 representa, portanto, a versão mais honesta e generalizável do classificador, sendo a recomendada para aplicação prática.

## Vantagens observadas

- Bom desempenho com pouco ajuste de hiperparâmetros (configuração padrão já entregou métricas competitivas);
- Treinamento paralelizável (`n_jobs=-1`), aproveitando múltiplos núcleos de CPU;
- Geração nativa de ranking de importância das variáveis;
- Resistência ao overfitting comparado a uma árvore de decisão isolada.

## Desvantagens observadas

- **Tendência a favorecer a classe majoritária:** dado o desbalanceamento entre adimplentes (Status=0) e inadimplentes (Status=1) presente no dataset, o modelo tende a apresentar recall baixo para a classe minoritária (inadimplentes), que é justamente a classe de maior interesse no problema de negócio.
- **Custo computacional:** com 100 árvores e o volume de dados após o MICE, o tempo de treinamento é mais alto que o de modelos mais simples.
- **Menor interpretabilidade individual:** apesar do `feature_importances_`, não é possível extrair regras explícitas como em uma única árvore de decisão.
- **Possibilidade de viés em variáveis categóricas com muitos níveis:** após `pd.get_dummies`, o conjunto `X` passou a ter um número alto de colunas, e variáveis com muitas categorias podem inflar artificialmente sua importância.


# Avaliação dos modelos criados

## Métricas utilizadas

A avaliação do modelo foi conduzida utilizando um conjunto de métricas extraídas da Matriz de Confusão, fundamentais para problemas de classificação, especialmente em cenários onde as classes podem apresentar desbalanceamento (típico em bases de crédito). As métricas escolhidas e suas justificativas são

* **Precisão (Precision):** Mede a proporção de verdadeiros positivos em relação a todas as predições positivas feitas pelo modelo. Foi uma escolha essencial porque, no contexto do negócio, o custo de um Falso Positivo (classificar um bom cliente como inadimplente) é altíssimo, podendo gerar atritos, cobranças indevidas e a perda do cliente.

* **Recall / Revocação (Sensibilidade):** Avalia a proporção de casos positivos reais que o modelo conseguiu identificar corretamente. Esta métrica foi crucial para acompanhar a classe dos Adimplentes (Status 0), garantindo que a base de bons clientes fosse corretamente rastreada e preservada.

* **F1-Score:** Sendo a média harmônica entre a Precisão e o Recall, esta métrica foi utilizada para fornecer uma visão única do desempenho do modelo em cada classe, penalizando modelos que tenham uma disparidade muito grande entre Precisão e Recall.

* **Acurácia (Accuracy):** Utilizada como uma métrica de apoio para entender o percentual de acertos globais do modelo (soma dos verdadeiros positivos e verdadeiros negativos sobre o total), embora a tomada de decisão principal tenha se apoiado na Precisão e no Recall.

  

Nesta seção, as métricas utilizadas para avaliar os modelos desenvolvidos deverão ser apresentadas (p. ex.: acurácia, precisão, recall, F1-Score, MSE etc.). A escolha de cada métrica deverá ser justificada, pois esta escolha é essencial para avaliar de forma mais assertiva a qualidade do modelo construído. 

## Discussão dos resultados obtidos

Os resultados obtidos pelo modelo demonstram um alinhamento excepcional com a questão de pesquisa e com os objetivos estratégicos de negócio previamente propostos. O problema central consistia em identificar o risco de crédito sem que ações de cobrança agressivas ou negativas de crédito prejudicassem a experiência dos bons clientes (Adimplentes - Status 0).

Avaliando a qualidade do modelo pelas métricas, observa-se que a **Precisão para a classe de Inadimplentes (Status 1) foi de 0.91 (91%)**. Isso "conta" que, em 91% das vezes que o modelo aponta um cliente como mau pagador, ele está absolutamente correto. Consequentemente, a taxa de erro que resultaria em incomodar ou negar crédito a um "bom cliente" por engano é de apenas 9%, cumprindo com rigor a diretriz conservadora do negócio.

Adicionalmente, o modelo apresentou um **Recall de 0.98 (98%) para a classe de Adimplentes (Status 0).** Na prática, isso atesta que o sistema consegue identificar e proteger 98% da base de bons pagadores reais.

Como inerente a qualquer modelo de Machine Learning, existe um trade-off. Para atingir essa altíssima proteção aos bons clientes, o modelo assume uma postura mais branda nas predições de risco, o que se refletiu em um **Recall de 0.55 (55%) para a classe 1**. Isso indica que aproximadamente 45% dos inadimplentes reais não estão sendo bloqueados. Contudo, dentro do contexto prático em que os dados se inserem, este é um custo aceito (e planejado) pela estratégia de negócio: prefere-se deixar passar alguns maus pagadores a correr o risco de afastar clientes valiosos. O modelo desenvolvido é, portanto, altamente assertivo e cumpre com eficácia o seu papel estratégico.



Nesta seção, discuta os resultados obtidos pelo modelo construído, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade dele. Lembre-se de relacionar os resultados obtidos por cada uma das métricas ao problema identificado, a questão de pesquisa levantada e estabelecer relação com os objetivos previamente propostos. 
É fundamental compreender o que cada uma das métricas "conta" sobre a qualidade do modelo desenvolvido.

# Pipeline de pesquisa e análise de dados

Em pesquisa e experimentação em sistemas de informação, um pipeline de pesquisa e análise de dados refere-se a um conjunto organizado de processos e etapas que um profissional segue para realizar a coleta, preparação, análise e interpretação de dados durante a fase de pesquisa e desenvolvimento de modelos. Esse pipeline é essencial para extrair _insights_ significativos, entender a natureza dos dados e, construir modelos de aprendizado de máquina eficazes. 

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".
