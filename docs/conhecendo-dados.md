# Conhecendo os dados

No dataset escolhido, foi detectada uma parte significativa de dados faltantes em algumas features, enquanto outras apresentavam uma parcela menor de omissões. Foram aplicadas diferentes técnicas para mitigar essas lacunas: entre elas, a substituição pela mediana para dados numéricos e pela moda para os categóricos, nos casos com menor volume de dados faltantes. Para dados com um número de omissões mais expressivo e com importância substancial para a análise, utilizou-se a substituição pela mediana de agrupamento, segmentada por região e tipo de ocupação. Por fim, para as variáveis financeiras mais complexas, que apresentavam lacunas superiores a 20.000 registros, utilizou-se o algoritmo MICE (Multivariate Imputation by Chained Equations).

Após a limpeza inicial, observou-se que os valores de assimetria do LTV, que eram de 120,61, baixaram, mas continuavam acima de 100. Esse resultado indicou a presença de registros com valores fora dos limites operacionais esperados; ao analisar os valores máximos, percebeu-se a existência de entradas que não haviam sido corrigidas na etapa anterior.

<img width="505" height="212" alt="image" src="https://github.com/user-attachments/assets/bc3460ef-c4c1-4baa-97ce-4635126140dd" />

Nota-se a presença de valores de Loan-to-Value (LTV) acima de 1.000% e uma quantidade considerável acima de 200%, o que representa um potencial comprometimento da qualidade da análise, introduzindo registros fora dos limites operacionais esperados. Observou-se que simplesmente deletar os registros atípicos sem critério poderia introduzir viés na análise. Após filtrar esses registros com valores dentro de limites plausíveis para o contexto de crédito imobiliário (LTV < 150% e Renda > 0) e re-executar o preenchimento dos dados faltantes, foi possível recalcular as medidas de tendência central com o dataset devidamente tratado.

<img width="609" height="426" alt="image" src="https://github.com/user-attachments/assets/51865190-0e6d-4f78-8394-64023d5707ca" />

Ao observar o Income (renda), nota-se uma assimetria na distribuição. Analisando a curtose, a assimetria, o máximo e a média, verifica-se que a mediana da renda na base é de aproximadamente 6.000, porém a presença de valores elevados em cauda longa eleva a média acima da mediana, indicando assimetria positiva.

Já analisando o LTV, a média (72,22) e a mediana (74,74) estão relativamente próximas. Isso sugere que a maioria dos empréstimos se concentra em torno de 75% do valor do imóvel, patamar compatível com critérios bancários comuns. Mesmo havendo casos que excedem 100%, isso pode, em parte, refletir produtos que cobrem também custos administrativos e fiscais, sendo recomendável avaliar se tais registros devem ser tratados separadamente no modelo.

Em contrapartida à distribuição da renda, o LTV apresenta um comportamento mais centralizado, indicando que este dataset contempla um espectro amplo de perfis, desde créditos habitacionais de menor valor até financiamentos de maior porte.

Por fim, o Credit Score, variável central nos processos de concessão de crédito, apresenta distribuição relativamente equilibrada na amostra, com proporções comparáveis de clientes em faixas de pontuação intermediária e elevada — o que sugere ausência de concentração extrema em um único segmento de risco, embora análises mais detalhadas sejam necessárias para caracterizar a distribuição com precisão.

### Análise Exploratória de Dados tendo como base os gráficos Boxplot/Histograma

A análise abaixo foca nas variáveis quantitativas principais (DTI, Loan Amount, Credit Score e LTV) e como elas se comportam em relação ao Status (0 para adimplentes, 1 para inadimplentes).

Códigos utilizados para calcular em relação ao status:

- display(df.groupby('Status')['dtir1'].mean())

- display(df.groupby('Status')['loan_amount'].median())

- display(df.groupby('Status')['LTV'].median())

#### 1. Comprometimento de Renda (dtir1)

O dtir1 mede a porcentagem da renda mensal que o cliente compromete com dívidas.

<img src="https://github.com/user-attachments/assets/ae269cd9-4c04-4b39-8b5e-06693af2452c" width="400" height="400">

<img src="https://github.com/user-attachments/assets/977915e8-5a27-4b01-81b5-a4cc170c1cfd" width="400" height="400">

- **Observação:** A média geral está em torno de 38,5%.

- **Comparação por Status:** Observa-se que clientes em inadimplência (Status 1) apresentam média levemente superior (39.1%) em comparação aos adimplentes (37.4%).

- **Insight:** Embora a diferença seja modesta em termos absolutos, o DTI é reconhecido na literatura de crédito como um indicador associado ao risco: comprometimentos de renda mais elevados tendem a reduzir a margem do tomador para absorver choques financeiros imprevistos.

#### 2. Valor do Empréstimo (loan_amount)

Observa-se uma diferença entre os grupos.

<img src="https://github.com/user-attachments/assets/bf01b031-506c-49f2-b5ed-795963abfb1e" width="400">

<img src="https://github.com/user-attachments/assets/f1bfdfaa-4458-4d1e-a2c0-6a71f1b0a45e" width="400">

- **Distribuição:** A média geral é de R$ 327.755, mas a mediana é de R$ 296.500, indicando uma assimetria positiva (alguns empréstimos de valor muito alto elevam a média acima da mediana).

- **Diferença por Status::** \* Status 0 (Adimplentes): Mediana de R$ 306.500.

- Status 1 (Inadimplentes): Mediana de R$ 266.500.

- **Insight:** Os empréstimos de menor valor apresentam frequência relativamente maior de inadimplência neste dataset. Uma hipótese plausível é que tomadores nessa faixa tenham perfil de renda mais suscetível a variações econômicas, embora essa interpretação exija validação com variáveis adicionais.

#### 3. Relação Empréstimo-Valor (LTV)

O LTV indica o quanto do valor do imóvel foi financiado.

- **Médias:** Clientes inadimplentes possuem um LTV médio ligeiramente maior (76,2%) do que os adimplentes (74,5%).

- **Insight:** De acordo com a literatura de crédito, LTVs mais elevados tendem a estar associados a maior risco, uma vez que o tomador possui menor participação própria no bem financiado. Ressalta-se, porém, que a diferença observada entre os grupos é modesta, e que outros fatores podem ter influência relevante.

#### 4. Score de Crédito (Credit_Score)

<img width="400" height="400" alt="Image" src="https://github.com/user-attachments/assets/fb7bc317-9a33-413d-b5e4-7c349ca967a9" />

- **Equilíbrio:** A mediana do Score está em 699.

- **Comportamento:** Ao analisar os histogramas, percebe-se que o Score de Crédito está distribuído de forma relativamente uniforme entre os grupos. Isso sugere que, isoladamente, o Score pode não ser o único preditor determinante de inadimplência nesta base, exigindo uma análise combinada com o DTI e LTV.

---

### Análise de Correlação entre Variáveis Numéricas

Esta seção investiga as relações entre as variáveis do dataset de crédito imobiliário utilizando o **coeficiente de correlação de Pearson**, gráficos de dispersão, box plots e gráficos de barras comparativos. O objetivo é identificar padrões associados ao comportamento de inadimplência (`Status = 1`) e avaliar hipóteses sobre a política de concessão de crédito.

> **Nota metodológica:** O coeficiente de Pearson mede a força e a direção de relações lineares entre variáveis numéricas. Valores próximos de ±1 indicam correlação forte; próximos de 0, correlação fraca ou ausente. Para variáveis categóricas ordinais (como faixa etária e região), foram utilizados boxplots, barras de taxa de inadimplência e análise comparativa de médias/medianas.

---

#### LTV vs. Status de Inadimplência

**Hipótese:** Clientes com LTV (Loan-to-Value) mais alto — ou seja, que financiam uma proporção maior do valor do imóvel — têm maior probabilidade de inadimplência (teoria do *Equity Negativo*).

```python

from scipy import stats

d = df[['LTV','Status']].dropna()

r, p = stats.pearsonr(d['LTV'], d['Status'])

print(f"r = {r:.4f}, p = {p:.2e}")

# r = 0.0389, p = 6.83e-46

df.groupby('Status')['LTV'].agg(['mean','median'])

# mean median

# Status

# 0 72.06 74.50

# 1 76.29 79.36

```

![LTV vs Status](graficos_correlacao/01_ltv_vs_status.png)

**Resultado:** A correlação de Pearson entre LTV e Status é **r = 0,039 (p < 0,001)**, estatisticamente significativa, porém **fraca**. O LTV médio dos inadimplentes (76,3%) é ligeiramente superior ao dos adimplentes (72,1%), e a mediana confirma essa diferença (~5 p.p.). O box plot revela distribuições similares entre os dois grupos, com sobreposição elevada, o que indica que o LTV isolado apresenta poder preditivo limitado — ainda que a tendência seja consistente com a hipótese.

---

#### DTI (dtir1) vs. Status de Inadimplência

**Hipótese:** Quanto maior o comprometimento de renda com dívidas (DTI — *Debt-to-Income Ratio*), maior o risco de inadimplência. Esperava-se identificar um limiar (*threshold*) a partir do qual o risco se eleva de forma mais pronunciada.

```python

d = df[['dtir1','Status']].dropna()

r, p = stats.pearsonr(d['dtir1'], d['Status'])

print(f"r = {r:.4f}, p = {p:.2e}")

# r = 0.0781, p = 1.16e-167

df.groupby('Status')['dtir1'].agg(['mean','median'])

# mean median

# Status

# 0 37.37 38.0

# 1 39.60 42.0

```

![DTI vs Status](graficos_correlacao/02_dti_vs_status.png)

**Resultado:** Esta é a **correlação de maior magnitude** entre as variáveis numéricas e o status de inadimplência (**r = 0,078, p < 0,001**). Embora ainda fraca em magnitude absoluta, o DTI apresenta diferença consistente: inadimplentes possuem DTI mediano de 42%, contra 38% dos adimplentes. O gráfico de taxa de inadimplência por faixa de DTI sugere uma tendência de elevação progressiva a partir de 40%, o que é consistente com a hipótese, embora não permita estabelecer um limiar de corte definitivo com base apenas nesta análise.

---

#### Credit Score vs. Taxa de Juros

**Hipótese:** Clientes com maior Credit Score recebem taxas de juros menores, validando uma política de precificação baseada em risco.

```python

d = df[['Credit_Score','rate_of_interest']].dropna()

r, p = stats.pearsonr(d['Credit_Score'], d['rate_of_interest'])

print(f"r = {r:.4f}, p = {p:.4f}")

# r = -0.0013, p = 0.6559

```

![Credit Score vs Taxa de Juros](graficos_correlacao/03_creditscore_vs_interest.png)

**Resultado:** **Ausência de correlação linear** (r = -0,0013, p = 0,656 — **não significativo**). O gráfico de dispersão confirma uma distribuição de pontos sem tendência discernível. A análise por faixa de score também não revela diferença expressiva na taxa de juros média entre clientes de baixo e alto score. Esse resultado sugere que a política de precificação observada nos dados **não apresenta, nesta amostra, alinhamento com o risco individual mensurado pelo Credit Score**, o que pode indicar limitações na modelagem de taxas ou características específicas do produto financeiro em questão.

---

#### Valor do Empréstimo vs. Taxas Iniciais (Upfront Charges)

**Hipótese:** As taxas iniciais crescem proporcionalmente ao valor do empréstimo. Taxas elevadas em empréstimos pequenos poderiam pressionar financeiramente o cliente já no início do contrato.

```python

d = df[['loan_amount','Upfront_charges']].dropna()

r, p = stats.pearsonr(d['loan_amount'], d['Upfront_charges'])

print(f"r = {r:.4f}, p = {p:.2e}")

# r = 0.0656, p = 4.01e-104

```

![Empréstimo vs Taxas Iniciais](graficos_correlacao/04_loanamount_vs_upfront.png)

**Resultado:** Correlação positiva fraca (**r = 0,066, p < 0,001**). Embora as taxas absolutas cresçam com o valor do empréstimo, a análise percentual (taxas / valor do empréstimo) sugere que **empréstimos menores suportam uma taxa inicial proporcionalmente maior**. Esse resultado é consistente com a hipótese de que o peso relativo das taxas iniciais é mais elevado para tomadores de crédito de menor volume, o que pode ser um fator a considerar na análise de inadimplência precoce.

---

#### Região vs. Valor do Imóvel

**Hipótese:** Regiões distintas apresentam valores medianos de imóveis diferentes, indicando desigualdades nas garantias oferecidas.

```python

df.groupby('Region')['property_value'].median().sort_values(ascending=False)

# south 428.000

# North 408.000

# North-East 388.000

# central 378.000

```

![Região vs Valor do Imóvel](graficos_correlacao/05_region_vs_propertyvalue.png)

**Resultado:** A região **Sul** apresenta imóveis com valor mediano ligeiramente superior ($428k), enquanto a região **Central** possui os menores valores medianos ($378k). A diferença entre as regiões, embora presente, não é pronunciada; a dispersão (box plot) revela alta variabilidade interna em todas elas, com presença de imóveis de alto valor em cada uma. A região **Nordeste**, com valor mediano intermediário ($388k), também é a que apresenta maior taxa de inadimplência (seção 4.6), o que indica que o valor da garantia isolado não é suficiente para explicar o risco regional.

---

#### Região vs. Taxa de Inadimplência

**Hipótese:** Algumas regiões apresentam taxas de inadimplência sistematicamente superiores, indicando risco concentrado geograficamente.

```python

df.groupby('Region')['Status'].mean() * 100

# North 22.51%

# North-East 30.45%

# central 27.54%

# south 26.63%

```

![Região vs Inadimplência](graficos_correlacao/06_region_vs_status.png)

**Resultado:** Observa-se variação notável entre regiões. A região **Nordeste** apresenta taxa de inadimplência estimada em **30,5%**, cerca de 8 pontos percentuais acima da região Norte (22,5%). A região Central (27,5%) e Sul (26,6%) ficam em posições intermediárias. Dado que o Nordeste concentra apenas ~830 contratos (menos de 1% da amostra), esse resultado deve ser interpretado com cautela — pode refletir tanto características sistêmicas regionais quanto variabilidade amostral decorrente do tamanho reduzido do subgrupo.

---

#### Tipo de Ocupação vs. LTV

**Hipótese:** Investidores (imóveis para renda/aluguel) assumem LTVs mais altos que moradores, por terem maior tolerância ao risco financeiro.

```python

df.groupby('occupancy_type')['LTV'].agg(['mean','median'])

# mean median

# occupancy_type

# ir (Investimento) 62.80 67.69

# pr (Residência P.) 73.30 76.01

# sr (2ª Residência) 71.58 75.30

```

![Ocupação vs LTV](graficos_correlacao/07_occupancy_vs_ltv.png)

**Resultado:** Contrariamente à hipótese, **imóveis de investimento (ir) apresentam o menor LTV médio (62,8%)**, enquanto residências principais (pr) têm o maior (73,3%). Isso pode indicar que investidores realizam maiores entradas (down payment), provavelmente por não terem acesso a programas de financiamento subsidiados para imóveis não residenciais — ou por adotarem estratégia mais conservadora em ativos de renda. Residências principais, muitas vezes financiadas por programas com entrada mínima, acabam com LTV mais elevado.

---

#### Faixa Etária vs. Renda

**Hipótese:** A renda segue uma curva de ciclo de vida, com pico nas faixas de 45–54 anos e queda nas extremidades etárias.

```python

df[df['age'].isin(age_order)].groupby('age')['income'].median().reindex(age_order)

# <25 ~3.960

# 25-34 ~4.920

# 35-44 ~5.760

# 45-54 ~6.480

# 55-64 ~6.240

# 65-74 ~5.520

# >74 ~4.440

```

![Idade vs Renda](graficos_correlacao/08_age_vs_income.png)

**Resultado:** Os dados confirmam parcialmente a hipótese. A renda mediana cresce progressivamente até a faixa **45–54 anos** (pico) e depois declina gradualmente. Clientes com menos de 25 anos e acima de 74 anos apresentam as menores rendas medianas, com alta variabilidade (box plot). A curva em formato de "sino assimétrico" é consistente com a teoria do ciclo de vida financeiro, indicando que tomadores jovens e idosos têm base de renda mais frágil.

---

#### Gênero vs. Credit Score

**Hipótese:** Pode existir diferença no comportamento de score entre gêneros, reflexo de padrões históricos de acesso ao crédito.

```python

df[df['Gender'].isin(['Male','Female'])].groupby('Gender')['Credit_Score'].agg(['mean','median'])

# mean median

# Female 698.71 698.0

# Male 699.80 700.0

```

![Gênero vs Credit Score](graficos_correlacao/09_gender_vs_creditscore.png)

**Resultado:** A diferença de Credit Score entre homens (média 699,8) e mulheres (média 698,7) é **negligenciável**, inferior a 2 pontos em uma escala de centenas. As distribuições são praticamente idênticas (box plot). **Não há evidência de disparidade de score por gênero** neste dataset, o que é um resultado positivo do ponto de vista de equidade no acesso ao crédito.

---

#### Faixa Etária vs. Inadimplência

**Hipótese:** Clientes mais jovens teriam maior risco de inadimplência por instabilidade de emprego; idosos por renda reduzida.

```python

df[df['age'].isin(age_order)].groupby('age')['Status'].mean().reindex(age_order) \* 100

# <25 28.95%

# 25-34 22.19%

# 35-44 22.27%

# 45-54 24.05%

# 55-64 25.89%

# 65-74 26.86%

# >74 30.01%

```

![Idade vs Inadimplência](graficos_correlacao/10_age_vs_status.png)

**Resultado:** Os resultados são consistentes com a hipótese para **ambos os extremos etários**. Clientes com menos de 25 anos (28,9%) e acima de 74 anos (30,0%) apresentam as maiores taxas de inadimplência. O ponto de menor risco está na faixa **25–44 anos** (~22%), correspondendo ao período de maior estabilidade laboral e menor endividamento relativo. A partir dos 45 anos, a taxa cresce progressivamente — possivelmente associada ao aumento do DTI em idades mais avançadas e à redução de renda na aposentadoria.

---

#### Modalidade Interest-Only vs. Inadimplência

**Hipótese:** Contratos onde o cliente paga apenas juros (sem amortização do principal) apresentam maior taxa de inadimplência.

```python

df.groupby('interest_only')['Status'].mean() \* 100

# int_only 27.31%

# not_int 24.51%

```

![Interest-Only vs Status](graficos_correlacao/11_interestonly_vs_status.png)

**Resultado:** Clientes em modalidade *interest-only* apresentam taxa de inadimplência de **27,3%**, contra 24,5% na modalidade convencional — uma diferença de ~2,8 pontos percentuais. A hipótese se confirma, ainda que a magnitude seja moderada. A explicação teórica é consistente: sem redução progressiva do saldo devedor, o tomador mantém exposição integral ao principal ao longo de todo o contrato, de modo que qualquer deterioração de renda pode resultar em inadimplência.

---

#### Mapa de Calor — Correlação de Pearson Geral

Para uma visão integrada das relações lineares entre todas as variáveis numéricas:

```python

num_cols = ['loan_amount','rate_of_interest','Upfront_charges','property_value',

'income','Credit_Score','LTV','dtir1','term','Status']

corr = df[num_cols].corr(method='pearson')

```

![Mapa de Calor de Correlação](graficos_correlacao/12_heatmap_correlacao.png)

**Principais observações do heatmap:**

- `loan_amount` e `property_value` apresentam a **maior correlação positiva** do dataset (**r = 0,73**), o que é esperado — o valor financiado é naturalmente proporcional ao valor do bem.

- `loan_amount` e `Upfront_charges` também têm correlação moderada positiva (~0,07 a 0,10), confirmando a análise da seção 4.4.

- `Credit_Score` e `rate_of_interest` apresentam correlação próxima de **zero** (~-0,001), reforçando o achado da seção 4.3.

- A variável `Status` apresenta correlações muito fracas com todas as demais variáveis numéricas isoladas, indicando que a inadimplência é um fenômeno **multivariado**, não explicado por nenhuma variável isolada.

---

## Descrição dos achados

### Em Relação aos Boxplots e Histogramas:

A análise dos Boxplots e Histogramas indica que, neste dataset, os registros classificados como inadimplentes (Status 1) tendem a apresentar as seguintes características em comparação aos adimplentes — ressaltando que tais tendências são de natureza estatística e não constituem perfis determinísticos:

1. Ter um menor valor de empréstimo (mediana menor)

2. Ter um maior comprometimento de renda (DTI mais alto)

3. Possuir um financiamento que cobre uma parcela maior do valor do bem (LTV mais alto).

---

### Análise de Correlação entre Variáveis Numéricas

A análise de correlação realizada sobre o dataset de inadimplência imobiliária revelou um conjunto de achados com diferentes graus de relevância prática e estatística:

**Correlações com o Status de inadimplência**

Nenhuma das variáveis numéricas analisadas apresentou correlação forte com o status de inadimplência. A variável de maior coeficiente foi o DTI (*r* = 0,078), seguida pelo LTV (*r* = 0,039) e pela taxa de juros (*r* = 0,023). Embora todas sejam estatisticamente significativas (p < 0,001) dada a grande amostra (148k registros), os valores absolutos indicam **correlações fracas**. Isso sugere que o risco de inadimplência é um fenômeno complexo, não capturado linearmente por variáveis isoladas, e que modelos preditivos com capacidade de capturar interações entre variáveis tendem a apresentar desempenho potencialmente superior ao de análises univariadas.

<img width="577" height="379" alt="image" src="https://github.com/user-attachments/assets/af7edfe7-d559-4651-b60c-5022a40e1b7b" />

**DTI como principal preditor linear**

O índice de comprometimento de renda (DTI) foi a variável com maior correlação com inadimplência. A análise por faixas sugere uma tendência de elevação progressiva de risco a partir de 40%, com inadimplentes apresentando DTI mediano de 42% contra 38% dos adimplentes. Esse achado reforça a pertinência de considerar limites de DTI nas políticas de crédito.

<img width="457" height="313" alt="image" src="https://github.com/user-attachments/assets/d441c79f-db23-4f6b-9431-6cec3ae66bcd" />

<img width="352" height="261" alt="image" src="https://github.com/user-attachments/assets/eee49a83-c86e-4636-8930-eb7ecc56709a" />

**Ausência de precificação baseada em score**

Um resultado que merece atenção foi a **correlação praticamente nula entre Credit Score e taxa de juros** (*r* = -0,001, p = 0,66 — não significativo). Esse resultado diverge da teoria padrão de precificação por risco e levanta questões sobre a lógica de formação de taxas na instituição analisada, embora limitações do dataset (como ausência de variáveis contextuais sobre produtos e períodos) impeçam conclusões definitivas.

<img width="450" height="312" alt="image" src="https://github.com/user-attachments/assets/b382c028-82d2-434f-9727-6a08efe16273" />

<img width="235" height="198" alt="image" src="https://github.com/user-attachments/assets/2d7d5038-0a19-4e4c-b33b-1ea754267241" />

**Risco geográfico maior no Nordeste**

A região Nordeste apresenta taxa de inadimplência estimada em 30,5%, numericamente superior à média geral de 24,6%. Associada ao menor valor mediano de imóveis dessa região, essa concentração de risco geográfico sugere a necessidade de monitoramento adicional, mesmo considerando o tamanho reduzido da amostra regional.

<img width="461" height="317" alt="image" src="https://github.com/user-attachments/assets/6edc29db-4e39-4396-a624-33638fd64363" />

<img width="482" height="315" alt="image" src="https://github.com/user-attachments/assets/6ef30e87-0a3d-4cda-b26a-8abf4505ca4d" />

**Perfil de risco por ciclo de vida**

Tanto a análise etária quanto a análise de renda são consistentes com um padrão de ciclo de vida: os extremos etários (<25 e >74 anos) concentram maior inadimplência e menor renda mediana. A faixa de 25–44 anos representa o segmento de menor risco.

<img width="514" height="367" alt="image" src="https://github.com/user-attachments/assets/e2d72838-9645-496b-837d-4450cc3d316a" />

<img width="529" height="368" alt="image" src="https://github.com/user-attachments/assets/d95cae53-a9b6-48ea-af3a-0ac17776fdcd" />

**Interest-Only como fator de risco moderado**

A modalidade *interest-only* apresenta taxa de inadimplência ~2,8 p.p. superior à modalidade convencional, confirmando que a ausência de amortização do principal representa risco adicional, ainda que moderado neste dataset.

<img width="407" height="316" alt="image" src="https://github.com/user-attachments/assets/cfa23f9b-760d-4cdd-a4d4-b39deeecb88f" />

**Equidade de gênero no score**

Não foi identificada diferença relevante no Credit Score médio entre homens e mulheres (< 2 pontos), o que representa um resultado positivo do ponto de vista de equidade no sistema de avaliação de crédito.

<img width="410" height="313" alt="image" src="https://github.com/user-attachments/assets/29371c07-e3d5-4439-a6da-3018d1e428ee" />

<img width="460" height="310" alt="image" src="https://github.com/user-attachments/assets/928b441a-de89-475b-b1cb-52669ef1d845" />

**Comportamento do LTV por tipo de ocupação**

Contrariamente à hipótese inicial, investidores (imóveis para renda) apresentam LTV médio inferior ao de moradores. A explicação mais provável é a exigência de maior entrada para financiamentos não residenciais, reduzindo o LTV médio desse segmento.

<img width="685" height="475" alt="image" src="https://github.com/user-attachments/assets/cc606982-09a8-42e5-9824-f9eee960485c" />

<img width="772" height="471" alt="image" src="https://github.com/user-attachments/assets/00804e0a-385a-4a1c-9260-e58d4f0cc206" />

---

## Ferramentas utilizadas

### Análise de Correlação entre Variáveis Numéricas

| Ferramenta | Aplicação |
|---|---|
| Python | Linguagem principal de análise |
| pandas | Manipulação e agregação do dataset |
| scipy.stats | Cálculo do coeficiente de correlação de Pearson e p-values |
| matplotlib | Geração de gráficos de dispersão, barras e box plots |
| seaborn | Mapa de calor de correlação e estilização dos gráficos |

O código fonte completo está disponível em `src/inadimplência.ipynb`.

---

## Reflexão Ética

O desenvolvimento de modelos preditivos de inadimplência envolve responsabilidades que vão além da acurácia técnica. Esta seção apresenta uma reflexão sobre os principais aspectos éticos identificados ao longo da análise exploratória.

### Uso de atributos sensíveis

O dataset contém variáveis como faixa etária (`age`), gênero (`Gender`) e região (`Region`), cujo uso em modelos preditivos deve ser cuidadosamente avaliado. Embora a análise tenha demonstrado que o Credit Score não apresenta disparidade relevante por gênero — o que é um indicador positivo — a correlação entre faixa etária e inadimplência levanta um ponto de atenção: um modelo treinado com esses dados pode aprender a penalizar grupos etários extremos (jovens e idosos) com base em padrões históricos, mesmo quando o risco individual não justifica essa penalização. O uso de atributos protegidos como preditores diretos deve ser evitado ou ao menos explicitamente justificado com base em critérios regulatórios e de equidade.

### Risco de perpetuação de desigualdades

A concentração de inadimplência observada na região Nordeste, mesmo que estatisticamente modesta, ilustra como padrões socioeconômicos históricos podem ser incorporados inadvertidamente aos modelos. Se o modelo aprender a associar determinada região a maior risco de crédito, poderá recomendar negação de crédito ou taxas mais altas para populações já em situação de vulnerabilidade econômica, amplificando desigualdades preexistentes em vez de mitigá-las. Qualquer análise regional deve ser acompanhada de análise de causalidade e contexto socioeconômico antes de subsidiar decisões automatizadas.

### Limitações do dataset e validade das inferências

Todas as correlações identificadas são estatisticamente significativas, porém de magnitude fraca. A interpretação dos resultados deve respeitar essa limitação: correlação não implica causalidade, e padrões identificados em dados históricos não necessariamente refletem relações estruturais. Afirmações sobre "o que determina a inadimplência" devem ser substituídas por formulações mais cautelosas, como "variáveis associadas a maior frequência de inadimplência nesta amostra". A ausência de informações sobre o período de coleta, a instituição financeira específica e as políticas de concessão vigentes à época também limita a generalização dos achados.

### Transparência e explicabilidade

Modelos de machine learning que utilizem este dataset para decisões de crédito devem ser acompanhados de mecanismos de explicabilidade (como SHAP ou LIME), de forma que seja possível auditar quais variáveis influenciam cada decisão individual. Decisões automatizadas que afetam o acesso de pessoas a crédito imobiliário têm impacto direto sobre seus projetos de vida, e a opacidade algorítmica representa um risco ético considerável.

### Responsabilidade sobre o uso dos resultados

Os achados deste projeto têm caráter exploratório e acadêmico. Qualquer aplicação dos modelos desenvolvidos em contextos reais de concessão de crédito requer validação por especialistas em crédito, revisão jurídica à luz da legislação aplicável (como a LGPD no Brasil), e auditoria de viés antes da implantação. A equipe se compromete a documentar as limitações dos modelos e a não recomendar sua aplicação direta sem as devidas etapas de validação e conformidade regulatória.
