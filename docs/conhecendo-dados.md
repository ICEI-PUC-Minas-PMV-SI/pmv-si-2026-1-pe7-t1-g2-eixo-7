# Conhecendo os dados

Nesta seção, deverá ser registrada uma detalhada análise descritiva e exploratória sobre a base de dados selecionada na Etapa 1 com o objetivo de compreender a estrutura dos dados, detectar eventuais _outliers_ e também, avaliar/detectar as relações existentes entre as variáveis analisadas.

Para isso, sugere-se que sejam utilizados cálculos de medidas de tendência central, como média, mediana e moda, para entender a centralidade dos dados; sejam exploradas medidas de dispersão como desvio padrão e intervalos interquartil para avaliar a variabilidade dos dados; sejam utilizados gráficos descritivos como histogramas e box plots, para representar visualmente as características essenciais dos dados, pois essas visualizações podem facilitar a identificação de padrões e anomalias; sejam analisadas as relações entre as variáveis por meio de análise de correlação, gráficos de dispersões, mapas de calor, entre outras técnicas. 

Inclua nesta seção, gráficos, tabelas, trechos de código e demais artefatos que você considere relevantes para entender os dados com os quais você irá trabalhar.  Além disso, inclua e comente os trechos de código mais relevantes desenvolvidos para realizar suas análises. Na pasta "src", inclua o código fonte completo.



## Análise Exploratória de Dados tendo como base os gráficos Boxplot/Histograma
A análise abaixo foca nas variáveis quantitativas principais (DTI, Loan Amount, Credit Score e LTV) e como elas se comportam em relação ao Status (0 para adimplentes, 1 para inadimplentes).


### 1. Comprometimento de Renda (dtir1)
O dtir1 mede a porcentagem da renda mensal que o cliente compromete com dívidas.

- **Observação:** A média geral está em torno de 38,5%.

- **Comparação por Status:** Notamos que clientes em inadimplência (Status 1) possuem uma média levemente superior (39.1%) em comparação aos adimplentes (37.4%).
  
- **Insight:** Embora a diferença pareça pequena numericamente, o DTI é um indicador clássico de risco: quanto maior o comprometimento, menor a margem para imprevistos financeiros.

### 2. Valor do Empréstimo (loan_amount)
Aqui observamos uma diferença significativa entre os grupos.

- **Distribuição:** A média geral é de R$ 327.755, mas a mediana é de R$ 296.500, indicando uma assimetria positiva (alguns empréstimos de valor muito alto puxam a média para cima).

- **Diferença por Status::** * Status 0 (Adimplentes): Mediana de R$ 306.500.
  - Status 1 (Inadimplentes): Mediana de R$ 266.500.

- **Insight:** Curiosamente, os empréstimos de menor valor apresentam uma frequência maior de inadimplência. Isso pode sugerir que o perfil de renda desses tomadores é mais sensível a variações econômicas.


## Descrição dos achados

A partir da análise descrita e exploratória realizada, descreva todos os achados considerados relevantes para o contexto em que o trabalho se insere. Por exemplo: com relação à centralidade dos dados algo chamou a atenção? Foi possível identificar correlação entre os atributos? Que tipo de correlação (forte, fraca, moderada)? 

## Ferramentas utilizadas

Existem muitas ferramentas diferentes que podem ser utilizadas para fazer a análise dos dados. Nesta seção, descreva as ferramentas/tecnologias utilizadas e sua aplicação. Vale destacar que, preferencialmente, as análises deverão ser realizadas utilizando a linguagem de programação Python.


