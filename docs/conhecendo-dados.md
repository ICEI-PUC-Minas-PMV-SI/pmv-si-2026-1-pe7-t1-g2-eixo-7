# Conhecendo os dados

Nesta seção, deverá ser registrada uma detalhada análise descritiva e exploratória sobre a base de dados selecionada na Etapa 1 com o objetivo de compreender a estrutura dos dados, detectar eventuais _outliers_ e também, avaliar/detectar as relações existentes entre as variáveis analisadas.

Para isso, sugere-se que sejam utilizados cálculos de medidas de tendência central, como média, mediana e moda, para entender a centralidade dos dados; sejam exploradas medidas de dispersão como desvio padrão e intervalos interquartil para avaliar a variabilidade dos dados; sejam utilizados gráficos descritivos como histogramas e box plots, para representar visualmente as características essenciais dos dados, pois essas visualizações podem facilitar a identificação de padrões e anomalias; sejam analisadas as relações entre as variáveis por meio de análise de correlação, gráficos de dispersões, mapas de calor, entre outras técnicas. 

Inclua nesta seção, gráficos, tabelas, trechos de código e demais artefatos que você considere relevantes para entender os dados com os quais você irá trabalhar.  Além disso, inclua e comente os trechos de código mais relevantes desenvolvidos para realizar suas análises. Na pasta "src", inclua o código fonte completo.


  ### Análise Exploratória de Dados tendo como base os gráficos Boxplot/Histograma
  A análise abaixo foca nas variáveis quantitativas principais (DTI, Loan Amount, Credit Score e LTV) e como elas se comportam em relação ao Status (0 para adimplentes, 1 para inadimplentes).

  Códigos utilizados para calcular em relação ao status:
  -  display(df.groupby('Status')['dtir1'].mean())
  -  display(df.groupby('Status')['loan_amount'].median())
  -  display(df.groupby('Status')['LTV'].median())
  
  #### 1. Comprometimento de Renda (dtir1)
  O dtir1 mede a porcentagem da renda mensal que o cliente compromete com dívidas.

  <img src="https://github.com/user-attachments/assets/bf01b031-506c-49f2-b5ed-795963abfb1e" width="400">
  <img src="https://github.com/user-attachments/assets/f1bfdfaa-4458-4d1e-a2c0-6a71f1b0a45e" width="400">
  
  - **Observação:** A média geral está em torno de 38,5%.
  
  - **Comparação por Status:** Notamos que clientes em inadimplência (Status 1) possuem uma média levemente superior (39.1%) em comparação aos adimplentes (37.4%).
    
  - **Insight:** Embora a diferença pareça pequena numericamente, o DTI é um indicador clássico de risco: quanto maior o comprometimento, menor a margem para imprevistos financeiros.
  
  #### 2. Valor do Empréstimo (loan_amount)
  Aqui observamos uma diferença significativa entre os grupos.

  <img width="400" height="400" alt="Image" src="https://github.com/user-attachments/assets/df936c4e-781d-4923-9dc0-54c361f6eee3" />
  
  - **Distribuição:** A média geral é de R$ 327.755, mas a mediana é de R$ 296.500, indicando uma assimetria positiva (alguns empréstimos de valor muito alto puxam a média para cima).
  
  - **Diferença por Status::** * Status 0 (Adimplentes): Mediana de R$ 306.500.
    - Status 1 (Inadimplentes): Mediana de R$ 266.500.
  
  - **Insight:** Curiosamente, os empréstimos de menor valor apresentam uma frequência maior de inadimplência. Isso pode sugerir que o perfil de renda desses tomadores é mais sensível a variações econômicas.
  
  #### 3. Relação Empréstimo-Valor (LTV)
  O LTV indica o quanto do valor do imóvel foi financiado.
  
  - **Médias:** Clientes inadimplentes possuem um LTV médio maior (76,2%) do que os adimplentes (74,5%).
  
  - **Insight:** Um LTV mais alto significa que o cliente tem menos "capital próprio" no imóvel. Historicamente, quanto maior o LTV, maior o risco, pois o cliente tem menos a perder em caso de execução da dívida.
  
  #### 4. Score de Crédito (Credit_Score)

  <img width="400" height="400" alt="Image" src="https://github.com/user-attachments/assets/fb7bc317-9a33-413d-b5e4-7c349ca967a9" />
  
  - **Equilíbrio:** A mediana do Score está em 699.
  
  - **Comportamento:** Ao analisar os histogramas, percebe-se que o Score de Crédito está distribuído de forma relativamente uniforme entre os grupos. Isso sugere que, isoladamente, o Score pode não ser o único preditor determinante de inadimplência nesta base, exigindo uma análise combinada com o DTI e LTV.

## Descrição dos achados

A partir da análise descrita e exploratória realizada, descreva todos os achados considerados relevantes para o contexto em que o trabalho se insere. Por exemplo: com relação à centralidade dos dados algo chamou a atenção? Foi possível identificar correlação entre os atributos? Que tipo de correlação (forte, fraca, moderada)? 

  ### Em Relação aos Boxplots e Histogramas:
  A análise dos Boxplots e Histogramas revela que o perfil do inadimplente (Status 1) neste dataset tende a:

  1. Ter um menor valor de empréstimo (mediana menor)
  2. Ter um maior comprometimento de renda (DTI mais alto)
  3. Possuir um financiamento que cobre uma parcela maior do valor do bem (LTV mais alto).
  

## Ferramentas utilizadas

Existem muitas ferramentas diferentes que podem ser utilizadas para fazer a análise dos dados. Nesta seção, descreva as ferramentas/tecnologias utilizadas e sua aplicação. Vale destacar que, preferencialmente, as análises deverão ser realizadas utilizando a linguagem de programação Python.


