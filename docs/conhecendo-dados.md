# Conhecendo os dados

No dataset escolhido, foi detectada uma parte significativa de dados faltantes em algumas features, enquanto outras apresentavam uma parcela menor de omissões. Foram aplicadas diferentes técnicas para mitigar essas lacunas: entre elas, a substituição pela mediana para dados numéricos e pela moda para os categóricos, nos casos com menor volume de dados faltantes. Para dados com um número de omissões mais expressivo e com importância substancial para a análise, utilizou-se a substituição pela mediana de agrupamento, segmentada por região e tipo de ocupação. Por fim, para as variáveis financeiras mais complexas, que apresentavam lacunas superiores a 20.000 registros, utilizou-se o algoritmo MICE (Multivariate Imputation by Chained Equations).

Após a limpeza inicial, observou-se que os valores de assimetria do LTV, que eram de 120,61, baixaram, mas continuavam acima de 100. Esse resultado chamou a atenção para dados que não condizem com a realidade; ao analisar os valores máximos, percebeu-se a existência de registros alterados que não haviam sido corrigidos.

<img width="505" height="212" alt="image" src="https://github.com/user-attachments/assets/bc3460ef-c4c1-4baa-97ce-4635126140dd" />

Nota-se a presença de valores de Loan-to-Value (LTV) acima de 1.000% e uma quantidade considerável acima de 200%, o que oferece um risco tanto ao banco quanto ao projeto, criando variáveis com dados irreais. Observou-se que simplesmente deletar as linhas irreais sem critério poderia causar falhas na análise. Após filtrar essas linhas com valores próximos à realidade (LTV < 150% e Renda > 0) e re-executar o preenchimento dos dados faltantes, foi possível averiguar as medidas de tendência central corretas com o dataset devidamente tratado.

<img width="609" height="426" alt="image" src="https://github.com/user-attachments/assets/51865190-0e6d-4f78-8394-64023d5707ca" />

Ao observar o Income (renda), nota-se uma disparidade social. Analisando a curtose, a assimetria, o máximo e a média, entende-se que a base ganha, em média, 6.000, mas que poucos indivíduos ganham valores extremamente altos que puxam essa média para cima, visto que a mediana é inferior à média. 

Já analisando o LTV, a média (72,22) e a mediana (74,74) estão muito próximas. Isso indica que a maioria dos empréstimos gira em torno de 75% do valor do imóvel  um valor condizenteaos críterios bancarios. Mesmo havendo casos que excedem 100%, isso se justifica pois alguns tipos de empréstimos cobrem não apenas o valor do imóvel, mas também custos administrativos e fiscais, sendo proveitoso observar se essa escolha agrega valor ao modelo e à análise do projeto.

Em contrapartida à distribuição da renda, o LTV apresenta um comportamento mais centralizado indicando que este dataset abrange desde pequenos créditos habitacionais até financiamentos de alto padrão.

Por fim o Credit_Score sendo uma variavel central para os bancos ter uma distribuição equilibrada mostra que tem tantos cliente com credito medio e com credito alto em proporções semelhantes



  ### Análise Exploratória de Dados tendo como base os gráficos Boxplot/Histograma
  A análise abaixo foca nas variáveis quantitativas principais (DTI, Loan Amount, Credit Score e LTV) e como elas se comportam em relação ao Status (0 para adimplentes, 1 para inadimplentes).

  Códigos utilizados para calcular em relação ao status:
  -  display(df.groupby('Status')['dtir1'].mean())
  -  display(df.groupby('Status')['loan_amount'].median())
  -  display(df.groupby('Status')['LTV'].median())
  
  #### 1. Comprometimento de Renda (dtir1)
  O dtir1 mede a porcentagem da renda mensal que o cliente compromete com dívidas.

  <img src="https://github.com/user-attachments/assets/ae269cd9-4c04-4b39-8b5e-06693af2452c" width="400" height="400">
  <img src="https://github.com/user-attachments/assets/977915e8-5a27-4b01-81b5-a4cc170c1cfd" width="400" height="400">

  - **Observação:** A média geral está em torno de 38,5%.
  
  - **Comparação por Status:** Notamos que clientes em inadimplência (Status 1) possuem uma média levemente superior (39.1%) em comparação aos adimplentes (37.4%).
    
  - **Insight:** Embora a diferença pareça pequena numericamente, o DTI é um indicador clássico de risco: quanto maior o comprometimento, menor a margem para imprevistos financeiros.
  
  #### 2. Valor do Empréstimo (loan_amount)
  Aqui observamos uma diferença significativa entre os grupos.
  
  <img src="https://github.com/user-attachments/assets/bf01b031-506c-49f2-b5ed-795963abfb1e" width="400">
  <img src="https://github.com/user-attachments/assets/f1bfdfaa-4458-4d1e-a2c0-6a71f1b0a45e" width="400">
  
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


