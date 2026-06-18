# Implantação e Apresentação da Solução

Nesta etapa, detalhamos o processo de implantação do modelo preditivo de risco de crédito da PayShield Analytics em um ambiente de computação em nuvem. O objetivo é garantir que a solução opere de forma escalável, processando novos pedidos de empréstimo em tempo de execução para auxiliar a tomada de decisão.

## 1. Planejamento da Capacidade Operacional

Para garantir que a infraestrutura suporte a carga de inferências dinâmicas sem gargalos de desempenho, realizamos um planejamento de capacidade fundamentado na Teoria das Filas, utilizando a Lei de Little (L = λW). 

* **Modelagem Matemática:** Estimamos uma taxa de chegada (λ) de 50 requisições simultâneas por minuto em horário de pico (analistas submetendo propostas de crédito). O tempo médio de processamento de inferência do modelo Random Forest (carregado em memória) e da API (W) foi cronometrado em 0.2 segundos por requisição.
* **Simulação:** O número médio de requisições no sistema simultaneamente é L = (50/60) * 0.2 ≈ 0.16. Isso indica que uma arquitetura baseada em instâncias computacionais leves, porém com boa alocação de memória RAM (para carregar a matriz do modelo *Ensemble* e as bibliotecas de processamento como Pandas e Scikit-Learn), é perfeitamente capaz de lidar com o volume sem formar filas de espera significativas.

## 2. Avaliação de Provedores em Nuvem

Analisamos os três principais provedores de mercado (AWS, Microsoft Azure e Google Cloud Platform) com base nos requisitos do projeto:
* **Azure:** Excelente integração para ecossistemas baseados em C#/.NET, mas possui uma curva de custo ligeiramente maior para instâncias Linux de entrada.
* **Google Cloud (GCP):** Ótimo ecossistema de *Machine Learning* nativo (Vertex AI), porém complexo para a implantação de uma API web simples na camada gratuita/estudantil.
* **Amazon Web Services (AWS):** Escolhida como a plataforma oficial do projeto. A AWS oferece o serviço **EC2 (Elastic Compute Cloud)**, que proporciona controle total sobre a máquina virtual, alta escalabilidade e um nível gratuito (Free Tier) robusto. Além disso, permite a integração nativa com o **CloudWatch** para monitoramento.

## 3. Configuração do Ambiente na Nuvem

O ambiente na AWS foi configurado para equilibrar custo e eficiência computacional:
* **Máquina Virtual (VM):** Provisionamento de uma instância EC2 `t3.micro` (ou `t2.micro`), configurada com Ubuntu Server. Esta configuração oferece 2 vCPUs e 1 GB de RAM, suficiente para atender aos requisitos matemáticos calculados no planejamento de capacidade.
* **Armazenamento:** Volume EBS (*Elastic Block Store*) de 8 GB para o sistema operacional, dependências da API e armazenamento dos artefatos do modelo.
* **Redes e Segurança:** Configuração de um *Security Group* liberando tráfego HTTP/HTTPS (portas 80 e 443) para acesso externo da aplicação, e porta 22 restrita ao IP dos desenvolvedores para manutenção via SSH.

## 4. Implantação e Inferência Dinâmica

A aplicação foi projetada para realizar inferências estritamente dinâmicas, sem qualquer reprocessamento ou atualização de treinamento em tempo de execução, garantindo eficiência e evitando a degradação da performance. O processo seguiu as seguintes etapas:

1.  **Empacotamento do Modelo:** O modelo de *Random Forest* treinado na Etapa 3, juntamente com o `StandardScaler` (para padronização de variáveis contínuas como `loan_amount`) e o *One-Hot Encoder* (para variáveis como `Region` e `credit_type`), foram exportados utilizando a biblioteca `joblib` ou `pickle` na forma de arquivos `.pkl`.
2.  **Desenvolvimento da API:** Construímos uma API RESTful utilizando Python (via FastAPI ou Flask). A API carrega os arquivos `.pkl` na memória apenas uma vez durante a inicialização do servidor.
3.  **Interface e Predição:** A API disponibiliza um *endpoint* (ex: `/predict`) que recebe um arquivo JSON com os dados do novo cliente (valor do empréstimo, renda, DTI, LTV, tipo de crédito `EQUI`, etc.). O sistema padroniza esses novos dados utilizando os mesmos *scalers* do treinamento, submete ao modelo de Random Forest e retorna instantaneamente a probabilidade de inadimplência (Status 0 ou 1) para o usuário final.

## 5. Monitoramento de Desempenho e Alertas

Para garantir o alinhamento com os objetivos organizacionais e a evolução contínua da ferramenta, implementamos um plano de monitoramento nativo:
* **AWS CloudWatch:** Configurado para coletar métricas de utilização de CPU e memória da instância EC2 a cada 5 minutos.
* **Ajustes e Alertas:** Foram definidos alarmes para disparar notificações automáticas caso a utilização da CPU ultrapasse **75%** de forma sustentada, indicando uma possível anomalia de tráfego ou gargalo no processamento das árvores de decisão. Isso permite escalonar a instância de forma proativa (ex: migrando para uma `t3.medium`) antes que ocorra lentidão na experiência do analista de crédito.
