# Mini-Projeto Machine Learning Aplicado — Classificador de Interacoes Medicamentosas


## 1. Identificacao do Grupo

- Integrantes:
  - Eric Endres
  - Felipe Kurt Pohling
  - Igor Kussumoto do Nascimento
  - Manuela Bechara Cannizza

---

## 2. Objetivo do Software

O sistema tem como objetivo classificar automaticamente interacoes medicamentosas em tres niveis de gravidade — leve, moderada e grave — a partir de descricoes textuais de pares de medicamentos. O problema que o software resolve e pratico e relevante: profissionais de saude e pacientes frequentemente precisam avaliar o risco de combinar dois ou mais medicamentos, e essa avaliacao manual e demorada e sujeita a erros.

O programa utiliza tecnicas de Machine Learning classico (aprendizado supervisionado e nao supervisionado) para treinar modelos em tempo de execucao a partir de um dataset em formato CSV contendo milhares de interacoes rotuladas. Ao final do treinamento, o sistema permite ao usuario consultar novas combinacoes de medicamentos e obter uma classificacao imediata da gravidade da interacao, comparando os resultados de dois algoritmos distintos (Naive Bayes e SVM).

---

## 3. Publico-alvo

O sistema e destinado a estudantes de areas da saude, farmaceuticos e profissionais que precisam de uma ferramenta rapida de triagem para avaliar a gravidade potencial de interacoes entre medicamentos. Pode ser utilizado como ferramenta de apoio a decisao em ambientes academicos e clinicos, sem substituir a consulta a fontes farmacologicas oficiais.

O sistema e voltado para uso individual em ambiente local (desktop), sendo executado via terminal/console.

---

## 4. Funcionalidades do Sistema

### 4.1 Carregamento e Preparacao dos Dados

O sistema carrega automaticamente um arquivo CSV (`interacoes_medicamentosas.csv`) contendo duas colunas:
- `interacao`: texto no formato "medicamento X com medicamento Y"
- `gravidade`: rotulo da interacao — leve, moderada ou grave

- O sistema deve ler o arquivo CSV e exibir a quantidade total de interacoes carregadas.
- O sistema deve converter os rotulos textuais (leve, moderada, grave) em valores numericos (0, 1, 2) para uso nos algoritmos de ML.
- O sistema deve exibir a distribuicao de classes (quantas interacoes de cada gravidade existem no dataset).

### 4.2 Vetorizacao de Texto (Bag of Words)

O sistema transforma as descricoes textuais das interacoes em representacoes numericas utilizando a tecnica de Bag of Words, conforme ensinado no Bloco 7.

- O sistema deve utilizar `CountVectorizer` do scikit-learn para criar o vocabulario a partir dos dados de treino (`.fit_transform()`).
- O sistema deve aplicar o vocabulario aprendido aos dados de teste e a novas entradas com `.transform()`.
- O sistema deve exibir o tamanho do vocabulario e as dimensoes da matriz de features.

### 4.3 Modulo 1 — Classificacao com Naive Bayes (Bloco 7)

Extensao direta do classificador de spam do Bloco 7, adaptado para tres classes de gravidade.

- O sistema deve dividir os dados em treino (70%) e teste (30%) com `train_test_split`.
- O sistema deve treinar um modelo `MultinomialNB` com os dados de treino vetorizados.
- O sistema deve prever as classes do conjunto de teste e calcular a acuracia.
- O sistema deve gerar um relatorio detalhado com precision, recall e f1-score para cada classe (LEVE, MODERADA, GRAVE) utilizando `classification_report`.

### 4.4 Modulo 2 — Clustering com K-Means (Bloco 9)

Aplicacao de aprendizado nao supervisionado para descobrir agrupamentos naturais nas interacoes, conforme ensinado no Bloco 9.

- O sistema deve treinar o algoritmo K-Means com K=3 clusters sobre a matriz de features completa.
- O sistema deve exibir a quantidade de interacoes em cada cluster.
- O sistema deve mostrar exemplos representativos de cada cluster com a gravidade real correspondente.
- O sistema deve apresentar a distribuicao de gravidades dentro de cada cluster, permitindo analisar se os agrupamentos por palavras refletem a gravidade.
- O sistema deve experimentar diferentes valores de K (2, 3, 4, 5) e exibir a inertia de cada um, demonstrando o impacto de K na qualidade do agrupamento.

### 4.5 Modulo 3 — Classificacao com SVM (Bloco 6)

Segundo modelo supervisionado para comparacao de desempenho, utilizando Support Vector Machine conforme apresentado no Bloco 6.

- O sistema deve treinar um modelo `SVC` com kernel linear sobre os mesmos dados de treino.
- O sistema deve prever as classes do conjunto de teste e calcular a acuracia.
- O sistema deve gerar relatorio de classificacao identico ao do Modulo 1 para comparacao direta.

### 4.6 Comparacao de Modelos

- O sistema deve comparar a acuracia dos dois modelos supervisionados (Naive Bayes vs SVM) no mesmo conjunto de teste.
- O sistema deve indicar qual modelo obteve melhor desempenho.

### 4.7 Modulo 4 — Consulta Interativa

Interface para o usuario testar novas interacoes em tempo real.

- O sistema deve testar automaticamente um conjunto de exemplos pre-definidos, exibindo a classificacao de ambos os modelos lado a lado.
- O sistema deve permitir que o usuario digite novas interacoes no formato "medicamento X com medicamento Y".
- O sistema deve classificar a entrada com Naive Bayes e SVM simultaneamente.
- O sistema deve indicar quando os modelos concordam ou divergem na classificacao.
- O sistema deve permitir que o usuario encerre a consulta digitando "sair".

---

## 5. Requisitos Funcionais

RF01 — O sistema deve carregar dados de interacoes medicamentosas a partir de um arquivo CSV.
RF02 — O sistema deve converter rotulos textuais (leve, moderada, grave) em valores numericos.
RF03 — O sistema deve vetorizar textos com Bag of Words usando CountVectorizer.
RF04 — O sistema deve dividir os dados em conjuntos de treino (70%) e teste (30%).
RF05 — O sistema deve treinar um classificador Naive Bayes (MultinomialNB).
RF06 — O sistema deve treinar um classificador SVM (SVC com kernel linear).
RF07 — O sistema deve treinar um modelo K-Means para clustering nao supervisionado.
RF08 — O sistema deve calcular e exibir acuracia para cada modelo supervisionado.
RF09 — O sistema deve gerar relatorio de classificacao com precision, recall e f1-score.
RF10 — O sistema deve comparar o desempenho dos modelos Naive Bayes e SVM.
RF11 — O sistema deve permitir consulta interativa de novas interacoes medicamentosas.
RF12 — O sistema deve classificar novas entradas com ambos os modelos e indicar concordancia ou divergencia.
RF13 — O sistema deve exibir a distribuicao de gravidades por cluster do K-Means.
RF14 — O sistema deve experimentar diferentes valores de K e exibir a inertia correspondente.

---

## 6. Requisitos Nao Funcionais

RNF01 — O sistema deve treinar os modelos em menos de 10 segundos com o dataset completo (~10.000 linhas).
RNF02 — O sistema deve responder a consultas interativas em menos de 1 segundo apos o treinamento.
RNF03 — O sistema deve funcionar em ambiente local sem necessidade de conexao com a internet.
RNF04 — O sistema deve exibir saida formatada e legivel no terminal/console.
RNF05 — O sistema deve ser executavel com Python 3.10+ e scikit-learn instalado.
RNF06 — O codigo deve ser organizado em funcoes modulares, com separacao clara entre carregamento de dados, treinamento, avaliacao e interface.

---

## 7. Tecnologias Utilizadas

- Linguagem:
  Python 3

- Bibliotecas:
  - scikit-learn (CountVectorizer, MultinomialNB, SVC, KMeans, train_test_split, accuracy_score, classification_report)
  - NumPy (manipulacao de arrays)
  - csv (leitura do dataset)

- Ambiente de desenvolvimento:
  Aplicacao local executada via terminal (console). Compativel com Google Colab.

- Tecnicas de ML:
  - Bag of Words (vetorizacao de texto) — Bloco 7
  - Naive Bayes (classificacao probabilistica supervisionada) — Bloco 7
  - SVM (classificacao por hiperplano separador) — Bloco 6
  - K-Means (clustering nao supervisionado) — Bloco 9

---

## 8. Escopo (O que NAO sera feito)

- O sistema nao utiliza redes neurais, deep learning ou modelos de linguagem (LLMs).
- O sistema nao realiza fine-tuning de modelos pre-treinados.
- O sistema nao possui interface grafica (GUI) — a interacao e exclusivamente via terminal.
- O sistema nao acessa APIs externas ou bases de dados online.
- O sistema nao substitui consulta medica ou farmacologica profissional.
- O sistema nao realiza processamento avancado de linguagem natural (NLP profundo) — utiliza exclusivamente Bag of Words conforme o escopo da disciplina.
- O sistema nao armazena historico de consultas entre execucoes.
