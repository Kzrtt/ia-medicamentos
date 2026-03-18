# =============================================================================
# Classificador de Interacoes Medicamentosas
# Perfil: Machine Learning aplicado (Blocos 6 -> 7 -> 9)
#
# Integrantes:
#     - Eric Endres
#     - Felipe Kurt Pohling
#     - Igor Kussumoto do Nascimento
#     - Manuela Bechara Cannizza
#
# Disciplina: INC702-7SI — Fundamentos de IA
# =============================================================================
#
# Este programa implementa uma IA treinada em tempo de execucao para
# classificar interacoes medicamentosas em tres niveis de gravidade:
#   - leve (0)
#   - moderada (1)
#   - grave (2)
#
# Modulo 1: Classificacao supervisionada com Naive Bayes (Bloco 7)
# Modulo 2: Clustering nao supervisionado com K-Means (Bloco 9)
# Modulo 3: Classificacao supervisionada com SVM (Bloco 6)
# Modulo 4: Interface interativa para consulta de novas interacoes
# =============================================================================

import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# FUNCAO: Carregar dados do CSV
# =============================================================================
def carregar_dados(caminho_csv):
    """Carrega o arquivo CSV com interacoes medicamentosas."""
    interacoes = []
    gravidades = []

    with open(caminho_csv, mode="r", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        for linha in leitor:
            interacoes.append(linha["interacao"].strip())
            gravidades.append(linha["gravidade"].strip())

    print(f"Dados carregados: {len(interacoes)} interacoes")
    return interacoes, gravidades


# =============================================================================
# FUNCAO: Converter rotulos de texto para numeros
# =============================================================================
def converter_rotulos(gravidades):
    """Converte gravidades textuais em numeros: leve=0, moderada=1, grave=2."""
    mapa = {"leve": 0, "moderada": 1, "grave": 2}
    rotulos = []
    for g in gravidades:
        rotulos.append(mapa[g])
    return rotulos


# =============================================================================
# FUNCAO: Converter numeros de volta para texto
# =============================================================================
def rotulo_para_texto(numero):
    """Converte numero de volta para texto da gravidade."""
    mapa = {0: "leve", 1: "moderada", 2: "grave"}
    return mapa[numero]


# =============================================================================
# MODULO 1: Classificacao com Naive Bayes (Bloco 7)
# =============================================================================
def modulo_naive_bayes(X_treino, X_teste, y_treino, y_teste, vetorizador):
    """Treina e avalia um classificador Naive Bayes."""
    print("\n" + "=" * 60)
    print("MODULO 1 — Classificacao com Naive Bayes (Bloco 7)")
    print("=" * 60)

    # --- Criar e treinar o modelo (Naive Bayes) ---
    modelo = MultinomialNB()
    modelo.fit(X_treino, y_treino)

    # --- Prever no conjunto de teste ---
    previsoes = modelo.predict(X_teste)

    # --- Avaliar ---
    acuracia = accuracy_score(y_teste, previsoes)
    print(f"\nAcuracia: {acuracia:.2f}")

    print("\nRelatorio de Classificacao:")
    relatorio = classification_report(
        y_teste, previsoes,
        target_names=["LEVE", "MODERADA", "GRAVE"],
        zero_division=0
    )
    print(relatorio)

    return modelo


# =============================================================================
# MODULO 2: Clustering com K-Means (Bloco 9)
# =============================================================================
def modulo_kmeans(X_completo, interacoes, gravidades_texto):
    """Agrupa interacoes usando K-Means e compara com rotulos reais."""
    print("\n" + "=" * 60)
    print("MODULO 2 — Clustering com K-Means (Bloco 9)")
    print("=" * 60)

    # --- Converter matriz esparsa para array denso ---
    X_array = X_completo.toarray()

    # --- Treinar K-Means com K=3 (leve, moderada, grave) ---
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_array)

    # --- Obter os rotulos atribuidos pelo K-Means ---
    rotulos_cluster = kmeans.labels_

    print(f"\nNumero de clusters: 3")
    print(f"Total de interacoes agrupadas: {len(rotulos_cluster)}")

    # --- Mostrar quantas interacoes em cada cluster ---
    for k in range(3):
        quantidade = list(rotulos_cluster).count(k)
        print(f"  Cluster {k}: {quantidade} interacoes")

    # --- Mostrar exemplos de cada cluster ---
    print("\nExemplos por cluster:")
    for k in range(3):
        print(f"\n  --- Cluster {k} ---")
        contador = 0
        for i in range(len(interacoes)):
            if rotulos_cluster[i] == k and contador < 3:
                print(f"    \"{interacoes[i]}\" (gravidade real: {gravidades_texto[i]})")
                contador += 1

    # --- Comparar clusters com gravidades reais ---
    print("\nDistribuicao de gravidades por cluster:")
    for k in range(3):
        leve = 0
        moderada = 0
        grave = 0
        for i in range(len(interacoes)):
            if rotulos_cluster[i] == k:
                if gravidades_texto[i] == "leve":
                    leve += 1
                elif gravidades_texto[i] == "moderada":
                    moderada += 1
                elif gravidades_texto[i] == "grave":
                    grave += 1
        print(f"  Cluster {k}: leve={leve}, moderada={moderada}, grave={grave}")

    # --- Experimentar diferentes valores de K ---
    print("\nExperimentando diferentes valores de K:")
    valores_de_k = [2, 3, 4, 5]
    for k_val in valores_de_k:
        km_temp = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        km_temp.fit(X_array)
        inertia = km_temp.inertia_
        print(f"  K={k_val} -> Inertia (soma das distancias): {inertia:.2f}")

    print("\nNota: O K-Means agrupa por similaridade de PALAVRAS, nao por gravidade.")
    print("Clusters refletem quais medicamentos aparecem juntos nos textos.")

    return kmeans


# =============================================================================
# MODULO 3: Classificacao com SVM (Bloco 6)
# =============================================================================
def modulo_svm(X_treino, X_teste, y_treino, y_teste, vetorizador):
    """Treina e avalia um classificador SVM."""
    print("\n" + "=" * 60)
    print("MODULO 3 — Classificacao com SVM (Bloco 6)")
    print("=" * 60)

    # --- Criar e treinar o modelo (SVM) ---
    modelo = SVC(kernel="linear", random_state=42)
    modelo.fit(X_treino, y_treino)

    # --- Prever no conjunto de teste ---
    previsoes = modelo.predict(X_teste)

    # --- Avaliar ---
    acuracia = accuracy_score(y_teste, previsoes)
    print(f"\nAcuracia: {acuracia:.2f}")

    print("\nRelatorio de Classificacao:")
    relatorio = classification_report(
        y_teste, previsoes,
        target_names=["LEVE", "MODERADA", "GRAVE"],
        zero_division=0
    )
    print(relatorio)

    return modelo


# =============================================================================
# MODULO 4: Interface interativa para novas interacoes
# =============================================================================
def modulo_interativo(modelo_nb, modelo_svm, vetorizador):
    """Permite ao usuario consultar novas interacoes medicamentosas."""
    print("\n" + "=" * 60)
    print("MODULO 4 — Consulta Interativa de Interacoes")
    print("=" * 60)

    # --- Testar com exemplos pre-definidos ---
    print("\n--- Testando com Exemplos Pre-definidos ---")
    exemplos = [
        "ibuprofeno com metformina",
        "varfarina com aspirina",
        "paracetamol com dipirona",
        "fluoxetina com alcool",
        "vitamina C com paracetamol",
        "carbamazepina com lítio",
    ]

    exemplos_vet = vetorizador.transform(exemplos)
    pred_nb = modelo_nb.predict(exemplos_vet)
    pred_svm = modelo_svm.predict(exemplos_vet)

    print(f"\n{'Interacao':<40} {'Naive Bayes':<15} {'SVM':<15}")
    print("-" * 70)
    for msg, nb, svm in zip(exemplos, pred_nb, pred_svm):
        nb_texto = rotulo_para_texto(nb)
        svm_texto = rotulo_para_texto(svm)
        print(f"  \"{msg}\"{'':>{38 - len(msg)}} {nb_texto:<15} {svm_texto:<15}")

    # --- Modo interativo ---
    print("\n--- Modo Interativo ---")
    print("Digite uma interacao (ex: 'ibuprofeno com varfarina') ou 'sair' para encerrar.\n")

    while True:
        entrada = input(">> Interacao: ").strip()

        if entrada.lower() == "sair":
            print("Encerrando consulta interativa.")
            break

        if len(entrada) == 0:
            print("Digite uma interacao valida.\n")
            continue

        # Vetorizar a entrada do usuario
        entrada_vet = vetorizador.transform([entrada])

        # Prever com ambos os modelos
        pred_nb = modelo_nb.predict(entrada_vet)
        pred_svm = modelo_svm.predict(entrada_vet)

        nb_texto = rotulo_para_texto(pred_nb[0])
        svm_texto = rotulo_para_texto(pred_svm[0])

        print(f"  Naive Bayes: {nb_texto}")
        print(f"  SVM:         {svm_texto}")

        # Se os modelos concordam, exibir com mais confianca
        if nb_texto == svm_texto:
            print(f"  -> Ambos os modelos concordam: gravidade {nb_texto.upper()}")
        else:
            print(f"  -> Modelos divergem: NB diz {nb_texto}, SVM diz {svm_texto}")
        print()


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main():
    print("=" * 60)
    print("  CLASSIFICADOR DE INTERACOES MEDICAMENTOSAS")
    print("  Perfil: Machine Learning aplicado (Blocos 6, 7 e 9)")
    print("=" * 60)

    # --- 1. Carregar dados do CSV ---
    interacoes, gravidades_texto = carregar_dados("interacoes_medicamentosas.csv")

    # --- 2. Converter rotulos para numeros ---
    labels = converter_rotulos(gravidades_texto)

    print(f"Leve (0):     {labels.count(0)}")
    print(f"Moderada (1): {labels.count(1)}")
    print(f"Grave (2):    {labels.count(2)}")

    # --- 3. Vetorizar: texto -> numeros (Bag of Words) ---
    vetorizador = CountVectorizer()
    X = vetorizador.fit_transform(interacoes)

    vocab = vetorizador.get_feature_names_out()
    print(f"\nVocabulario: {len(vocab)} palavras")
    print(f"Matriz: {X.shape[0]} interacoes x {X.shape[1]} palavras")

    # --- 4. Dividir em treino e teste ---
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, labels, test_size=0.3, random_state=42
    )

    print(f"\nTreino: {X_treino.shape[0]} interacoes")
    print(f"Teste:  {X_teste.shape[0]} interacoes")

    # --- 5. Executar Modulo 1: Naive Bayes ---
    modelo_nb = modulo_naive_bayes(X_treino, X_teste, y_treino, y_teste, vetorizador)

    # --- 6. Executar Modulo 2: K-Means ---
    modulo_kmeans(X, interacoes, gravidades_texto)

    # --- 7. Executar Modulo 3: SVM ---
    modelo_svm = modulo_svm(X_treino, X_teste, y_treino, y_teste, vetorizador)

    # --- 8. Comparacao dos modelos ---
    print("\n" + "=" * 60)
    print("COMPARACAO DOS MODELOS")
    print("=" * 60)

    pred_nb = modelo_nb.predict(X_teste)
    pred_svm = modelo_svm.predict(X_teste)

    acc_nb = accuracy_score(y_teste, pred_nb)
    acc_svm = accuracy_score(y_teste, pred_svm)

    print(f"\nAcuracia Naive Bayes: {acc_nb:.2f}")
    print(f"Acuracia SVM:         {acc_svm:.2f}")

    if acc_nb > acc_svm:
        print("-> Naive Bayes obteve melhor desempenho neste dataset.")
    elif acc_svm > acc_nb:
        print("-> SVM obteve melhor desempenho neste dataset.")
    else:
        print("-> Ambos os modelos obtiveram desempenho igual.")

    # --- 9. Executar Modulo 4: Interface interativa ---
    modulo_interativo(modelo_nb, modelo_svm, vetorizador)

    print("\n" + "=" * 60)
    print("  Programa encerrado. Obrigado!")
    print("=" * 60)


# --- Executar o programa ---
if __name__ == "__main__":
    main()
