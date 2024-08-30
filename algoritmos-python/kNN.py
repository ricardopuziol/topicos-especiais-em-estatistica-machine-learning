#######################################################################################################################
# PROGRAMA: kNN.py
# 
# USO: Algoritmo de Classificação Supervisionada usando k-NN (k-vizinhos mais próximos)
#
# AUTOR: Ricardo Puziol de Oliveira (rp.oliveira@unesp.br)
#
# DATA: 30 de Agosto de 2024
#######################################################################################################################
#
# DESCRIÇÃO:
#
# Este script realiza a classificação de dados usando o algoritmo k-NN. O k-NN é um método de aprendizado supervisionado
# que classifica novas amostras com base na similaridade com amostras conhecidas. As distâncias permitidas incluem 
# Euclidiana, Manhattan, Minkowski, Canberra e Chebyshev. O script também calcula e visualiza a matriz de confusão, 
# e fornece medidas de desempenho como acurácia, precisão, recall, F1-score e coeficiente Kappa.
#
# ENTRADA:
#
# train             - Matriz (matrix) ou quadro de dados (data.frame) contendo os dados do conjunto de treinamento.
# test              - Matriz (matrix) ou quadro de dados (data.frame) contendo os dados do conjunto de teste.
# tr_class          - Vetor (vector) ou matriz (matrix) contendo as classificações dos dados do conjunto de treinamento.
# te_class          - Vetor (vector) ou matriz (matrix) contendo as classificações dos dados do conjunto de teste.
# dist              - Métrica de distância a ser utilizada pelo algoritmo (padrão = "euclidean"). 
#                     As distâncias permitidas são: "euclidean", "manhattan", "minkowski", "canberra" e "chebyshev".
# k                 - Número inteiro representando o número de vizinhos a serem considerados (padrão = 3).
# lambda_param      - Número inteiro. Se a métrica de distância for 'minkowski', lambda deve ser definido.
#
# SAÍDA:
#
# result            - Lista contendo as seguintes medidas de desempenho:
#                     - Acurácia
#                     - Precisão
#                     - Recall
#                     - F1-Score
#                     - Especificidade
#                     - Valor Preditivo Negativo 
#                     - Taxa de Falso Positivo
#                     - Taxa de Falso Negativo
#                     - Coeficiente Kappa
#                     - Intervalos de Confiança para Acurácia e Kappa
#
# conf_matrix_plot  - Representação gráfica da matriz de confusão, visualizando a performance da classificação.
#
# EXEMPLO DE USO:
#
# from sklearn.datasets import load_iris
# 
# iris = load_iris()
# iris_data = iris.data
# iris_target = iris.target
# 
# np.random.seed(123)  # Semente para reprodução do algoritmo
# 
# train_indices = np.random.choice(range(len(iris_data)), size=int(0.7 * len(iris_data)), replace=False)
# train_data = iris_data[train_indices]
# test_data = iris_data[~np.isin(range(len(iris_data)), train_indices)]
# train_labels = iris_target[train_indices]
# test_labels = iris_target[~np.isin(range(len(iris_data)), train_indices)]
#
# kNN(train=train_data, test=test_data, tr_class=train_labels, te_class=test_labels, dist='euclidean', k=3)
#######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

def kNN(train, test, tr_class, te_class, dist='euclidean', k=3, lambda_param=None):

    # Códigos de erro para entrada de argumentos inválidos

    if not isinstance(train, np.ndarray) or not isinstance(test, np.ndarray):
        raise ValueError("Erro001: 'train' e 'test' devem ser do tipo ndarray.")

    if tr_class is not None:
        if len(tr_class) != train.shape[0]:
            raise ValueError("Erro002: Os objetos 'tr_class' e 'train' devem ter o mesmo número de linhas.")

    if dist not in ['euclidean', 'manhattan', 'minkowski', 'canberra', 'chebyshev']:
        raise ValueError("Erro003: 'dist' deve ser: 'euclidean', 'manhattan', 'minkowski', 'canberra', ou 'chebyshev'.")

    if dist == 'minkowski' and lambda_param is None:
        raise ValueError("Erro004: 'lambda_param' deve ser definido quando 'dist' é 'minkowski'.")

    # Inicialização dos objetos

    nrow_tr = train.shape[0]
    nrow_te = test.shape[0]
    dist_matrix = np.zeros((nrow_te, nrow_tr))  # Distâncias
    predict = [] # Predições

    # Cálculo das distâncias

    for i in range(nrow_te):
        for j in range(nrow_tr):
            if dist == 'euclidean':
                dist_matrix[i, j] = np.sqrt(np.sum((train[j, :] - test[i, :]) ** 2))
            elif dist == 'manhattan':
                dist_matrix[i, j] = np.sum(np.abs(train[j, :] - test[i, :]))
            elif dist == 'canberra':
                dist_matrix[i, j] = np.sum(np.abs(train[j, :] - test[i, :]) / (train[j, :] + test[i, :]))
            elif dist == 'chebyshev':
                dist_matrix[i, j] = np.max(np.abs(train[j, :] - test[i, :]))
            elif dist == 'minkowski':
                dist_matrix[i, j] = np.power(np.sum(np.abs(train[j, :] - test[i, :]) ** lambda_param), 1 / lambda_param)

        # Determinar os k vizinhos mais próximos

        nearest_indices = np.argsort(dist_matrix[i, :])[:k]
        nearest_classes = np.array(tr_class)[nearest_indices]
        unique, counts = np.unique(nearest_classes, return_counts=True)
        tb_knn = dict(zip(unique, counts))

        # Classificação das instâncias

        predict.append(max(tb_knn, key=tb_knn.get))

    predict = np.array(predict)

    # Matriz de confusão

    labels = np.unique(te_class)
    conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    
    for true_label, pred_label in zip(te_class, predict):
        conf_matrix[int(true_label), int(pred_label)] += 1

    # Plotar a matriz de confusão

    plt.figure(figsize=(10, 7))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, conf_matrix[i, j], horizontalalignment='center', color='black')

    plt.xlabel('Classes Preditas')
    plt.ylabel('Classes Atuais')
    plt.show()

    # Cálculo das métricas de desempenho

    acc = np.trace(conf_matrix) / np.sum(conf_matrix)
    n = len(te_class)
    z = 1.96  # Aproximação para 95% CI
    se = np.sqrt((acc * (1 - acc)) / n)
    ci_low = acc - z * se
    ci_high = min(acc + z * se, 1)

    def precision(conf_matrix, label):
        tp = conf_matrix[label, label]
        fp = np.sum(conf_matrix[:, label]) - tp
        return tp / (tp + fp)

    def recall(conf_matrix, label):
        tp = conf_matrix[label, label]
        fn = np.sum(conf_matrix[label, :]) - tp
        return tp / (tp + fn)

    def specificity(conf_matrix, label):
        tn = np.sum(conf_matrix) - np.sum(conf_matrix[label, :]) - np.sum(conf_matrix[:, label]) + conf_matrix[label, label]
        fp = np.sum(conf_matrix[:, label]) - conf_matrix[label, label]
        return tn / (tn + fp)

    def f1_score(precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def false_positive_rate(conf_matrix, label):
        tp = conf_matrix[label, label]
        fp = np.sum(conf_matrix[:, label]) - tp
        tn = np.sum(conf_matrix) - np.sum(conf_matrix[label, :]) - np.sum(conf_matrix[:, label]) + conf_matrix[label, label]
        return fp / (fp + tn)

    def false_negative_rate(conf_matrix, label):
        tp = conf_matrix[label, label]
        fn = np.sum(conf_matrix[label, :]) - tp
        return fn / (fn + tp)

    def negative_predictive_value(conf_matrix, label):
        tn = np.sum(conf_matrix) - np.sum(conf_matrix[label, :]) - np.sum(conf_matrix[:, label]) + conf_matrix[label, label]
        fn = np.sum(conf_matrix[label, :]) - conf_matrix[label, label]
        return tn / (tn + fn)

    precision_values = [precision(conf_matrix, i) for i in range(len(labels))]
    recall_values = [recall(conf_matrix, i) for i in range(len(labels))]
    specificity_values = [specificity(conf_matrix, i) for i in range(len(labels))]
    f1_scores = [f1_score(p, r) for p, r in zip(precision_values, recall_values)]
    fpr_values = [false_positive_rate(conf_matrix, i) for i in range(len(labels))]
    fnr_values = [false_negative_rate(conf_matrix, i) for i in range(len(labels))]
    npv_values = [negative_predictive_value(conf_matrix, i) for i in range(len(labels))]

    observed_acc = acc
    expected_acc = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (n ** 2)
    kappa = (observed_acc - expected_acc) / (1 - expected_acc)
    se_kappa = np.sqrt((observed_acc * (1 - observed_acc)) / (n * (1 - expected_acc) ** 2))
    ci_kappa_low = kappa - z * se_kappa
    ci_kappa_high = min(kappa + z * se_kappa, 1)

    # Descrição das distâncias

    dist_description = {
        'euclidean': 'Euclidiana',
        'manhattan': 'Manhattan',
        'minkowski': 'Minkowski',
        'canberra': 'Canberra',
        'chebyshev': 'Chebyshev'
    }[dist]

    # Gerar texto com os resultados

    conf_matrix_text = "Matriz de Confusão:\n" + str(conf_matrix)
    metrics_text = (
        f"Métricas:\n"
        f"Distância Utilizada: {dist_description}\n"
        f"Acurácia: {acc:.3f}\n"
        f"Intervalo de Confiança da Acurácia: [{ci_low:.3f}, {ci_high:.3f}]\n"
        f"Coeficiente Kappa: {kappa:.3f}\n"
        f"Intervalo de Confiança do Kappa: [{ci_kappa_low:.3f}, {ci_kappa_high:.3f}]\n"
        f"Recall: {', '.join(f'{v:.3f}' for v in recall_values)}\n"
        f"Especificidade: {', '.join(f'{v:.3f}' for v in specificity_values)}\n"
        f"Precisão: {', '.join(f'{v:.3f}' for v in precision_values)}\n"
        f"Previsão Negativa: {', '.join(f'{v:.3f}' for v in npv_values)}\n"
        f"Taxa de Falso Positivo: {', '.join(f'{v:.3f}' for v in fpr_values)}\n"
        f"Taxa de Falso Negativo: {', '.join(f'{v:.3f}' for v in fnr_values)}\n"
        f"F1-Score: {', '.join(f'{v:.3f}' for v in f1_scores)}\n"
    )

    print(conf_matrix_text)
    print(metrics_text)

    # Resultados como dicionário

    result = {
        'Distance': dist,
        'Accuracy': acc,
        'Confusion Matrix': conf_matrix,
        'Prediction': predict,
        'Precision': precision_values,
        'Recall': recall_values,
        'Specificity': specificity_values,
        'F1-Score': f1_scores,
        'False Positive Rate': fpr_values,
        'False Negative Rate': fnr_values,
        'Negative Predictive Value': npv_values,
        'Confidence Interval of Accuracy': {'Lower': ci_low, 'Upper': ci_high},
        'Kappa': kappa,
        'Confidence Interval of Kappa': {'Lower': ci_kappa_low, 'Upper': ci_kappa_high}
    }

    return result

# Exemplo de uso com o conjunto de dados Iris

from sklearn.datasets import load_iris

iris = load_iris()
iris_data = iris.data
iris_target = iris.target

np.random.seed(123)  # Semente para reprodução do algoritmo

train_indices = np.random.choice(range(len(iris_data)), size=int(0.7 * len(iris_data)), replace=False)
train_data = iris_data[train_indices]
test_data = iris_data[~np.isin(range(len(iris_data)), train_indices)]
train_labels = iris_target[train_indices]
test_labels = iris_target[~np.isin(range(len(iris_data)), train_indices)]

result = kNN(train=train_data, test=test_data, tr_class=train_labels, te_class=test_labels, dist='euclidean', k=3)
