#######################################################################################################################
# PROGRAMA: kNN-Ricardo.R
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
# tr.class          - Vetor (vector) ou matriz (matrix) contendo as classificações dos dados do conjunto de treinamento.
# te.class          - Vetor (vector) ou matriz (matrix) contendo as classificações dos dados do conjunto de teste.
# dist              - Métrica de distância a ser utilizada pelo algoritmo (padrão = "euclidean"). 
#                     As distâncias permitidas são: "euclidean", "manhattan", "minkowski", "canberra" e "chebyshev".
# k                 - Número inteiro representando o número de vizinhos a serem considerados (padrão = 3).
# lambda            - Número inteiro. Se a métrica de distância for 'minkowski', lambda deve ser definido.
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
# data(iris)
#
# set.seed(123)
#
# train_indices <- sample(1:nrow(iris), size = 0.7 * nrow(iris))
#
# train_data <- iris[train_indices, 1:4]
# test_data <- iris[-train_indices, 1:4]
# train_labels <- iris[train_indices, 5]
# test_labels <- iris[-train_indices, 5]
#
# kNN(train = train_data, 
#     test = test_data, 
#     tr.class = train_labels, 
#     te.class = test_labels, 
#     dist = 'manhattan', 
#     k = 5)
#######################################################################################################################

kNN <- function(train, test, tr.class, te.class, dist = 'euclidean', k = 3, lambda)
{
    # Códigos de erro para entrada de argumentos inválidos 

    if(!is.data.frame(train) && !is.matrix(train))
        stop("Erro001: O objeto 'train' deve ser do tipo data.frame ou matrix.")
   
    if(!is.na(tr.class[1]))
    {
        tr.class    <- as.matrix(tr.class)
        if(nrow(train) != length(tr.class))
            stop("Erro002: Os objetos 'class' e 'train' devem ter o mesmo número de linhas.")
    }

    if(!(dist %in% c('euclidean', 'manhattan', 'minkowski', 'canberra', 'chebyshev')))
        stop("Erro003: 'dist' deve ser: 'euclidean', 'manhattan', 'minkowski', 'canberra', ou 'chebyshev'.")
   
    if (dist == 'minkowski' && is.null(lambda)) 
    {
        stop("Erro004: 'lambda' deve ser definido quando 'dist' é 'minkowski'.")
    }

    # Inicialização dos objetos

    nrow.tr      <- nrow(train)
    nrow.te      <- nrow(test)
    dist_matrix  <- matrix(0, nrow = nrow.te, ncol = nrow.tr) # Distâncias
    predict      <- as.character(nrow.te) # Predições

    # Cálculo das distâncias

    for (i in 1:nrow.te) 
    {
        for (j in 1:nrow.tr) 
        {
            if(dist == 'euclidean'){
                dist_matrix[i, j] <- sqrt(sum((train[j, ] - test[i, ])^2))
            } else if(dist == 'manhattan'){
                dist_matrix[i, j] <- sum(abs(train[j, ] - test[i, ]))
            } else if(dist == 'canberra'){
                dist_matrix[i, j] <- sum(abs(train[j, ] - test[i, ]) / (train[j, ] + test[i, ]))
            } else if(dist == 'chebyshev'){
                dist_matrix[i, j] <- max(abs(train[j, ] - test[i, ]))
            } else if (dist == 'minkowski'){
                dist_matrix[i, j] <- (sum(abs(train[j, ] - test[i, ])^lambda))^(1 / lambda)
            }
        }

        # Determinar os k vizinhos mais próximos
    
        nearest_indices <- order(dist_matrix[i, ])[1:k]
        tb.knn          <- table(tr.class[nearest_indices])
    
        # Classificação das instâncias

        predict[i] <- names(tb.knn[which.max(tb.knn)])
    }
   
    # Matriz de confusão   

    labs <- names(table(te.class))
    conf_matrix <- matrix(0, nrow = length(labs), ncol = length(labs))
    rownames(conf_matrix) <- labs
    colnames(conf_matrix) <- labs
    
    for (k in 1:length(predict))
    {
        pred <- as.character(predict[k])
        true <- as.character(te.class[k])
        conf_matrix[pred, true] <- conf_matrix[pred, true] + 1
    }
   
    # Gráfico da matriz de confusão usando ggplot2

    if (!requireNamespace("ggplot2", quietly = TRUE)) 
    {
      install.packages("ggplot2")
    }
    suppressWarnings(library(ggplot2))

    conf_matrix_melted           <- as.data.frame(as.table(conf_matrix))
    colnames(conf_matrix_melted) <- c("Predicted", "True", "Count")
    conf_matrix_melted$FillColor <- ifelse(conf_matrix_melted$Predicted == conf_matrix_melted$True, 
                                            conf_matrix_melted$Count, 
                                            NA)
    conf_matrix_plot     <- ggplot(conf_matrix_melted, aes(x = True, y = Predicted, fill = FillColor)) +
                                    geom_tile(color = "white") +
                                    geom_text(aes(label = Count), color = "black") +
                                    scale_fill_gradient(low = "gray", high = "salmon", na.value = "white") +
                                    scale_y_discrete(limits = rev(labs)) + 
                                    labs(x = "Classes Atuais", y = "Classes Preditas", title = "Matriz de Confusão") +
                                    theme_minimal() +
                                    theme(axis.text.x = element_text(angle = 45, hjust = 1),
                                          axis.text.y = element_text(angle = 45), 
                                          axis.title.x = element_text(margin = margin(t = 12)),
                                          axis.title.y = element_text(margin = margin(r = 12)),
                                          legend.position = "none")
    print(conf_matrix_plot)

    # Cálculo das métricas de desempenho:

        # - Acurácia (acc)
        # - Precisão (Ou Valor Preditivo Positivo) (precision)
        # - Valor Preditivo Negativo (negative_predictive_value)
        # - Recall (recall)
        # - Especificidade (specificity)
        # - F1-Score (f1_score)
        # - Taxa de Falso Positivo (false_positive_rate)
        # - Taxa de Falso Negativo (false_negative_rate)
        # - Coeficiente Kappa (kappa)
        # - Intervalos de Confiança para Acurácia e Kappa

    acc         <- sum(diag(conf_matrix)) / sum(conf_matrix) 
    n           <- length(te.class)
    z           <- qnorm(1 - 0.05 / 2)
    se          <- sqrt((acc * (1 - acc)) / n)
    ci_low      <- acc - z * se
    ci_high     <- min(acc + z * se, 1)

    precision   <- function(conf_matrix, label) 
    {
        tp      <- conf_matrix[label, label]
        fp      <- sum(conf_matrix[, label]) - tp
        return(tp / (tp + fp))
    }
    
    precision_values <- sapply(labs, function(label) precision(conf_matrix, label))

    recall      <- function(conf_matrix, label) 
    {
        tp      <- conf_matrix[label, label]
        fn      <- sum(conf_matrix[label, ]) - tp
        return(tp / (tp + fn))
    }

    recall_values <- sapply(labs, function(label) recall(conf_matrix, label))

    specificity <- function(conf_matrix, label) 
    {
        tn      <- sum(conf_matrix) - sum(conf_matrix[label, ]) - sum(conf_matrix[, label]) + conf_matrix[label, label]
        fp      <- sum(conf_matrix[, label]) - conf_matrix[label, label]
        return(tn / (tn + fp))
    }

    specificity_values <- sapply(labs, function(label) specificity(conf_matrix, label))

    f1_score    <- function(precision, recall) 
    {
        if(precision + recall == 0) return(0)
        return(2 * (precision * recall) / (precision + recall))
    }
    
    f1_scores   <- mapply(f1_score, precision_values, recall_values)

    false_positive_rate <- function(conf_matrix, label) 
    {
        tp      <- conf_matrix[label, label]
        fp      <- sum(conf_matrix[, label]) - tp
        tn      <- sum(conf_matrix) - sum(conf_matrix[label, ]) - sum(conf_matrix[, label]) + conf_matrix[label, label]
        return(fp / (fp + tn))
    }

    fpr_values  <- sapply(labs, function(label) false_positive_rate(conf_matrix, label))

    false_negative_rate <- function(conf_matrix, label) 
    {
        tp      <- conf_matrix[label, label]
        fn      <- sum(conf_matrix[label, ]) - tp
        return(fn / (fn + tp))
    }

    fnr_values  <- sapply(labs, function(label) false_negative_rate(conf_matrix, label))

    negative_predictive_value <- function(conf_matrix, label) 
    {
        tn      <- sum(conf_matrix) - sum(conf_matrix[label, ]) - sum(conf_matrix[, label]) + conf_matrix[label, label]
        fn      <- sum(conf_matrix[label, ]) - conf_matrix[label, label]
        return(tn / (tn + fn))
    }

    npv_values  <- sapply(labs, function(label) negative_predictive_value(conf_matrix, label))
      
    observed_acc        <- acc
    expected_acc        <- sum(rowSums(conf_matrix) * colSums(conf_matrix)) / (n^2)
   
    kappa               <- (observed_acc - expected_acc) / (1 - expected_acc)
    se_kappa            <- sqrt((observed_acc * (1 - observed_acc)) / (n*(1 - expected_acc)^2))
    ci_kappa_low        <- kappa - z * se_kappa
    ci_kappa_high       <- min(kappa + z * se_kappa, 1)

    # Descrição das distâncias

    dist_description <- switch(dist,
                               'euclidean' = 'Euclidiana',
                               'manhattan' = 'Manhattan',
                               'minkowski' = 'Minkowski',
                               'canberra'  = 'Canberra',
                               'chebyshev' = 'Chebyshev')

   # Gerar texto com os resultados

   conf_matrix_text <- paste("Matriz de Confusão:\n",
                             paste(capture.output(print(conf_matrix)), collapse = "\n"))

    metrics_text <- paste(
                        "Métricas:\n",
                        sprintf("Distância Utilizada: %s", dist_description), "\n",
                        sprintf("Acurácia: %.3f", acc), "\n",
                        sprintf("Intervalo de Confiança da Acurácia: [%.3f, %.3f]", ci_low, ci_high), "\n",
                        sprintf("Coeficiente Kappa: %.3f", kappa), "\n",
                        sprintf("Intervalo de Confiança do Kappa: [%.3f, %.3f]", ci_kappa_low, ci_kappa_high), "\n",
                        sprintf("Recall: %s", paste(sprintf("%.3f", recall_values), collapse = ", ")), "\n",
                        sprintf("Especificidade: %s", paste(sprintf("%.3f", specificity_values), collapse = ", ")), "\n",
                        sprintf("Precisão: %s", paste(sprintf("%.3f", precision_values), collapse = ", ")), "\n",
                        sprintf("Previsão Negativa: %s", paste(sprintf("%.3f", npv_values), collapse = ", ")), "\n",
                        sprintf("Taxa de Falso Positivo: %s", paste(sprintf("%.3f", fpr_values), collapse = ", ")), "\n",
                        sprintf("Taxa de Falso Negativo: %s", paste(sprintf("%.3f", fnr_values), collapse = ", ")), "\n",
                        sprintf("F1-Score: %s", paste(sprintf("%.3f", f1_scores), collapse = ", ")), "\n")
   
   result_text   <- paste(conf_matrix_text, metrics_text, sep = "\n\n")

   # Mostrar a saída como texto
   
   cat(result_text)
   
   # Resultados como lista para compatibilidade

   result <- list('Distance' = dist,
                  'Accuracy' = acc,
                  'Confusion Matrix' = conf_matrix,
                  'Prediction' = as.factor(predict),
                  'Precision' = precision_values,
                  'Recall' = recall_values,
                  'Specificity' = specificity_values,
                  'F1-Score' = f1_scores,
                  'False Positive Rate' = fpr_values,
                  'False Negative Rate' = fnr_values,
                  'Negative Predictive Value' = npv_values,
                  'Confidence Interval of Accuracy' = c(Lower = ci_low, Upper = ci_high),
                  'Kappa' = kappa,
                  'Confidence Interval of Kappa' = c(Lower = ci_kappa_low, Upper = ci_kappa_high))
}

set.seed(123)
train_indices <- sample(1:nrow(iris), size = 0.7 * nrow(iris))
train_data <- iris[train_indices, 1:4]
test_data <- iris[-train_indices, 1:4]
train_labels <- iris[train_indices, 5]
test_labels <- iris[-train_indices, 5]

kNN(train = train_data, test = test_data, tr.class = train_labels, te.class = test_labels, dist = 'euclidean', k = 3)
