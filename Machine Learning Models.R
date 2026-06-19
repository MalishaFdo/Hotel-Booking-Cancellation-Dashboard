# ---------------------
# 1. Load Libraries
# ---------------------
library(tidyverse)
library(caret)
library(pROC)
library(xgboost)
library(nnet)
library(e1071)
library(rpart)
library(randomForest)
library(naivebayes)
library(ROSE)

# ---------------------
# 2. Load and Sample Data
# ---------------------
df <- read.csv("hotel_bookings_updated.csv")

set.seed(42)
df <- df[sample(nrow(df), 5000), ]

# ---------------------
# 3. Preprocess
# ---------------------
df <- df %>% select(-c(company, agent, reservation_status_date, reservation_status,
                       is_canceled_label, is_canceled_numeric, arrival_date, country))

# Convert character to factor
df[] <- lapply(df, function(x) if (is.character(x)) as.factor(x) else x)

# Make sure target is factor
df$is_canceled <- as.factor(df$is_canceled)

# ---------------------
# 4. Handle Class Imbalance with ROSE
# ---------------------
set.seed(42)
df_bal <- ROSE(is_canceled ~ ., data = df, seed = 1)$data
table(df_bal$is_canceled)  # Check balance


# ---------------------
# 5. Train/Test Split
# ---------------------
set.seed(123)
trainIndex <- createDataPartition(df$is_canceled, p = 0.8, list = FALSE)
train_data <- df[trainIndex, ]
test_data  <- df[-trainIndex, ]

# Align factor levels in test to match train
for (col in names(train_data)) {
  if (is.factor(train_data[[col]])) {
    test_data[[col]] <- factor(test_data[[col]], levels = levels(train_data[[col]]))
  }
}

# ---------------------
# 6. Define Training Control
# ---------------------
control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# ---------------------
# 7. Model Training
# ---------------------
# Rename target to "Yes"/"No" for caret
train_data$is_canceled <- factor(ifelse(train_data$is_canceled == 1, "Yes", "No"))
test_data$is_canceled  <- factor(ifelse(test_data$is_canceled == 1, "Yes", "No"))

# 7.1 Random Forest
rf_model <- train(is_canceled ~ ., data = train_data, method = "rf", trControl = control, metric = "ROC")
rf_pred <- predict(rf_model, test_data)
rf_prob <- predict(rf_model, test_data, type = "prob")[, "Yes"]
rf_cm <- confusionMatrix(rf_pred, test_data$is_canceled)
rf_auc <- auc(roc(test_data$is_canceled, rf_prob))

# 7.2 XGBoost
xgb_model <- train(is_canceled ~ ., data = train_data, method = "xgbTree", trControl = control, metric = "ROC")
xgb_pred <- predict(xgb_model, test_data)
xgb_prob <- predict(xgb_model, test_data, type = "prob")[, "Yes"]
xgb_cm <- confusionMatrix(xgb_pred, test_data$is_canceled)
xgb_auc <- auc(roc(test_data$is_canceled, xgb_prob))

# 7.3 SVM
svm_model <- train(is_canceled ~ ., data = train_data, method = "svmRadial", trControl = control, metric = "ROC")
svm_pred <- predict(svm_model, test_data)
svm_prob <- predict(svm_model, test_data, type = "prob")[, "Yes"]
svm_cm <- confusionMatrix(svm_pred, test_data$is_canceled)
svm_auc <- auc(roc(test_data$is_canceled, svm_prob))

# 7.4 Decision Tree
tree_model <- train(is_canceled ~ ., data = train_data, method = "rpart", trControl = control, metric = "ROC")
tree_pred <- predict(tree_model, test_data)
tree_prob <- predict(tree_model, test_data, type = "prob")[, "Yes"]
tree_cm <- confusionMatrix(tree_pred, test_data$is_canceled)
tree_auc <- auc(roc(test_data$is_canceled, tree_prob))

# 7.5 Naive Bayes
nb_model <- train(is_canceled ~ ., data = train_data, method = "naive_bayes", trControl = control, metric = "ROC")
nb_pred <- predict(nb_model, test_data)
nb_prob <- predict(nb_model, test_data, type = "prob")[, "Yes"]
nb_cm <- confusionMatrix(nb_pred, test_data$is_canceled)
nb_auc <- auc(roc(test_data$is_canceled, nb_prob))

# ---------------------
# 8. Compare Results
# ---------------------
results <- data.frame(
  Model = c("Random Forest", "XGBoost", "SVM", "Decision Tree", "Naive Bayes"),
  Accuracy = c(rf_cm$overall["Accuracy"], xgb_cm$overall["Accuracy"],
               svm_cm$overall["Accuracy"], tree_cm$overall["Accuracy"],
               nb_cm$overall["Accuracy"]),
  Recall = c(rf_cm$byClass["Sensitivity"], xgb_cm$byClass["Sensitivity"],
             svm_cm$byClass["Sensitivity"], tree_cm$byClass["Sensitivity"],
             nb_cm$byClass["Sensitivity"]),
  AUC = c(rf_auc, xgb_auc, svm_auc, tree_auc, nb_auc)
)

print(results)

# ---------------------
# 9. Print Confusion Matrices
# ---------------------
cat("\n--- Confusion Matrix: Random Forest ---\n")
print(rf_cm)

cat("\n--- Confusion Matrix: XGBoost ---\n")
print(xgb_cm)

cat("\n--- Confusion Matrix: SVM ---\n")
print(svm_cm)

cat("\n--- Confusion Matrix: Decision Tree ---\n")
print(tree_cm)

cat("\n--- Confusion Matrix: Naive Bayes ---\n")
print(nb_cm)

# Load ggplot2 if not already loaded
library(ggplot2)
library(reshape2)

# Melt results for plotting
results_long <- melt(results, id.vars = "Model")

# Plot Accuracy, Recall, AUC
ggplot(results_long, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison", y = "Score", fill = "Metric") +
  theme_minimal()


# Plot ROC curves
rf_roc <- roc(test_data$is_canceled, rf_prob)
xgb_roc <- roc(test_data$is_canceled, xgb_prob)
svm_roc <- roc(test_data$is_canceled, svm_prob)
tree_roc <- roc(test_data$is_canceled, tree_prob)
nb_roc <- roc(test_data$is_canceled, nb_prob)

plot(rf_roc, col = "blue", main = "ROC Curves for All Models")
plot(xgb_roc, col = "green", add = TRUE)
plot(svm_roc, col = "red", add = TRUE)
plot(tree_roc, col = "orange", add = TRUE)
plot(nb_roc, col = "purple", add = TRUE)
legend("bottomright", legend = c("Random Forest", "XGBoost", "SVM", "Decision Tree", "Naive Bayes"),
       col = c("blue", "green", "red", "orange", "purple"), lwd = 2)


###########################################################################################################


# ---------------------
# 10. Select Numeric Features Only (for clustering)
# ---------------------
df_clust <- df %>%
  select_if(is.numeric) %>%
  na.omit()

# ---------------------
# 11. Normalize the Data
# ---------------------
df_scaled <- scale(df_clust)

# ---------------------
# 12. Determine Optimal Number of Clusters (Elbow Method)
# ---------------------
wss <- sapply(1:10, function(k){
  kmeans(df_scaled, centers = k, nstart = 10)$tot.withinss
})
plot(1:10, wss, type = "b", pch = 19,
     xlab = "Number of Clusters K",
     ylab = "Total Within-Cluster Sum of Squares",
     main = "Elbow Method for Choosing K")

# ---------------------
# 13. Perform K-Means Clustering
# ---------------------
set.seed(123)
km_model <- kmeans(df_scaled, centers = 3, nstart = 25)
df$cluster <- factor(km_model$cluster)

# ---------------------
# 14. Visualize Clusters with PCA
# ---------------------
library(factoextra)
install.packages("factoextra")
fviz_cluster(km_model, data = df_scaled,
             geom = "point", ellipse.type = "convex",
             ggtheme = theme_minimal())


# 15. K-Means Clustering (Bonus Unsupervised)
# ---------------------
# Keep only numeric variables, drop target, and scale
df_clust <- df %>%
  select_if(is.numeric) %>%
  select(-is_canceled) %>%
  na.omit()
df_scaled <- scale(df_clust)

# Elbow method for optimal K
fviz_nbclust(df_scaled, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2) +
  labs(title = "Elbow Method for Optimal K")

# Apply K-means
set.seed(123)
kmeans_result <- kmeans(df_scaled, centers = 3, nstart = 20)
df$cluster <- as.factor(kmeans_result$cluster)

# PCA visualization
pca <- prcomp(df_scaled)
pca_df <- data.frame(pca$x[, 1:2], Cluster = df$cluster)

ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "K-Means Clustering on Hotel Bookings (PCA View)",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal()

# Cluster sizes
kmeans_result$size

# Cluster centers (scaled space)
kmeans_result$centers

# Add cluster label if not already added
df$cluster <- as.factor(kmeans_result$cluster)

# Summarize numeric features by cluster
df %>%
  group_by(cluster) %>%
  summarise(across(where(is.numeric), list(mean = mean, sd = sd), .names = "{.col}_{.fn}"))

# Example: distribution of 'hotel' by cluster
table(df$hotel, df$cluster)

# Example: proportion of 'deposit_type' in each cluster
prop.table(table(df$deposit_type, df$cluster), margin = 2)

# Visualize lead time across clusters
ggplot(df, aes(x = cluster, y = lead_time, fill = cluster)) +
  geom_boxplot() +
  labs(title = "Lead Time by Cluster", y = "Lead Time") +
  theme_minimal()


summarize_clusters <- function(data, cluster_col = "cluster") {
  if (!cluster_col %in% names(data)) {
    stop("Cluster column not found in data.")
  }
  
  cat("===== NUMERIC VARIABLES SUMMARY =====\n")
  numeric_vars <- data %>% select(where(is.numeric), -all_of(cluster_col))
  num_summary <- data %>%
    group_by(.data[[cluster_col]]) %>%
    summarise(across(all_of(names(numeric_vars)),
                     list(mean = ~mean(.x, na.rm = TRUE),
                          sd = ~sd(.x, na.rm = TRUE)),
                     .names = "{.col}_{.fn}"))
  print(num_summary)
  
  cat("\n===== CATEGORICAL VARIABLES SUMMARY =====\n")
  cat_vars <- data %>% select(where(is.factor), -all_of(cluster_col))
  for (var in names(cat_vars)) {
    cat(paste0("\n--- Variable: ", var, " ---\n"))
    print(table(data[[var]], data[[cluster_col]]))
  }
}
# Call after clustering and adding df$cluster
summarize_clusters(df, cluster_col = "cluster")


write.csv(df_final, "hotel_dashboard_ready.csv", row.names = FALSE)

df_final <- hotel_data %>%
  select(is_canceled, hotel, arrival_month, lead_time, adr, guest_type, total_night_stays, cluster)

write.csv(df_final, "hotel_dashboard_ready.csv", row.names = FALSE)
ls()
write.csv(df_bal, "hotel_dashboard_ready.csv", row.names = FALSE)


head(df_bal)
str(df_bal)

df_bal$cluster_label <- dplyr::recode(df_bal$cluster,
                                      "1" = "Early Bookers",
                                      "2" = "Budget Travelers",
                                      "3" = "Luxury Long-Stayers"
)

df_bal$cluster <- as.factor(kmeans_result$cluster)

write.csv(df_bal, "hotel_with_clusters1.csv", row.names = FALSE)

df_bal$is_canceled_label <- ifelse(df_bal$is_canceled == 1, "Canceled", "Not Canceled")
