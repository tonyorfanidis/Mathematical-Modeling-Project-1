# Load necessary libraries
library(e1071)   # For SVM

# Load and preprocess the dataset
car_data <- read.csv("train.csv", stringsAsFactors = FALSE, na.strings = c("", "NA"))
car_data <- na.omit(car_data)  # Remove rows with missing values

# Convert numeric columns to numeric and handle missing values
numeric_columns <- c("price", "mileage", "year", "engine_size")
for (col in numeric_columns) {
  car_data[[col]] <- as.numeric(car_data[[col]])
}

# Ensure no NAs in the dataset
car_data <- na.omit(car_data)

# Select relevant numeric features
features <- car_data[, numeric_columns]

# Scale the features (important for SVM)
features_scaled <- as.data.frame(scale(features))

# Convert target variable (brand) to a factor
car_data$brand <- as.factor(car_data$brand)

# Split data into train (80%) and test (20%)
set.seed(42)
train_indices <- sample(seq_len(nrow(car_data)), size = floor(0.8 * nrow(car_data)))

train_features <- features_scaled[train_indices, ]
test_features <- features_scaled[-train_indices, ]
train_labels <- car_data$brand[train_indices]
test_labels <- car_data$brand[-train_indices]

# Ensure no missing values in features
if (any(is.na(train_features)) | any(is.na(test_features))) {
  stop("Error: Missing values detected in train or test features. Please check data preprocessing.")
}

# Hyperparameter tuning using grid search
cost_values <- c(0.1, 1, 10, 100)      # Different values for cost
gamma_values <- c(0.01, 0.1, 1, 10)    # Different values for gamma
best_accuracy <- 0
best_cost <- NULL
best_gamma <- NULL

# Loop over different cost and gamma values
for (cost_val in cost_values) {
  for (gamma_val in gamma_values) {
    # Train SVM model with current hyperparameters
    svm_model <- svm(x = train_features, y = train_labels, scale = TRUE, 
                     kernel = "radial", cost = cost_val, gamma = gamma_val)
    
    # Make predictions on the test set
    svm_predictions <- predict(svm_model, test_features)
    
    # Evaluate accuracy
    accuracy <- sum(svm_predictions == test_labels) / length(test_labels)
    
    print(paste("Cost =", cost_val, "Gamma =", gamma_val, " Accuracy:", round(accuracy * 100, 2), "%"))
    
    # Update best parameters if accuracy is better
    if (accuracy > best_accuracy) {
      best_accuracy <- accuracy
      best_cost <- cost_val
      best_gamma <- gamma_val
    }
  }
}

# Output best hyperparameters and their corresponding accuracy
print(paste("Best Cost:", best_cost, "Best Gamma:", best_gamma, "with Accuracy:", round(best_accuracy * 100, 2), "%"))

# Train the final model with the best hyperparameters
svm_model_best <- svm(x = train_features, y = train_labels, 
                      scale = TRUE, kernel = "radial", cost = best_cost, gamma = best_gamma)

# Make final predictions
svm_predictions_best <- predict(svm_model_best, test_features)

# Final accuracy
accuracy_best <- sum(svm_predictions_best == test_labels) / length(test_labels)
print(paste("Final SVM Accuracy with best parameters:", round(accuracy_best * 100, 2), "%"))

# Save results to CSV
test_results <- data.frame(Actual_Brand = test_labels, Predicted_Brand = svm_predictions_best)
write.csv(test_results, "svm_predictions_best.csv", row.names = FALSE)

# Display confusion matrix
print("Confusion Matrix:")
table(test_labels, svm_predictions_best)
