# Load necessary libraries
install.packages("class")  # Install kNN package if not already installed
library(class)

# Load the dataset
car_data <- read.csv("train.csv", stringsAsFactors = FALSE, na.strings = c("", "NA"))

# Remove rows with missing values
car_data <- na.omit(car_data)

# Convert all required columns to numeric safely (avoid NAs)
numeric_columns <- c("price", "mileage", "year", "engine_size")
for (col in numeric_columns) {
  car_data[[col]] <- as.numeric(car_data[[col]])
}

# Ensure there are no NA values after conversion
car_data <- na.omit(car_data)

# Select relevant numeric features
features <- car_data[, numeric_columns]

# Normalize features
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

# Set the number of neighbors (k)
# Try different values of k
k_values <- c(7, 9, 11, 15, 19)
accuracy_results <- c()

for (k in k_values) {
  knn_predictions <- knn(train = train_features, test = test_features, cl = train_labels, k = k)
  accuracy <- sum(knn_predictions == test_labels) / length(test_labels)
  accuracy_results <- c(accuracy_results, accuracy)
  print(paste("k =", k, " Accuracy:", round(accuracy * 100, 2), "%"))
}

# Select the best k (highest accuracy)
best_k <- k_values[which.max(accuracy_results)]
print(paste("Best k =", best_k))

# Train final kNN model with the best k
knn_predictions <- knn(train = train_features, test = test_features, cl = train_labels, k = best_k)

# Evaluate accuracy
final_accuracy <- sum(knn_predictions == test_labels) / length(test_labels)
print(paste("Final kNN Accuracy:", round(final_accuracy * 100, 2), "%"))

# Save results
test_results <- data.frame(Actual_Brand = test_labels, Predicted_Brand = knn_predictions)
write.csv(test_results, "knn_predictions.csv", row.names = FALSE)

# Display confusion matrix
print("Confusion Matrix:")
print(table(Actual = test_labels, Predicted = knn_predictions))
