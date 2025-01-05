#Load the required packages
install.packages("caret")
library(caret)

#Load the Iris Dataset
data(iris)

# It contains information about three species of the iris flower: setosa, versicolor, and virginica. 
# Each flower species is represented by four measurable features, 
# and the goal is often to predict the species based on these features.

#Exploratory Data Analysis to understand the data
#View the first 6 rows
head(iris)

#Structure of the dataset
str(iris)

#Summary of the dataset
summary(iris)

# Rows (Observations): 150 (50 observations for each species).
# Columns (Features): 5 (4 numeric features and 1 categorical target).
# Balanced Dataset: Each class (species) has an equal number of observations (50)

#Visualize the dataset for better understanding
library(ggplot2)

# Pair plot (ggpairs from GGally package is a good alternative for pair plots in ggplot2)
install.packages("GGally")
library(GGally)

# Save the pair plot
pair_plot <- ggpairs(iris, aes(color = Species))
pair_plot
ggsave("figures/pair_plot.png", plot = pair_plot, width = 10, height = 8)
# Interpretation of the Pair Plot for the Iris Dataset:
# 
# 1. **Diagonal Elements (Feature Distributions):**
#    - **Petal.Length** and **Petal.Width** show more distinct distributions across species, 
#      especially for **setosa**. This suggests these features are good predictors for classification.
#    - **Sepal.Length** and **Sepal.Width** overlap between **versicolor** and **virginica**, 
#      implying these features alone may not fully separate these two species.
#
# 2. **Pairwise Scatterplots (Relationships Between Features):**
#    - **Petal.Length vs. Petal.Width**: 
#      - Clear separation between all three species. **setosa** forms a distinct cluster, 
#        and while **versicolor** and **virginica** overlap, they are still mostly separable.
#    - **Sepal.Length vs. Sepal.Width**: 
#      - Considerable overlap among **versicolor** and **virginica**, making them harder to separate based 
#        on these features alone.
#    - **Sepal.Length vs. Petal.Length**: 
#      - **setosa** forms a clear cluster, while **versicolor** and **virginica** overlap partially.
#    - **Sepal.Width vs. Petal.Width**: 
#      - Similar trends to **Sepal.Length vs. Petal.Length** with clear clusters for **setosa**.
#
# 3. **Species-Specific Observations:**
#    - **setosa**: 
#      - Clearly separable across all feature pairs. This species can be easily classified.
#    - **versicolor** and **virginica**: 
#      - Show overlap in several feature combinations, especially for **Sepal.Length** and **Sepal.Width**, 
#        making classification more challenging for these two species.
#
# 4. **Key Insights:**
#    - **Petal features** (**Petal.Length** and **Petal.Width**) provide the strongest separation between species 
#      and should be prioritized when building a classification model.
#    - **Sepal features** are less useful for distinguishing **versicolor** and **virginica** due to overlap.
#    - **setosa** is easy to classify, while distinguishing **versicolor** from **virginica** is more challenging.



# Boxplot for Sepal Length by Species
box_plot <- ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Sepal Length by Species")
box_plot
ggsave("figures/box_plot_sepal_length.png", plot = box_plot, width = 8, height = 6)

# Interpretation of the Box Plot for Sepal Length by Species:
# 
# 1. **Setosa:**
#    - **Setosa** has the smallest range of values for **Sepal.Length**.
#    - The median is around 5.0, with a narrow interquartile range (IQR) indicating consistent measurements.
#    - No outliers, making it easy to distinguish **setosa** based on **Sepal.Length**.
#
# 2. **Versicolor:**
#    - **Versicolor** has a larger IQR compared to **setosa**, indicating more variation in **Sepal.Length**.
#    - The median is approximately 5.8, and there are a few mild outliers.
#    - Although the range overlaps with **virginica**, it’s still somewhat distinguishable from **setosa**.
#
# 3. **Virginica:**
#    - **Virginica** shows the largest range and variation in **Sepal.Length**.
#    - The median is around 6.6, and the IQR is wider compared to **setosa** and **versicolor**.
#    - There are some mild outliers, but overall, it is distinct from **setosa** and **versicolor**.
#
# 4. **Key Observations:**
#    - **Setosa** is clearly separated from the other two species with a distinct and smaller range of values for **Sepal.Length**.
#    - **Versicolor** and **Virginica** have overlapping ranges, with **Virginica** having a larger spread.
#    - There are no extreme outliers, but mild outliers are present in both **versicolor** and **virginica**.
#
# 5. **Implications for Classification:**
#    - **Setosa** is easily distinguishable by **Sepal.Length** due to its smaller range.
#    - **Versicolor** and **Virginica** are harder to separate based on **Sepal.Length**, and additional features will be required for accurate classification.




# Scatter plot: Petal Length vs Petal Width by Species
scatter_plot <- ggplot(iris, aes(x = Petal.Length, y = Petal.Width, color = Species)) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(title = "Petal Length vs Petal Width")
scatter_plot
ggsave("figures/scatter_plot_petal_length_width.png", plot = scatter_plot, width = 8, height = 6)
# Interpretation of the Scatter Plot for Petal Length vs. Petal Width by Species:
# 
# 1. **Setosa:**
#    - **Setosa** forms a clear and distinct cluster in the scatter plot, located in the lower left corner.
#    - The points are tightly packed, with relatively small **Petal.Length** and **Petal.Width** values.
#    - This species is easily separable from the others based on both **Petal.Length** and **Petal.Width**.
#
# 2. **Versicolor:**
#    - **Versicolor** is scattered in the middle of the plot with moderate **Petal.Length** and **Petal.Width** values.
#    - There is some overlap with **virginica**, but it generally forms a separate group.
#    - The distribution of points shows moderate variation in both features, but the species is distinguishable from **setosa**.
#
# 3. **Virginica:**
#    - **Virginica** is located towards the upper right of the plot, with higher **Petal.Length** and **Petal.Width** values.
#    - There is noticeable overlap with **versicolor**, but **virginica** tends to have higher petal dimensions.
#    - It is also generally distinguishable from **setosa**, though there is some overlap with **versicolor**.
#
# 4. **Key Observations:**
#    - **Setosa** is clearly separable from **versicolor** and **virginica** due to its small petal dimensions.
#    - **Versicolor** and **Virginica** overlap somewhat, but **virginica** generally has larger petals than **versicolor**.
#    - **Petal Length** and **Petal Width** are strong predictors for species classification, with **setosa** forming the most distinct cluster.
#
# 5. **Implications for Classification:**
#    - **Setosa** can be easily classified based on the combination of **Petal.Length** and **Petal.Width**.
#    - **Versicolor** and **Virginica** require more nuanced modeling techniques, as their petal features overlap.
#    - Features such as **Petal.Length** and **Petal.Width** should be prioritized when building a classification model like KNN.

#Data Preprocessing-Clean the Data before Modeling

# Check for missing values
any(is.na(iris))

# Normalize the numeric features (if needed)
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

iris[1:4] <- as.data.frame(lapply(iris[1:4], normalize))

#Split the Data into Training and Test Sets
# Create a partition
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

#Build the KNN Model
# Train the model
set.seed(123)
knnModel <- train(
  Species ~ ., 
  data = trainData, 
  method = "knn", 
  trControl = trainControl(method = "cv", number = 10), 
  tuneGrid = expand.grid(k = 1:20)
)

# View the results
print(knnModel)

# K = 13 means that for any new observation (such as a new flower in the Iris dataset), 
# the KNN algorithm will look at the 13 nearest training samples (neighbors) to make a classification decision.

#Evaluate the Model
#Train the model on unseen data and evaluate its performance
# Make predictions
knnPredictions <- predict(knnModel, testData)

# Confusion matrix
confusionMatrix(knnPredictions, testData$Species)

# Interpretation of Confusion Matrix and Model Performance Statistics:
#
# 1. **Confusion Matrix:**
#    - **Setosa**: The model predicted all 10 instances of **setosa** correctly.
#    - **Versicolor**: The model predicted 10 instances of **versicolor** correctly, but 2 instances were misclassified as **virginica**.
#    - **Virginica**: The model predicted 8 instances of **virginica** correctly.
#
# 2. **Overall Accuracy:**
#    - The model achieved an **Accuracy** of **0.9333** (93.33%), meaning that 93.33% of the predictions were correct.
#    - The **95% Confidence Interval (CI)** for accuracy is (0.7793, 0.9918), which indicates a high level of certainty in the accuracy estimate.
#    - The **No Information Rate (NIR)** is 0.3333 (33.33%), which represents the accuracy you would expect if you simply guessed the most common class. The model's accuracy significantly exceeds this baseline, as indicated by the **P-Value [Acc > NIR]** of **8.747e-12** (extremely low).
#    
# 3. **Kappa Statistic:**
#    - The **Kappa** value is **0.9**, which indicates excellent agreement between the predicted and actual class labels beyond what would be expected by chance. A value close to 1 suggests near-perfect prediction.
#
# 4. **Statistics by Class:**
#    - **Setosa**:
#        - **Sensitivity**: 1.0 (100%) – The model correctly identified all instances of **setosa**.
#        - **Specificity**: 1.0 (100%) – The model correctly identified all non-setosa instances.
#        - **Positive Predictive Value (PPV)**: 1.0 (100%) – Every prediction of **setosa** was correct.
#        - **Negative Predictive Value (NPV)**: 1.0 (100%) – All non-setosa predictions were correct.
#        - **Balanced Accuracy**: 1.0 (100%) – The model performs perfectly in terms of correctly identifying **setosa**.
#
#    - **Versicolor**:
#        - **Sensitivity**: 1.0 (100%) – The model correctly identified all **versicolor** instances (except for the 2 misclassified as **virginica**).
#        - **Specificity**: 0.9 (90%) – The model had a 90% success rate in correctly identifying non-**versicolor** instances.
#        - **Positive Predictive Value (PPV)**: 0.8333 (83.33%) – 83.33% of **versicolor** predictions were correct.
#        - **Negative Predictive Value (NPV)**: 1.0 (100%) – All non-**versicolor** predictions were correct.
#        - **Balanced Accuracy**: 0.95 (95%) – The model has strong performance in detecting **versicolor**, despite a few misclassifications.
#
#    - **Virginica**:
#        - **Sensitivity**: 0.8 (80%) – The model identified 80% of the **virginica** instances correctly.
#        - **Specificity**: 1.0 (100%) – The model correctly identified all non-**virginica** instances.
#        - **Positive Predictive Value (PPV)**: 1.0 (100%) – All predictions of **virginica** were correct.
#        - **Negative Predictive Value (NPV)**: 0.9091 (90.91%) – 90.91% of non-**virginica** predictions were correct.
#        - **Balanced Accuracy**: 0.9 (90%) – The model performs well in classifying **virginica** but is slightly less sensitive compared to **setosa** and **versicolor**.
#
# 5. **Key Observations:**
#    - The model performs exceptionally well overall, with a **high accuracy** of 93.33%, indicating it is a reliable classifier.
#    - **Setosa** is perfectly classified, while **versicolor** and **virginica** have slight misclassifications, particularly with **versicolor** being confused with **virginica**.
#    - The **Kappa statistic** of 0.9 indicates strong agreement between predicted and actual labels, and the **Balanced Accuracy** scores show that the model handles class imbalances well.
#    - The **specificity** and **NPV** are also high across all classes, indicating the model is good at identifying true negatives.


#Making Predictions
#Using the model to classify new data points.
# Example new observation
newObservation <- data.frame(Sepal.Length = 0.5, Sepal.Width = 0.4, Petal.Length = 0.7, Petal.Width = 0.2)
newObservation
# Predict the species
predict(knnModel, newObservation)

# Interpretation of Prediction for New Observation:
#
# 1. **New Observation:**
#    - The new observation has the following feature values:
#        - Sepal Length = 0.5
#        - Sepal Width = 0.4
#        - Petal Length = 0.7
#        - Petal Width = 0.2
#
# 2. **Prediction:**
#    - The KNN model predicts that this new observation belongs to the species **"versicolor"**.
#
# 3. **Levels:**
#    - The predicted class is one of the three species in the dataset:
#        - setosa
#        - versicolor
#        - virginica
#
# 4. **Conclusion:**
#    - Based on the KNN classification model, the new observation is most similar to the **versicolor** species based on its feature values.



