🧬 Data Analysis and Breast Cancer Prediction Using SVM

This project focuses on analyzing the Breast Cancer Wisconsin (Diagnostic) dataset to classify breast cancer tumors as malignant or benign using a Support Vector Machine (SVM) model.
The project involves data preprocessing, feature selection, feature scaling, model training, and evaluation.

📊 1. Data Preprocessing
1.1 Loading the Dataset

The dataset is loaded using Pandas from the file data.csv.

1.2 Initial Exploration

The dataset’s structure is examined using:

df.shape, df.head(), df.dtypes


Detailed information is obtained with:

df.info()

1.3 Handling Missing Values

Missing values are identified using:

df.isnull().sum()


The column Unnamed: 32 (containing only missing values) is dropped.

Missing values are rechecked to confirm successful removal.

1.4 Handling Duplicate Rows

Duplicate rows are detected using:

df.duplicated().sum()

1.5 Dropping Unnecessary Columns

The id column is dropped as it is not relevant for model training.

1.6 Encoding Target Variable

The diagnosis column is encoded into numeric values:

M (Malignant) → 1

B (Benign) → 0

1.7 Correlation Analysis

A heatmap of the correlation matrix is created using Seaborn to visualize feature relationships.

1.8 Feature Selection

The correlation of each feature with the target variable (diagnosis) is calculated.

Features with an absolute correlation greater than 0.5 with the target are selected for further analysis.

⚙️ 2. Feature Engineering and Scaling
2.1 Scaling the Features

Selected features are scaled using StandardScaler to standardize their ranges and improve model performance.

2.2 Creating the Final DataFrame

The scaled features are combined with the target variable to form the final dataset dfn.

🤖 3. Machine Learning Model: Support Vector Machine (SVM)
3.1 Splitting the Data

The data is split into training (80%) and testing (20%) sets using:

train_test_split()

3.2 Initializing and Training the Model

An SVM model with a linear kernel is initialized and trained on the training data.

3.3 Making Predictions

Predictions are made for both the training and test datasets.

3.4 Evaluating the Model

The model is evaluated using:

Confusion Matrix

Accuracy

F1-Score for both classes (Malignant and Benign)

📈 4. Model Performance Metrics
4.1 Training Data Performance

Confusion Matrix: Summarizes prediction outcomes on training data.

Accuracy: Measures the proportion of correct predictions.

F1-Score (Malignant): Harmonic mean of precision and recall for class 1.

F1-Score (Benign): Harmonic mean of precision and recall for class 0.

4.2 Test Data Performance

Confusion Matrix: Summarizes prediction outcomes on test data.

Accuracy: Measures model performance on unseen data.

F1-Score (Malignant): Indicates performance for correctly identifying malignant tumors.

F1-Score (Benign): Indicates performance for correctly identifying benign tumors.

⚠️ 5. Challenges Faced

Feature Selection:
Determining the most relevant features based on correlation required careful analysis and domain understanding.

Feature Scaling:
Ensuring all features were standardized was crucial for optimal SVM performance.

Model Evaluation:
Evaluating the model using multiple metrics helped identify strengths and areas needing improvement.

✅ 6. Conclusion

This project demonstrated the end-to-end process of:

Cleaning and preprocessing the Breast Cancer Wisconsin (Diagnostic) dataset

Selecting the most relevant features

Scaling the data

Building and evaluating an SVM classifier

The SVM model effectively classified breast cancer tumors as malignant or benign, achieving strong performance metrics.
The workflow highlights the importance of proper data preprocessing, feature scaling, and model evaluation in developing reliable predictive models for medical diagnosis.

📂 7. Tools and Libraries Used

Python 3.x

Pandas – Data manipulation and analysis

NumPy – Numerical operations

Matplotlib & Seaborn – Data visualization

Scikit-learn (sklearn) – Machine learning tools and model evaluation
