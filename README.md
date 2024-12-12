# Diabetes Prediction Project

Welcome to the **Diabetes Prediction** project repository! This project utilizes exploratory data analysis (EDA) and machine learning to predict diabetes based on patient data.

## Project Overview

The goal of this project is to build a machine learning model capable of predicting whether a patient has diabetes based on diagnostic measurements. This project involves:
- Exploratory Data Analysis (EDA) to uncover insights from the dataset.
- Preprocessing the data for machine learning.
- Building and evaluating machine learning models.

## Dataset

The dataset used in this project comes from the Iraqi society as they data were acquired from the laboratory of Medical City Hospital and (the Specializes Center for Endocrinology and Diabetes-Al-Kindy Teaching Hospital (https://data.mendeley.com/datasets/wj9rwkp9c2/1)]. It contains information on:
- Sugar level blood
- Age
- Gender
- Creatinine ratio(Cr)
- Urea
- Body Mass Index (BMI)
- Cholesterol
- Fasting lipid profile
- Outcome (indicating if the patient has diabetes: 1 = yes, 0 = no)

## Key Steps

### 1. Exploratory Data Analysis (EDA)
- Visualized the distribution of each feature.
- Analyzed correlations between features.


### 2. Data Preprocessing
- Normalized feature values to ensure consistent scaling.
- Dropped irrelevant columns.
- Split the data into training and testing sets.
- Encoded categorical features.

### 3. Machine Learning Modeling
- Tested multiple classification algorithms, including:
  - Logistic Regression
  - k-Nearest Neighbors (k-NN)
  - eXtreme Gradient Boosting
  - Decision Tree
  - Random Forest
  - Support Vector 
  - Gaussian Naive Bayes
- Performed cross validation 
- Evaluated models using metrics such as accuracy, precision, recall, and F1-score.
- Selected the best-performing model for final predictions.

## Results

- The [Gradient Boosting Classifier]([https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html]) achieved the highest accuracy and robustness among the models tested.

## Dependencies

The project is implemented in Python and uses the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

To install the required dependencies, use the following command:
```bash
pip install -r requirements.txt
```

## Usage

To explore the notebook and reproduce the analysis:
1. Clone this repository:
   ```bash
   git clone https://github.com/RuthOlasupo/Diabetes-Prediction.git
   ```
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook EDA_&_ML_Model_on_Diabetes_Prediction.ipynb
   ```
3. Follow the steps in the notebook to run the analysis and models.

## Future Work

- Incorporate additional datasets to improve model generalizability.
- Experiment with advanced machine learning algorithms such as Neural Networks.
- Deploy the model as a web application for real-time predictions.

## Contributing

Contributions are welcome! If you'd like to collaborate, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Author:** Ruth Olasupo  
Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/rutholasupo/) or check out my [portfolio](https://github.com/RuthOlasupo).

