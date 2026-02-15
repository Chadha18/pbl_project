# PetroPhysical Parameter Prediction

This project focuses on the prediction of key **petrophysical parameters** — **Porosity (PHI)** and **Water Saturation (Sw)** — using **machine learning techniques** applied to well-log data.  
The work is carried out as part of **Project Based Learning-3 (PBL-3)**.

---

## Project Overview

Petrophysical parameters play a crucial role in **reservoir characterization** and **hydrocarbon evaluation**.  
Traditional empirical methods such as **Archie’s equation** rely on strong assumptions and often underperform in complex and heterogeneous formations.

This project adopts a **data-driven approach** using machine learning models to improve prediction accuracy by learning non-linear relationships from well-log data.

---

## Objectives

- Predict **Porosity (PHI)** and **Water Saturation (Sw)** from well-log data  
- Perform **Exploratory Data Analysis (EDA)** and data preprocessing  
- Apply **feature selection and outlier handling** techniques  
- Train and compare multiple machine learning models  
- Evaluate model performance using statistical metrics  

---

## Dataset and Preprocessing

- Well-log data is used as input features  
- Invalid water saturation values are recalculated using **Archie’s equation**  
- Outliers are handled using:
  - Interquartile Range (IQR)
  - Local median replacement
  - Global Winsorization  
- Feature correlation analysis is performed to remove redundant inputs  

---

## Machine Learning Models Used

- Random Forest Regressor  
- XGBoost Regressor  
- K-Nearest Neighbors (KNN)  

Hyperparameter tuning is carried out using **RandomizedSearchCV** to optimize model performance.

---

## Evaluation Metrics

Model performance is evaluated using:
- **R² Score**
- **Root Mean Square Error (RMSE)**

Actual vs predicted plots are used for visual validation of predictions.

---



---

## How to Run

1. Clone the repository  
2. Open the Jupyter Notebook located in the `notebook/` folder  
3. Run the cells sequentially to reproduce the results  

---

## Tools & Technologies

- Python  
- Jupyter Notebook  
- NumPy, Pandas  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  

---

## Outcome

- Successful prediction of porosity and water saturation  
- Improved accuracy compared to traditional empirical approaches  
- Structured ML workflow ready for further enhancement and deployment  

---

## Author

**Harshit Chadha**  
B.Tech Computer Science & Engineering  
Manipal University Jaipur  

---
