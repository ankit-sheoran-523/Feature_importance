# Interactive Feature Selection & Model Evaluation Tool

## Project Goal
Developed a comprehensive web application to make **feature selection and model evaluation transparent and accessible**, allowing users to interactively analyze how specific features impact predictive performance.  
The tool emphasizes that **better features often outperform complex models**, highlighting the importance of feature engineering alongside model development.

## Dashboard Preview
Below is a live look at the Streamlit web application dashboard built for this project:

![Streamlit Dashboard](img1.png,img2.png)

---

## Technical Core

- **Automated Preprocessing Pipelines**  
  Dynamically adaptable to diverse datasets, integrating standard scaling and categorical one‑hot encoding.

- **Advanced Feature Selection & Extraction**  
  Techniques include Recursive Feature Elimination (RFE), Sequential Feature Selection (SFS), PCA, t‑SNE, and LDA, benchmarked against baseline models.

- **Task‑Aware Dashboard**  
  Streamlit interface with dynamic logic to classify regression vs. classification problems.

- **Real‑Time Evaluation & Visualization**  
  Delivers scoring metrics (Accuracy, MSE, R²) and interactive visualizations using Seaborn heatmaps and Plotly charts.

---

## Key Performance Metrics
The application provides side‑by‑side comparisons of feature selection techniques, enabling users to identify which approach yields the best performance for their dataset.  

- **Classification:** Accuracy, Confusion Matrix, Classification Report  
- **Regression:** MSE, RMSE, R²  

---

## Repository Structure
The files are organized into logical groups to separate development work from production assets.
```
FeatureSelectionTool/
├── FeatureSelection.py        <-- Main Streamlit App
├── functions.py               <-- Data loading & preprocessing utilities
├── model.py                   <-- Model creation & evaluation functions
├── taskDetection.py           <-- Task type detection & target preprocessing
├── requirements.txt           <-- Project dependencies
└── README.md                  <-- Documentation

### How to Run the Live Dashboard

1.  **Install requirements:** Ensure all necessary Python libraries (streamlit, joblib, sklearn, pandas, etc.) are installed.
2.  **Execute:** Run the application from your terminal:
    ```bash
    streamlit run FeatureSelection.py
    ```