# Bank Marketing Prediction
![title](images//images.png)
## Motivation

This project aims to predict whether a client will subscribe to a term deposit based on various features collected during a bank's direct marketing campaign. The dataset provides information on client demographics, past contact results, and other attributes, and the goal is to develop a model that can accurately predict subscription to a term deposit.

## Libraries Used

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib**: For data visualization.
- **seaborn**: For statistical data visualization.
- **scikit-learn**: For machine learning algorithms and evaluation metrics.
- **xgboost**: For gradient boosting model (if used).
- **imblearn**: For handling class imbalance techniques (e.g., SMOTE).

## Files in the Repository

- `bank_data.csv`: The dataset used for model training and evaluation. Contains information on client demographics and past marketing campaign results.
- `notebook.ipynb`: Jupyter Notebook with detailed project analysis, including data exploration, preprocessing, model implementation, and results.
- `requirements.txt`: List of Python libraries and versions used in the project.
- `README.md`: This file, providing an overview and instructions for the project.

## Summary of Analysis

1. **Data Exploration**: Initial data analysis identified key features and the distribution of target classes. Missing values and outliers were addressed.
2. **Model Performance**: Various models were evaluated, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and Support Vector Machine (SVM). 
   - **Logistic Regression** and **Gradient Boosting** showed strong performance, with Gradient Boosting performing best in cross-validation.
   - **Random Forest** and **Decision Tree** had lower F1 Scores, indicating potential issues with model tuning or overfitting.
   - **SVM** performed poorly across all metrics.
3. **Best Parameters**: For Random Forest, the best parameters were identified as `max_depth=20` and `n_estimators=50`.

### Conclusion

The Gradient Boosting model was the most robust, performing well in cross-validation and providing competitive results. However, improvements are needed in recall and overall predictive performance. Future work should focus on addressing class imbalance, feature engineering, and further hyperparameter tuning to enhance model performance.

## Acknowledgements

- **Dataset**: The dataset used in this project is publicly available for research and was created by Paulo Cortez and Sérgio Moro. The details of the dataset are described in [Moro et al., 2011].
- **References**:
  - Moro, S., Laureano, R., & Cortez, P. (2011). Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.
  - Available at: [PDF](http://hdl.handle.net/1822/14838), [Bib](http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt)

## Running the Project

To run the Jupyter Notebook and reproduce the results:

1. Clone this repository:
   ```bash
   git clone https://github.com/dibang99/bank-marketing-prediction.git
   ```

2. Navigate to the project directory
   ```bash
   cd bank-marketing-prediction
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter Notebook
   ```bash
   jupyter notebook notebook.ipynb
   ```

## Blog post
File blog_post.md
