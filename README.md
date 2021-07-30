# Feature Importance & Selection
Frequently, we think of the only focus of machine learning models is to optimize model performance on selected metrics. However, Understanding how a model works and how features/predictors contribute to its prediction is just as importance.<br>
In this project, I implemented various feature importance techniques and its application in python and made the summary document in jupyter notebook.<br>

Please check out the summary **[here](https://github.com/anawatbk/feature-importance-summary/blob/master/code/featimp.ipynb)**



## Topics cover
- Why do we need Feature Importance?
- What is Feature Importance?  
- Types of Feature Importance 
  - Model-based strategies 
    - Model-Specific
    - Model-Agnostic
      - Permutation Importance
      - Drop Importance
  - Calculate Feature Importance by working directly from the data (model-free)
    - Spearman's rank correlation coefficient 
    - mRMR
- Feature Importance Algorithms Comparison
    - RF vs Permutation vs SHAP vs Drop vs mRMR
- Automatic Feature Selection
- Variance and empirical p-values for feature importances
</br></br>
#### Author's  Note
code/featimp.py includes all of algorithm and support code for the report.