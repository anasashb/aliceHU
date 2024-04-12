[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# **A**utomated **L**earning for **I**nsightful **C**omparison and **E**valuation (ALICE)
Automated Learning for Insightful Comparison and Evaluation (ALICE) merges conventional feature selection and the concept of inter-rater agreeability in a simple, user-friendly manner to seek insights into black box Machine Learning models. Currently supports (and has been tested on) Scikit-Learn and Keras models.

![ALICE Framework Visualized](/alice_framework_graph.png)

Results included from the repository are from three experiments on the Telco Customer Churn dataset: 
- Mulit-Layer Perceptron (MLP) vs. Logistic Regression (Logit)
- Multi-Layer Perceptron (MLP) vs. Random Forest Classifier (RFC)
- Random Forest Classifier (RFC) vs. Logistic Regression (Logit)

![ALICE Experiment Results](/experiment_results.png)


- - -

## Main directory:

**Notebooks**<br>
- `customer_churn_test.ipynb` - Jupyter Notebook for experiments and use demonstration / instructions
- `results_analysis.ipynb` - Jupyter Notebook demonstrating experiment results and plots
- `customer_churn_dataprocessing.ipynb` - Jupyter Notebook for transparency of data cleaning and manipualtion

**Folders**<br>
- `alice` - Code modules for the framework
- `clean_data` - Saved train-test sets
- `test_results` - Saved experiment results
  - `test_results/experiment_results_20240301_1/experiment_results_20240301_1.json` - MLP vs. Logit Experiment
  - `test_results/experiment_results_20240302_1/experiment_results_20240302_1.json` - MLP vs. RFC Experiment
  - `test_results/experiment_results_20240302_2/experiment_results_20240302_2.json` - RFC vs. Logit Experiment

**Files**<br>
- `class_telco.pkl` - Processed and cleaned Telecom customer churn dataset for classification
- `reg_telco.pkl` - Processed and cleaned Telecom customer churn dataset for regression
- `Telco_customer_churn.xlsx` - Raw data
- `requirements.txt` - Required python libraries and their versions

- - -

## Reproducibility:
Note that the results may not be exactly reproducible due to tha nature of neural networks, random forests and their optimization.

For re-running the experiments, or testing the framework:

1) Make sure to set up a virtual Python environment and install the required packages

```python
! pip install -r requirements.txt
```

2) Run the `customer_churn_test.ipynb` Notebook.
    - Re-running sections 1, 2, 3, and 4 are mandatory to be able to re-run the experiments.
    - Experiments are contained under section 5. Given the computationally costly nature of the models, section 7 includes two simpler models, Logistic Regression and a Decision Tree Classifier, for those who want to quickly test the functionalities of the framework.
