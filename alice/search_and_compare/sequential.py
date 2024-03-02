import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import log_loss
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (1 = INFO, 2 = WARNING, 3 = ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TensorFlow messages

np.random.seed(66)
tf.random.set_seed(66)

plt.rcParams["figure.figsize"] = [15, 8]
plt.rcParams.update({"font.family": "serif"})
plt.style.use("ggplot")

from alice.agreeability.classify import cohen_kappa
from alice.agreeability.regress import pearson
from alice.metrics.classify import accuracy, f1, precision, recall
from alice.metrics.regress import mae, mse, rmse
from alice.testing.classify import mcnemar_binomial, mcnemar_chisquare
from alice.testing.regress import t_test
from alice.utils.feature_lists import (
    dummy_grouper,
    feature_fixer,
    feature_list_flatten,
)
from alice.utils.model_training import ModelTrainer


class BackEliminator:
    """
    The class is built for conducting backwards feature elimination in
    combination with model agreeability.

    Args:
        X (pd.DataFrame): A pandas dataframe containing predictors.
        y (pd.DataFrame): A pandas dataframe containing target.
        validation_data (tuple): A tuple of validation data
                                 (X_val, y_val).
        task_type (str): String for task type. Available options -
                         'classification' or 'regression'.
        criterion (str): String for intra-model evaluation criterion.
                         Available options: ('mse', 'rmse', 'mae',
                                            'accuracy', 'precision',
                                            'recall', 'f1')
        agreeability (str): String for inter-model comparison.
                            Available options: 'pearson', 'cohen_kappa'
        dummy_list (list): List of lists containing column names (str)
                           of dummy features generated from a
                           categorical variable. (Optional).
        features_to_fix (list): List containing column names (str) of
                                features that will be excluded from
                                feature elimination and thus always
                                included in modeling. (Optional)

    Regression Example:
        seeker = BackEliminator(
            X=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            task_type='regression',
            criterion='rmse',
            agreeability='pearson',
            dummy_list=[
                ['dummy_1_from_variable_1', 'dummy_2_from_variable_1'],
                [
                    'dummy_1_from_variable_2',
                    'dummy_2_from_variable_2',
                    'dummy_3_from_variable_2'
                ]
            ],
            features_to_fix=[
                'variable_3',
                'variable_4'
            ]
        )

    Classification Example:
        seeker = Backeliminator(
            X=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            task_type='classification',
            criterion='f1',
            agreeability='cohen_kappa',
            dummy_list=[
                ['dummy_1_from_variable_1', 'dummy_2_from_variable_1'],
                [
                    'dummy_1_from_variable_2',
                    'dummy_2_from_variable_2',
                    'dummy_3_from_variable_2']
            ],
            features_to_fix=[
                'variable_3',
                'variable_4'
            ]
        )
    """

    def __init__(
        self,
        X=None,
        y=None,
        validation_data=None,
        task_type=None,
        criterion=None,
        agreeability=None,
        dummy_list=None,
        features_to_fix=None,
    ):

        self.X = X
        self.y = y

        if validation_data:
            self.validation_data = validation_data
            self.X_val = self.validation_data[0]
            self.y_val = self.validation_data[1]

            self.criterion_registry = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "log_loss": log_loss,
            }
        self.criterion = criterion

        self.agreeability_registry = {
            "pearson": pearson,
            "cohen_kappa": cohen_kappa,
        }
        self.agreeability = agreeability

        self.testing_registry = {
            "mcnemar_binomial": mcnemar_binomial,
            "mcnemar_chisquare": mcnemar_chisquare,
            "t_test": t_test,
        }
        # To append all scores per dropped feature for all iterations
        # of while loop
        self.scores_and_preds_m1 = []
        self.scores_and_preds_m2 = []

        # SORTING ======================================================
        # Rationale:
        # In Classification metrics (except log-loss):
        # Higher Score <=> Better Predictive Performance
        # Worst feature will be that whose removal led to the highest
        # score in iteration

        # In Regression metrics:
        # Lower score <=> Better Predictive Performance
        # Worst feature will be that whose removal led to the lowest
        # score in iteration

        if task_type == "regression" or self.criterion == "log_loss":
            # Get the entry which has lowest score
            # NOTE redundant after deprecating compare_best_models()
            self.find_worst_feature = lambda scores: min(
                scores, key=lambda x: x[1]
            )
            # Order the container in ascending:
            # min score on top, max score on bottom
            self.sort_scores = lambda scores: sorted(
                scores, key=lambda x: x[1]
            )
        elif task_type == "classification" and self.criterion != "log_loss":
            # Get the entry which has highest score;
            # NOTE redundant after deprecating compare_best_models()
            self.find_worst_feature = lambda scores: max(
                scores, key=lambda x: x[1]
            )
            # Order the container in descending from max score to min
            self.sort_scores = lambda scores: sorted(
                scores, key=lambda x: x[1], reverse=True
            )
        else:
            raise ValueError(
                "Invalid task type specified. "
                "Choose 'regression' or 'classification'."
            )
        # SORTING END ==================================================

        # FEATURE LIST HANDLING ========================================
        # Dummy list to group:
        # will default to None if not provided
        self.dummy_list = dummy_list
        # Features to exclude from deselection:
        # defaults to None if not provided
        self.features_to_fix = features_to_fix
        # Group columns obtained from a dummified variable together
        if self.dummy_list:
            self.initial_feature_list = dummy_grouper(
                feature_list=list(self.X.columns),
                dummy_list=self.dummy_list,
            )
        else:
            self.initial_feature_list = list(self.X.columns)
        # Remove features we want to exclude from deselection
        # from the feature list for deselection list
        if self.features_to_fix:
            self.initial_feature_list = feature_fixer(
                self.initial_feature_list,
                self.features_to_fix,
            )
        # FEATURE LIST HANDLING END ====================================

    # DESELECTION METHOD ===============================================
    def _deselect_feature(self, feature_list, model, keras_params=None):
        # Empty list for scores
        score_per_dropped_feature = []
        counter = 0
        # Iterate over all features
        for feature in feature_list:
            counter += 1
            # Generate temporary feature set to manipulate
            temporary_set = feature_list.copy()
            # Drop a feature from set
            temporary_set.remove(feature)
            # Flatten list
            temporary_set = feature_list_flatten(temporary_set)
            X_temporary = self.X[temporary_set]

            # Train
            temp_model = ModelTrainer.fit(
                model=model, X=X_temporary, y=self.y, keras_params=keras_params
            )
            # Predict on validation set
            if self.validation_data:
                y_preds = ModelTrainer.predict(
                    model=temp_model,
                    X=self.X_val[temporary_set],
                    keras_params=keras_params,
                )
                # Evaluate
                score = self.criterion_registry[self.criterion](
                    self.y_val, y_preds
                )
            # Predict on training set if no validation set given
            else:
                y_preds = ModelTrainer.predict(
                    model=temp_model,
                    X=self.X[temporary_set],
                    keras_params=keras_params,
                )
                score = self.criterion_registry[self.criterion](
                    self.y, y_preds
                )
            score_per_dropped_feature.append((feature, score, y_preds))

        # Returns a list of tuples with three entries per i:
        # str(feature_name), float(score), np.array(preds)
        return score_per_dropped_feature

    # DESELECTION METHOD END ===========================================

    # MAIN ALGORIHM ====================================================
    # Compare different models
    def compare_models(self, m1, m2, keras_params=None):
        """
        Fits and evaluates two different models with various subsets of
        features. Measures inter-rater agreeability between models'
        predictions on the validation/test set.

        Args:
            m1: Sklearn or Keras model.
            m2: Sklearn or Keras model.
            keras_params (optional): KerasParams object carrying
                                     pre-defined configuration for
                                     training a Keras model -
                                     batch_size, epochs,
                                     validation_split, callbacks,
                                     verbose arguments do be called in
                                     training.

        Note on including Keras models:
            Keras models must be provided as a KerasSequential class
            object given in this library to ensure proper compiling,
            fitting and inference during the iteration's of the
            algorithm.

        Example use:
            # Define KerasSequential model
            mlp = KerasSequential()  # Initialize as KerasSequential
            mlp.add(  # 128 units, linear activation
                tf.keras.layers.Dense, units=128, activation='linear'
            )
            mlp.add(  # 64 units, linear activation
                tf.keras.layers.Dense, units=64, activation='linear'
            )
            mlp.add(  # Sigmoid output layer
                tf.keras.layers.Dense, units=1, activation='sigmoid'
            )
            mlp.compile(
                optimizer='adam',  # default lr: 0.001 for adam
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'],  # Track accuracy
            )

            # Define keras_params for training

            EARLY_STOPPING = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]  # Early stopping callback

            keras_params = KerasParams(
                batch_size=32,  # batch size for mini-batch training
                epochs=100,  # fit a model for 100 epochs
                validation_split=0.2,  # further 0.8-0.2 split
                callbacks=EARLY_STOPPING,  # include early stopping
                verbose=0  # no per-epoch logs
            )

            # Define Random Forest Classifier
            rfc = RandomForestClassifier(n_estimators=100)

            # Run the algorithm via an initialized BackEliminator class
            # (refer to BackEliminator documentation)
            results = seeker.compare_models(
                m1=rfc,  # Model 1: Random Forest Classifier
                m2=mlp,  # Model 2: MultiLayer Perceptron
                keras_params=keras_params  # Training Parameters for MLP
            )
        """
        # PATHS FOR RESULTS ============================================
        # Main folder for test results
        main_folder = "test_results"
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)
        # Subfolder for specific experiment
        sub_folder = "experiment_results"
        datestamp = datetime.now().strftime("%Y%m%d")
        count = 0
        while True:
            count += 1
            new_experiment_folder = f"{sub_folder}_{datestamp}_{count}"
            full_path = os.path.join(main_folder, new_experiment_folder)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                break
        self.new_experiment_folder = new_experiment_folder
        self.full_path = full_path
        # PATHS FOR RESULTS END ========================================

        # To fit with entire feature subset, copy the feature lists
        new_feature_list_m1 = self.initial_feature_list.copy()
        new_feature_list_m2 = self.initial_feature_list.copy()
        # Container for results
        results = []
        # Flatten lists
        full_fit_m1 = feature_list_flatten(new_feature_list_m1)
        full_fit_m2 = feature_list_flatten(new_feature_list_m2)
        # Fit the models with entire set of features
        # Model 1
        temp_m1 = ModelTrainer.fit(
            model=m1,
            X=self.X[full_fit_m1],
            y=self.y,
            keras_params=keras_params,
        )
        # Model 2
        temp_m2 = ModelTrainer.fit(
            model=m2,
            X=self.X[full_fit_m1],
            y=self.y,
            keras_params=keras_params,
        )
        # Predict
        if self.validation_data:
            # Model 1
            m1_preds = ModelTrainer.predict(
                model=temp_m1,
                X=self.X_val[full_fit_m1],
                keras_params=keras_params,
            )
            # Best prediction for M1
            best_score_m1 = self.criterion_registry[self.criterion](
                self.y_val, m1_preds
            )
            # Model 2
            m2_preds = ModelTrainer.predict(
                model=temp_m2,
                X=self.X_val[full_fit_m1],
                keras_params=keras_params,
            )
            # Best prediction for M2
            best_score_m2 = self.criterion_registry[self.criterion](
                self.y_val, m2_preds
            )
            # Aggreeability score between M1 and M2
            agreeability_coeff = self.agreeability_registry[self.agreeability](
                m1_preds, m2_preds
            )
        # Predict on training set if no validation set
        else:
            # Model 1
            m1_preds = ModelTrainer.predict(
                model=temp_m1, X=self.X[full_fit_m1], keras_params=keras_params
            )
            best_score_m1 = self.criterion_registry[self.criterion](
                self.y, m1_preds
            )
            # Model 2
            m2_preds = ModelTrainer.predict(
                model=temp_m2, X=self.X[full_fit_m1], keras_params=keras_params
            )
            best_score_m2 = self.criterion_registry[self.criterion](
                self.y, m2_preds
            )
            # Agreeability score
            agreeability_coeff = self.agreeability_registry[self.agreeability](
                m1_preds, m2_preds
            )
        # Append to results
        results.append(
            {
                "Best: M1 Included Features": full_fit_m1.copy(),
                f"Best: M1 {self.criterion}": best_score_m1,
                "Best: M2 Included Features": full_fit_m2.copy(),
                f"Best: M2 {self.criterion}": best_score_m2,
                f"Best: Agreeability ({self.agreeability})":
                    agreeability_coeff,
                f"All: M1 Mean {self.criterion}": best_score_m1,
                f"All: M1 STD {self.criterion}": 0,
                f"All: M2 Mean {self.criterion}": best_score_m2,
                f"All: M2 STD {self.criterion}": 0,
                f"All: Mean Agreeability ({self.agreeability})":
                    agreeability_coeff,
                "All: Agreeability St. Dev.": 0,
            }
        )
        # First iteration summary
        print("Initial run: fitted both models with full feature set.")
        print("-" * 150)
        print(
            f"Model 1 included: {new_feature_list_m1}. "
            f"{self.criterion.upper()}: {best_score_m1:.4f}"
        )
        print(
            f"Model 2 included: {new_feature_list_m2}. "
            f"{self.criterion.upper()}: {best_score_m2:.4f}"
        )
        print("-" * 150)
        print(
            f"Agreeability Coefficient ({self.agreeability}): "
            f"{agreeability_coeff:.4f}"
        )
        print("=" * 150)

        counter = 0
        # Begin loop to deselect and evaluate
        while len(new_feature_list_m1) > 1 and len(new_feature_list_m2) > 1:
            counter += 1
            # DESELECTION ==============================================
            # Obtain the score lists (removed feature, score, preds)
            score_per_dropped_feature_m1 = self._deselect_feature(
                new_feature_list_m1, m1, keras_params
            )
            score_per_dropped_feature_m2 = self._deselect_feature(
                new_feature_list_m2, m2, keras_params
            )
            # Sort the lists
            score_per_dropped_feature_m1 = self.sort_scores(
                score_per_dropped_feature_m1
            )
            score_per_dropped_feature_m2 = self.sort_scores(
                score_per_dropped_feature_m2
            )
            # DESELECTION END ==========================================

            # SCORE SORTING ============================================
            # Obtain all scores for m1 and m2 and get results
            all_scores_m1 = [row[1] for row in score_per_dropped_feature_m1]
            all_scores_m2 = [row[1] for row in score_per_dropped_feature_m2]
            # Obtain all preds for m1 and m2
            all_preds_m1 = [row[2] for row in score_per_dropped_feature_m1]
            all_preds_m2 = [row[2] for row in score_per_dropped_feature_m2]
            # Append to respective containers
            # NOTE this is essential for .compare_n_best() method
            self.scores_and_preds_m1.append((all_scores_m1, all_preds_m1))
            self.scores_and_preds_m2.append((all_scores_m2, all_preds_m2))
            # Get best scores
            best_score_m1 = all_scores_m1[0]
            best_score_m2 = all_scores_m2[0]
            # Get average of all scores
            mean_score_m1 = np.mean(all_scores_m1)
            mean_score_m2 = np.mean(all_scores_m2)
            # Get stdevs of all scores
            std_score_m1 = np.sqrt(
                np.mean((all_scores_m1 - mean_score_m1) ** 2)
            )
            std_score_m2 = np.sqrt(
                np.mean((all_scores_m2 - mean_score_m2) ** 2)
            )
            # SCORE SORTING END ========================================

            # AGREEABILITY =============================================
            # Get all predictions from both models as a list of lists
            all_preds_m1 = [row[2] for row in score_per_dropped_feature_m1]
            all_preds_m2 = [row[2] for row in score_per_dropped_feature_m2]

            # Get agreeability measures row for row
            # Result ordered s.t. entry on top is from the two
            # models with best performance going down to worst
            all_agreeabilities = [
                self.agreeability_registry[self.agreeability](
                    all_preds_m1[i],
                    all_preds_m2[i],
                )
                for i in range(len(all_preds_m1))
            ]
            # Get the agreeability coefficient between the predictions
            # of best models
            agreeability_coeff = all_agreeabilities[0]
            # Get average of all agreeability coeffs
            mean_agreeability = np.mean(all_agreeabilities)
            std_agreeability = np.std(all_agreeabilities)
            # AGREEABILITY END =========================================

            # DROPPING FEATURES ========================================
            worst_feature_m1 = score_per_dropped_feature_m1[0][0]
            worst_feature_m2 = score_per_dropped_feature_m2[0][0]
            # Update included feature lists
            new_feature_list_m1.remove(worst_feature_m1)
            new_feature_list_m2.remove(worst_feature_m2)
            # Flat lists to append to results
            flat_feature_list_m1 = feature_list_flatten(new_feature_list_m1)
            flat_feature_list_m2 = feature_list_flatten(new_feature_list_m2)
            # DROPPING FEATURES END ====================================

            # Append to results
            results.append(
                {
                    "Best: M1 Included Features": flat_feature_list_m1.copy(),
                    f"Best: M1 {self.criterion}": best_score_m1,
                    "Best: M2 Included Features": flat_feature_list_m2.copy(),
                    f"Best: M2 {self.criterion}": best_score_m2,
                    f"Best: Agreeability ({self.agreeability})":
                        agreeability_coeff,
                    f"All: M1 Mean {self.criterion}": mean_score_m1,
                    f"All: M1 STD {self.criterion}": std_score_m1,
                    f"All: M2 Mean {self.criterion}": mean_score_m2,
                    f"All: M2 STD {self.criterion}": std_score_m2,
                    f"All: Mean Agreeability ({self.agreeability})":
                        mean_agreeability,
                    "All: Agreeability St. Dev.": std_agreeability,
                }
            )

            # Print iter results
            print(f"Iteration {counter}:")
            print("-" * 150)
            print("Results from best models:")
            print(
                f"Best Model 1 included: {new_feature_list_m1}. "
                f"{self.criterion.upper()}: {best_score_m1:.4f}"
            )
            print(
                f"Best Model 2 included: {new_feature_list_m2}. "
                f"{self.criterion.upper()}: {best_score_m2:.4f}"
            )
            print(
                f"Agreeability Coefficient ({self.agreeability}) "
                f"between best models: {agreeability_coeff}"
            )
            print("-" * 150)
            print("Results from all models:")
            print(
                f"M1 mean score: {mean_score_m1:.4f}. "
                f"Standard deviation: {std_score_m1:.4f}"
            )
            print(
                f"M2 mean score: {mean_score_m2:.4f}. "
                f"Standard deviation: {std_score_m2:.4f}"
            )
            print(
                f"Mean agreeability coefficient ({self.agreeability}): "
                f"{mean_agreeability:.4f}. "
                f"Standard deviation: {std_agreeability:.4f}"
            )
            print("=" * 150)
        self.results = results
        # Save results
        results_filename = f"{self.new_experiment_folder}.json"
        with open(os.path.join(self.full_path, results_filename), "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {self.full_path}")

        return results

    def compare_n_best(self, n=None, test=None):
        """
        Method for pair-wise comparison of n amount of best predictions
        obtained by the models. The pairwise tests are conducted within
        the predictions of each models and will test if predictions
        obtained are statistically significantly different from each
        other.

        Args:
            n (int): How many best results to compare.
            test (str): Statistical test to use. Options:
                        'mcnemar_binomial' and 'mcnemar_chisquare' for
                        binary classification. 't_test' for regression.

        Returns:
            None. pval_and_stats_m1 and pval_and_stats_m2 are callable
                  lists containing corresponding test statistics and
                p-values.

        Example: Setting n=3 will test:
                - M1: best predictions against second best predictions;
                      second best predictions and third best predictions.
                - M2: best predictions against second best predictions;
                      second best predictions and third best
                      predictions.
        """
        # Make sure the search is alrady ran and results are there
        if not self.scores_and_preds_m1 and not self.scores_and_preds_m2:
            raise ValueError(
                "No predictions found. Run a comparison algorithm first."
            )
        # Make sure n != value more than available best predictions
        if n > len(self.scores_and_preds_m1):
            raise ValueError(
                "Picked n is more than available amount of best "
                "predictions. Use n <= {}.".format(
                    len(self.scores_and_preds_m1)
                )
            )

        # Make sure test supported
        if test not in self.testing_registry:
            raise ValueError(
                "Test not supported. Please use 'mcnemar_binomial' or "
                "'mcnemar_chisquare' for classification or 't_test' "
                "for regression."
            )

        # Empty containers
        self.pval_and_stats_m1 = []
        self.pval_and_stats_m2 = []
        # Iterate n-1 times
        for i in range(n - 1):
            # Get result for model 1
            pval_m1, stat_m1 = self.testing_registry[test](
                self.scores_and_preds_m1[i][1][0],
                self.scores_and_preds_m1[i + 1][1][0],
                self.y_val,
            )
            self.pval_and_stats_m1.append((pval_m1, stat_m1))
            # Get result for model 2
            pval_m2, stat_m2 = self.testing_registry[test](
                self.scores_and_preds_m2[i][1][0],
                self.scores_and_preds_m2[i + 1][1][0],
                self.y_val,
            )
            self.pval_and_stats_m2.append((pval_m2, stat_m2))
            print(
                f"Model 1: Results for No.{i+1} and No.{i+2} best preds: "
                f"P-value: {pval_m1:.8f}. Test statistic: {stat_m1:.8f}."
            )
            print(
                f"Model 2: Results for No.{i+1} and No.{i+2} best preds: "
                f"P-value: {pval_m2:.8f}. Test statistic: {stat_m2:.8f}."
            )
            print("=" * 120)

    # Method to turn results into a df
    def dataframe_from_results(self):
        """
        Return results as a dataframe.
        """
        # Check if results exist
        if not self.results:
            raise ValueError(
                "There are no results available. "
                "Make sure to run compare_models first."
            )
        # Return results
        return pd.DataFrame(self.results)

    def plot_best(self):
        """
        Makes a simplified line plot from the best results using Matplotlib.
        """
        if not self.results:
            raise ValueError(
                "There are no results available. "
                "Make sure to run the algorithm first."
            )
        df = pd.DataFrame(self.results)
        # Do two y axes
        fig, ax1 = plt.subplots()

        # Plot agreeability on lhs axis
        ax1.set_xlabel("Iteration", fontsize=18)
        ax1.set_ylabel("Agreeability", fontsize=18)
        ax1.plot(
            df.index + 1,
            df.iloc[:, 4],
            label=df.columns[4],
            linewidth=4,
            color="#9467bd",
        )
        ax1.tick_params(axis="y")
        # NOTE can be activated to fix lhs y axis between 0-1
        # ax1.set_ylim([0, 1])

        # Plot model scores on rhs axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Model Scores", fontsize=18)
        ax2.plot(
            df.index + 1,
            df.iloc[:, 1],
            label=df.columns[1],
            linewidth=4,
            color="#D8696F",
        )
        ax2.plot(
            df.index + 1,
            df.iloc[:, 3],
            label=df.columns[3],
            linewidth=4,
            color="#2CA02C",
        )
        ax2.tick_params(axis="y")
        # NOTE can be activated to fix lower bound of rhs y axis at 0
        # ax2.set_ylim([0, None])

        # Combined legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best", fontsize=16)

        # Title, layout, save
        fig.tight_layout()
        plt.title("Model Scores and Agreeability Over Iterations", fontsize=20)
        plot_name = f"{self.new_experiment_folder}_best_scores.png"
        plt.savefig(
            os.path.join(self.full_path, plot_name),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close()
        del df

    def plot_all(self):
        """
        Makes a simplified line plot from all (mean) results using Matplotlib.
        """
        if not self.results:
            raise ValueError(
                "There are no results available. "
                "Make sure to run the algorithm first."
            )
        df = pd.DataFrame(self.results)
        # Create figure and axis objects
        fig, ax1 = plt.subplots()

        # Plot agreeability on lhs axis
        ax1.set_xlabel("Iteration", fontsize=18)
        ax1.set_ylabel("Agreeability", fontsize=18)
        ax1.plot(
            df.index + 1,
            df.iloc[:, 9],
            label=df.columns[9],
            linewidth=4,
            color="#9467bd",
        )
        ax1.fill_between(
            df.index + 1,
            df.iloc[:, 9] - df.iloc[:, 10],
            df.iloc[:, 9] + df.iloc[:, 10],
            alpha=0.5,
            color="#9467bd",
        )
        ax1.tick_params(axis="y")
        # NOTE can be activated to fix lhs y axis between 0-1
        # ax1.set_ylim([0, 1])

        # Plot model scores on rhs axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Model Scores", fontsize=18)
        ax2.plot(
            df.index + 1,
            df.iloc[:, 5],
            label=df.columns[5],
            linewidth=4,
            color="#D8696F",
        )
        ax2.fill_between(
            df.index + 1,
            df.iloc[:, 5] - df.iloc[:, 6],
            df.iloc[:, 5] + df.iloc[:, 6],
            alpha=0.2,
            color="#D8696F",
        )
        ax2.plot(
            df.index + 1,
            df.iloc[:, 7],
            label=df.columns[7],
            linewidth=4,
            color="#2CA02C",
        )
        ax2.fill_between(
            df.index + 1,
            df.iloc[:, 7] - df.iloc[:, 8],
            df.iloc[:, 7] + df.iloc[:, 8],
            alpha=0.2,
            color="#2CA02C",
        )
        ax2.tick_params(axis="y")
        # NOTE can be activated to fix lower bound of rhs y axis at 0
        # ax2.set_ylim([0, None])

        # Combined legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best", fontsize=16)

        # Title, layout, save
        plt.tight_layout()
        plt.title(
            "Mean Model Scores and Agreeability Over Iterations", fontsize=20
        )
        plot_name = f"{self.new_experiment_folder}_mean_scores.png"
        plt.savefig(
            os.path.join(self.full_path, plot_name),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close()
        del df

    # Method to turn results into an interactive plot
    def interactive_plot(self):
        """
        Makes an interactive plot from the results.
        """
        if not self.results:
            raise ValueError(
                "There are no results available. "
                "Make sure to run compare_models first."
            )
        df = pd.DataFrame(self.results)

        df["Summary_Agreeability"] = df.apply(
            lambda row: (
                f"<br> {df.columns[4]}: <br> {row.iloc[4]:.4f} <br> "
                f"{df.columns[9]}: <br> {row.iloc[9]:.4f} <br> "
                f"{df.columns[10]}: <br> {row.iloc[10]:.4f}"
            ),
            axis=1,
        )
        df["Summary_M1"] = df.apply(
            lambda row: (
                f"<br> {df.columns[1]}: <br> {row.iloc[1]:.4f} <br> "
                # f"{df.columns[0]}: <br> {', '.join(row.iloc[0])} <br> "
                f"{df.columns[5]}: <br> {row.iloc[5]:.4f} <br> "
                f"{df.columns[6]}: <br> {row.iloc[6]:.4f}"
            ),
            axis=1,
        )
        df["Summary_M2"] = df.apply(
            lambda row: (
                f"<br> {df.columns[3]}: <br> {row.iloc[3]:.4f} <br> "
                # f"{df.columns[2]}: <br> {', '.join(row.iloc[2])} <br> "
                f"{df.columns[7]}: <br> {row.iloc[7]:.4f} <br> "
                f"{df.columns[8]}: <br> {row.iloc[8]:.4f}"
            ),
            axis=1,
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Plot agreeability
        fig.add_trace(
            go.Scatter(
                x=df.index + 1,
                y=df.iloc[:, 4],
                name=f"{df.columns[4]}",
                mode="lines+markers",
                hovertext=df["Summary_Agreeability"],
                hoverinfo="text",
            ),
            secondary_y=False,
        )

        # Plot model 1 score
        fig.add_trace(
            go.Scatter(
                x=df.index + 1,
                y=df.iloc[:, 1],
                name=f"{df.columns[1]}",
                mode="lines+markers",
                hovertext=df["Summary_M1"],
                hoverinfo="text",
            ),
            secondary_y=True,
        )

        # Plot model 2 score
        fig.add_trace(
            go.Scatter(
                x=df.index + 1,
                y=df.iloc[:, 3],
                name=f"{df.columns[3]}",
                mode="lines+markers",
                hovertext=df["Summary_M2"],
                hoverinfo="text",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=(
                "Agreeability Coefficients and Model Scores Over "
                "Algorithm Iterations"
            ),
            xaxis_title="Iteration",
            yaxis_title="Agreeability",
            yaxis2_title="Model Scores",
            hovermode="closest",
        )

        fig.update_xaxes(type="category")
        plot_name = f"{self.new_experiment_folder}_interactive.html"
        fig.write_html(os.path.join(self.full_path, plot_name))
        fig.show()
