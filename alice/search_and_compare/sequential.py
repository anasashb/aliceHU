from alice.metrics.regress import mse, rmse, mae
from alice.metrics.classify import accuracy, precision, recall, f1
from alice.agreeability.regress import pearson
from alice.agreeability.classify import cohen_kappa
from alice.utils.feature_lists import dummy_grouper
from alice.utils.feature_lists import feature_fixer
from alice.utils.feature_lists import feature_list_flatten
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

class BackEliminator():

    def __init__(self,
                 X=None,
                 y=None, 
                 validation_data=None,
                 task_type=None,
                 criterion=None,
                 agreeability=None,
                 dummy_list=None,
                 features_to_fix=None
                 ):

        self.X = X
        self.y = y
        if validation_data:
            self.validation_data = validation_data
            self.X_val = self.validation_data[0]
            self.y_val = self.validation_data[1]
        self.criterion_registry = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
            }
        self.criterion = criterion
        self.agreeability_registry = {
            'pearson': pearson,
            'cohen_kappa': cohen_kappa
        }
        self.agreeability = agreeability
        # To append all scores per dropped feature for all iterations of while loop
        self.scores_n_preds_m1 = []
        self.scores_n_preds_m2 = []
    
        #### =========================================================================================== ####
        #### NEW SORTING DEFINED                                                                            #
        #### ---------------------------------------------------------------------------------------------- #
        #### Rationale:                                                                                     #
        #### In Classification metrics: Higher Score <=> Better Predictive Performance                      #
        #### Worst feature will be that whose removal led to the highest score in iteration                 #
        #### In Regression metrics: Lower score <=> Better Predictive Performance                           #
        #### Worst feature will be that whose removal led to the lowest score in iteration                  #
        if task_type == 'classification':                                                                   #
            # Get the entry which has highest score (in second column - [1]) - used in compare_best_models  #
            self.find_worst_feature = lambda scores: max(scores, key=lambda x: x[1])                        #
            # Order the container in descending from max score to min score - used in compare_all_models    #
            self.sort_scores = lambda scores: sorted(scores, key=lambda x: x[1], reverse=True)              #
        elif task_type == 'regression':                                                                     #
            # Get the entry which has lowest score (in second column - [1])                                 #
            self.find_worst_feature = lambda scores: min(scores, key=lambda x: x[1])                        #
            # Order the container in ascending - min score on top max score on bottom                       #
            self.sort_scores = lambda scores: sorted(scores, key=lambda x: x[1])                            #
        else:                                                                                               #
            raise ValueError("Invalid task type specified. Choose 'regression' or 'classification'.")       #
        #### Return will be (worst_feature, best_score, best_preds) from iteration                          #
        #### =========================================================================================== ####

        # Handle feature lists
        # Will default to None if not provided
        self.dummy_list = dummy_list
        # Will default to None if not provided
        self.features_to_fix = features_to_fix
        # Group columns obtained from a one-hot-encoded variable together
        if self.dummy_list:
            self.initial_feature_list = dummy_grouper(feature_list=list(self.X.columns), dummy_list=self.dummy_list)
        else:
            self.initial_feature_list = list(self.X.columns)
        # Remove features we want to fix from the feature list
        if self.features_to_fix:
            self.initial_feature_list = feature_fixer(self.initial_feature_list, self.features_to_fix)
        
    
    # Method to be called in the main method of back elimination
    def _deselect_feature(self,
                          feature_list,
                          model):
        # Empty list for scores
        score_per_dropped_feature = []
        # Iterate over all features
        for feature in feature_list:
            # Generate temporary feature set to manipulate
            temporary_set = feature_list.copy()
            # Drop feature from set
            temporary_set.remove(feature)
            # Flatten list
            temporary_set = feature_list_flatten(temporary_set)
            # Train
            model.fit(self.X[temporary_set], self.y)
            # Predict on validation set
            if self.validation_data:
                y_preds = model.predict(self.X_val[temporary_set])
                # Evaluate
                score = self.criterion_registry[self.criterion](self.y_val, y_preds)
            # Predict on training set
            else:
                y_preds = model.predict(self.X[temporary_set])
                score = self.criterion_registry[self.criterion](self.y, y_preds)
            # Append feature name, score after dropping it, y_preds after dropping it
            score_per_dropped_feature.append((feature, score, y_preds))

        #### Deprecated ####
        # At the end of loop, identify feature
        # which led to the worst score when 
        # feature dropped
        # Descending sort based on score, (x[1])
        #score_per_dropped_feature = self.sort_scores(score_per_dropped_feature) #### REMOVE THIS

        # For ease of read
        #worst_feature = score_per_dropped_feature[0][0] #### REMOVE THIS
        #best_score = score_per_dropped_feature[0][1] ##### REMOVE THIS
        #best_preds = score_per_dropped_feature[0][2] ##### REMOVE THIS

        #del score_per_dropped_feature #### RETURN THIS
        # Return feature name
        #return worst_feature, best_score, best_preds
        
        #### =========================================================================================== ####
        #### NEW RETURN DEFINED                                                                             #
        #### ---------------------------------------------------------------------------------------------- #
        return score_per_dropped_feature                                                                    #
        #### Returns a list of tuples with three entries: str(feature_name), float(score), np.array(preds)  #
        #### =========================================================================================== ####
        ### TO DO ###
        # Add functionality to possibly save trained models 
        # Will take up large memory, may be unfeasible
        ### TO DO ###
    
    def compare_best_models(
            self,
            m1,
            m2
        ): 
        # Copy all features initially
        # for both models
        new_feature_list_m1 = self.initial_feature_list.copy()
        new_feature_list_m2 = self.initial_feature_list.copy()
        # Aggreeability scores
        results = []
        # First fit models w/o any removed features
        # Flat lists for fitting
        full_fit_m1 = feature_list_flatten(new_feature_list_m1)
        full_fit_m2 = feature_list_flatten(new_feature_list_m2)
        m1.fit(self.X[full_fit_m1], self.y)
        m2.fit(self.X[full_fit_m2], self.y)
        # Predict on validation set
        if self.validation_data:
            # Model 1
            m1_preds = m1.predict(self.X_val[full_fit_m1])
            m1_score = self.criterion_registry[self.criterion](self.y_val, m1_preds)
            # Model 2
            m2_preds = m2.predict(self.X_val[full_fit_m2])
            m2_score = self.criterion_registry[self.criterion](self.y_val, m2_preds)
            # Aggreeability Score
            agreeability_coeff = self.agreeability_registry[self.agreeability](m1_preds, m2_preds)
        # Predict on training set
        else:
            # Model 1
            m1_preds = m1.predict(self.X[full_fit_m1])
            m1_score = self.criterion_registry[self.criterion](self.y, m1_preds)
            # Model 2
            m2_preds = m2.predict(self.X[full_fit_m2])
            m2_score = self.criterion_registry[self.criterion](self.y, m2_preds)
            # Agreeability score
            agreeability_coeff = self.agreeability_registry[self.agreeability](m1_preds, m2_preds)
        
        # Append to results
        
        results.append({
            f'Best: M1 Included Features': full_fit_m1.copy(),
            f'Best: M1 {self.criterion.upper()}': m1_score,
            f'Best: M2 Included Features': full_fit_m2.copy(),
            f'Best: M2 {self.criterion.upper()}': m2_score,
            f'Best: Agreeability ({self.agreeability})': agreeability_coeff,
            })            

        ### DEBUG PRINTS
        print(f'Initial run: fitted both models with full feature set.')
        print(f'-' * 150)
        print(f'Model 1 included: {new_feature_list_m1}. {self.criterion.upper()}: {m1_score}')
        print(f'Model 2 included: {new_feature_list_m2}. {self.criterion.upper()}: {m2_score}')
        print(f'-' * 150)
        print(f'Agreeability Coefficient ({self.agreeability}): {agreeability_coeff}')
        print(f'=' * 150)
        ### DEBUG PRINTS   
        
        ### DEBUG
        counter = 0
        ### DEBUG

        # Begin loop to deselect and evaluate
        while len(new_feature_list_m1) > 1 and len(new_feature_list_m2) > 1:

            ### DEBUG
            counter += 1    
            ### DEBUG    

            # Obtain worst_feature, score and preds from deselect_feature functions
            #worst_feature_m1, m1_score, m1_preds = self._deselect_feature(new_feature_list_m1, m1)
            #worst_feature_m2, m2_score, m2_preds = self._deselect_feature(new_feature_list_m2, m2)
            # Update included feature lists
            #new_feature_list_m1.remove(worst_feature_m1) 
            #new_feature_list_m2.remove(worst_feature_m2)

            # Obtain the score lists (removed feature, corresponding score, corresponding preds)
            score_per_dropped_feature_m1 = self._deselect_feature(new_feature_list_m1, m1)
            score_per_dropped_feature_m2 = self._deselect_feature(new_feature_list_m2, m2)

            # Get the worst_feature, best_score, best_preds
            worst_feature_m1, m1_score, m1_preds = self.find_worst_feature(score_per_dropped_feature_m1)
            worst_feature_m2, m2_score, m2_preds = self.find_worst_feature(score_per_dropped_feature_m2)

            # Update included feature lists
            new_feature_list_m1.remove(worst_feature_m1)
            new_feature_list_m2.remove(worst_feature_m2)
            # Flat lists to append to results
            flat_feature_list_m1 = feature_list_flatten(new_feature_list_m1)
            flat_feature_list_m2 = feature_list_flatten(new_feature_list_m2)

            # Compute agreeability
            agreeability_coeff = self.agreeability_registry[self.agreeability](m1_preds, m2_preds)
            # Append to results
            results.append({
                'Model 1 Included Features': flat_feature_list_m1.copy(),
                f'Model 1 {self.criterion.upper()}': m1_score,
                'Model 2 Included Features': flat_feature_list_m2.copy(),
                f'Model 2 {self.criterion.upper()}': m2_score,
                f'Agreeability Coefficient ({self.agreeability})': agreeability_coeff
            })

            ### DEBUG PRINTS
            print(f'Iteration {counter}:')
            print(f'-' * 150)
            print(f'Model 1 included: {new_feature_list_m1}. {self.criterion.upper()}: {m1_score}')
            print(f'Model 2 included: {new_feature_list_m2}. {self.criterion.upper()}: {m2_score}')
            print(f'-' * 150)
            print(f'Agreeability Coefficient ({self.agreeability}): {agreeability_coeff}')
            print(f'=' * 150)
            ### DEBUG PRINTS
        # Save results
        self.results = results
        # Return results
        return results
    
### Order for best for best    
    def compare_all_models(
            self,
            m1,
            m2
        ):
        '''
        Note: feature elimination strategy same as compare_best_models().
        At higher computing costs, evaluates agreeability between sub-par models at each iteration and computers mean agreeability score and standard deviation.
        Results obtained from _deselect_feature are ordered from best to worst
        ''' 
        # Copy all features initially
        # for both models
        new_feature_list_m1 = self.initial_feature_list.copy()
        new_feature_list_m2 = self.initial_feature_list.copy()
        # Aggreeability scores
        results = []
        # Flat lists for fitting
        full_fit_m1 = feature_list_flatten(new_feature_list_m1)
        full_fit_m2 = feature_list_flatten(new_feature_list_m2)
        # First fit models w/o any removed features
        m1.fit(self.X[full_fit_m1], self.y)
        m2.fit(self.X[full_fit_m2], self.y)
        # Predict on validation set
        if self.validation_data:
            # Model 1
            m1_preds = m1.predict(self.X_val[full_fit_m1])
            best_score_m1 = self.criterion_registry[self.criterion](self.y_val, m1_preds)
            # Model 2
            m2_preds = m2.predict(self.X_val[full_fit_m2])
            best_score_m2 = self.criterion_registry[self.criterion](self.y_val, m2_preds)
            # Aggreeability Score
            agreeability_coeff = self.agreeability_registry[self.agreeability](m1_preds, m2_preds)
        # Predict on training set
        else:
            # Model 1
            m1_preds = m1.predict(self.X[full_fit_m1])
            best_score_m1 = self.criterion_registry[self.criterion](self.y, m1_preds)
            # Model 2
            m2_preds = m2.predict(self.X[full_fit_m2])
            best_score_m2 = self.criterion_registry[self.criterion](self.y, m2_preds)
            # Agreeability score
            agreeability_coeff = self.agreeability_registry[self.agreeability](m1_preds, m2_preds)
        
        # Append to results
        #### TO FIX
        #### Since the first run is on entire dataset, - mean agreeability == agreeability, stdev == 0
        #results.append({
            #f'Best: M1 Included Features': new_feature_list_m1.copy(),
            #f'Best: M1 {self.criterion.upper()}': best_score_m1,
            #f'Best: M2 Included Features': new_feature_list_m2.copy(),
            #f'Best: M2 {self.criterion.upper()}': best_score_m2,
            #f'Best: Agreeability ({self.agreeability})': agreeability_coeff,
            #f'All: Mean Agreeability ({self.agreeability})': np.mean(agreeability_coeff),
            #f'All: Agreeability St. Dev.': np.std(agreeability_coeff)
        #})          

        results.append({
            f'Best: M1 Included Features': full_fit_m1.copy(),
            f'Best: M1 {self.criterion}': best_score_m1,
            f'Best: M2 Included Features': full_fit_m2.copy(),
            f'Best: M2 {self.criterion}': best_score_m2,
            f'Best: Agreeability ({self.agreeability})': agreeability_coeff,
            f'All: M1 Mean {self.criterion}': best_score_m1,
            f'All: M1 STD {self.criterion}': 0,
            f'All: M2 Mean {self.criterion}': best_score_m2,
            f'All: M2 STD {self.criterion}': 0,
            f'All: Mean Agreeability ({self.agreeability})': agreeability_coeff,
            f'All: Agreeability St. Dev.': 0
            })      

        ### DEBUG PRINTS
        print(f'Initial run: fitted both models with full feature set.')
        print(f'-' * 150)
        print(f'Model 1 included: {new_feature_list_m1}. {self.criterion.upper()}: {best_score_m1:.4f}')
        print(f'Model 2 included: {new_feature_list_m2}. {self.criterion.upper()}: {best_score_m2:.4f}')
        print(f'-' * 150)
        print(f'Agreeability Coefficient ({self.agreeability}): {agreeability_coeff:.4f}')
        print(f'=' * 150)
        ### DEBUG PRINTS   
        
        ### DEBUG
        counter = 0
        ### DEBUG

        # Begin loop to deselect and evaluate
        while len(new_feature_list_m1) > 1 and len(new_feature_list_m2) > 1:

            ### DEBUG
            counter += 1    
            ### DEBUG    

            # Obtain worst_feature, score and preds from deselect_feature functions
            #worst_feature_m1, m1_score, m1_preds = self._deselect_feature(new_feature_list_m1, m1)
            #worst_feature_m2, m2_score, m2_preds = self._deselect_feature(new_feature_list_m2, m2)
            # Update included feature lists
            #new_feature_list_m1.remove(worst_feature_m1) 
            #new_feature_list_m2.remove(worst_feature_m2)

            # Obtain the score lists (removed feature, score, preds)
            score_per_dropped_feature_m1 = self._deselect_feature(new_feature_list_m1, m1)
            score_per_dropped_feature_m2 = self._deselect_feature(new_feature_list_m2, m2)

            # Sort the list
            # Note that after sorting row results will not match iteration for iteration in _deselect_feature runs for m1 and m2
            score_per_dropped_feature_m1 = self.sort_scores(score_per_dropped_feature_m1)
            score_per_dropped_feature_m2 = self.sort_scores(score_per_dropped_feature_m2)

            ####################################################################################################################
            ############################################### HANDLE SCORES ######################################################
            ####################################################################################################################
            
            # Obtain all scores for m1 and m2
            all_scores_m1 = [row[1] for row in score_per_dropped_feature_m1]
            all_scores_m2 = [row[1] for row in score_per_dropped_feature_m2]
            # Obtain all preds for m1 and m2
            all_preds_m1 = [row[2] for row in score_per_dropped_feature_m1]
            all_preds_m2 = [row[2] for row in score_per_dropped_feature_m2]
            # Append to respective containers ####### TO BE USED IN A NEW METHOD FOR TESTING #########
            self.scores_n_preds_m1.append((all_scores_m1, all_preds_m1))
            self.scores_n_preds_m2.append((all_scores_m2, all_preds_m2))
            # Get best scores 
            best_score_m1 = all_scores_m1[0]
            best_score_m2 = all_scores_m2[0]
            # Average of all scores
            mean_score_m1 = np.mean(all_scores_m1)
            mean_score_m2 = np.mean(all_scores_m2)
            # Get std-s of all scores (a bit manually not to recompute means implicitly by using np.std())
            std_score_m1 = np.sqrt(np.mean((all_scores_m1 - mean_score_m1) ** 2))
            std_score_m2 = np.sqrt(np.mean((all_scores_m2 - mean_score_m2) ** 2))

            ####################################################################################################################
            ############################################ HANDLE AGREEABILITY ###################################################
            ####################################################################################################################

            # Get all predictions from both models as a list of lists
            # This will iterate row for row in the third column of the containers, where prediction arrays are given. 
            all_preds_m1 = [row[2] for row in score_per_dropped_feature_m1]
            all_preds_m2 = [row[2] for row in score_per_dropped_feature_m2]

            # Get agreeability measures row for row
            # Result will be ordered s.t. entry on top is from the two models with best performance going all the way down to worst

            all_agreeabilities = [self.agreeability_registry[self.agreeability](all_preds_m1[i], all_preds_m2[i]) for i in range(len(all_preds_m1))]
            # Grab the agreeability coefficient between the predictions of best models
            agreeability_coeff = all_agreeabilities[0]
            # Takes average of all agreeability coeffs
            mean_agreeability = np.mean(all_agreeabilities)
            std_agreeability = np.std(all_agreeabilities)

            ####################################################################################################################
            ############################################## HANDLE FEATURES #####################################################
            #################################################################################################################### 

            #### FOR BETTER READABILITY DEFINE ALL VARIABLES INDIVIDUALLY
            worst_feature_m1 = score_per_dropped_feature_m1[0][0]
            worst_feature_m2 = score_per_dropped_feature_m2[0][0]
            # Update included feature lists
            new_feature_list_m1.remove(worst_feature_m1)
            new_feature_list_m2.remove(worst_feature_m2)
            # Flat lists to append to results
            flat_feature_list_m1 = feature_list_flatten(new_feature_list_m1)
            flat_feature_list_m2 = feature_list_flatten(new_feature_list_m2)
            #### ADD A TOPRINT METHOD SOMEWHERE TO MAKE SURE WE ARE NOT calling .upper() uselessly -- for the time being removed uppers.
            # Append to results
            results.append({
                f'Best: M1 Included Features': flat_feature_list_m1.copy(),
                f'Best: M1 {self.criterion}': best_score_m1,
                f'Best: M2 Included Features': flat_feature_list_m2.copy(),
                f'Best: M2 {self.criterion}': best_score_m2,
                f'Best: Agreeability ({self.agreeability})': agreeability_coeff,
                f'All: M1 Mean {self.criterion}': mean_score_m1,
                f'All: M1 STD {self.criterion}': std_score_m1,
                f'All: M2 Mean {self.criterion}': mean_score_m2,
                f'All: M2 STD {self.criterion}': std_score_m2,
                f'All: Mean Agreeability ({self.agreeability})': mean_agreeability,
                f'All: Agreeability St. Dev.': std_agreeability
            })  

        
            ### DEBUG PRINTS
            print(f'Iteration {counter}:')
            print(f'-' * 150)
            print(f'Results from best models:')
            print(f'Best Model 1 included: {new_feature_list_m1}. {self.criterion.upper()}: {best_score_m1:.4f}')
            print(f'Best Model 2 included: {new_feature_list_m2}. {self.criterion.upper()}: {best_score_m2:.4f}')
            print(f'Agreeability Coefficient ({self.agreeability}) between best models: {agreeability_coeff}')
            print(f'-' * 150)
            print(f'Results from all models:')
            print(f'M1 mean score: {mean_score_m1:.4f}. Standard deviation: {std_score_m1:.4f}')
            print(f'M1 mean score: {mean_score_m2:.4f}. Standard deviation: {std_score_m2:.4f}')
            print(f'Mean agreeability coefficient ({self.agreeability}): {mean_agreeability:.4f}. Standard deviation: {std_agreeability:.4f}')
            print(f'=' * 150)
            ### DEBUG PRINTS
        # Save results
        self.results = results
        # Return results
        return results


        #### REMOVE DESELECT_INPROG REMOVE DESELECT_INPROG REMOVE DESELECT_INPROG

    # Method to turn results into a df
    def dataframe_from_results(self):
        '''
        Return results as a dataframe.
        '''
        # Check if results exist
        if not self.results:
            raise ValueError("There are no results available. Make sure to run compare_models first.")
        # Return results
        return pd.DataFrame(self.results)
    
    # Method to turn results into an interactive plot
    def plot_from_results(self):
        '''
        Makes an interactive plot from the results.
        '''
        if not self.results:
            raise ValueError("There are no results available. Make sure to run compare_models first.")
        df = pd.DataFrame(self.results)

        df['Summary_Agreeability'] = df.apply(lambda row: f"<br> {df.columns[4]}: <br> {row.iloc[4]:.4f} <br> {df.columns[9]}: <br> {row.iloc[9]:.4f} <br> {df.columns[10]}: <br> {row.iloc[10]:.4f}", axis=1)
        df['Summary_M1'] = df.apply(lambda row: f"<br> {df.columns[1]}: <br> {row.iloc[1]:.4f} <br> {df.columns[0]}: <br> {', '.join(row.iloc[0])} <br> {df.columns[5]}: <br> {row.iloc[5]:.4f} <br> {df.columns[6]}: <br> {row.iloc[6]:.4f}", axis=1)
        df['Summary_M2'] = df.apply(lambda row: f"<br> {df.columns[3]}: <br> {row.iloc[3]:.4f} <br> {df.columns[2]}: <br> {', '.join(row.iloc[2])} <br> {df.columns[7]}: <br> {row.iloc[7]:.4f} <br> {df.columns[8]}: <br> {row.iloc[8]:.4f}", axis=1)


        fig = make_subplots(
            specs=[[{'secondary_y': True}]]
        )

        # Plot agreeability
        fig.add_trace(
            go.Scatter(
            x=df.index + 1,
            y=df.iloc[:, 4],
            name=f'{df.columns[4]}',
            mode='lines+markers',
            hovertext=df['Summary_Agreeability'],
            hoverinfo='text' 
            ),
            secondary_y=False
        )

        # Plot model 1 score
        fig.add_trace(
            go.Scatter(
                x=df.index + 1,
                y=df.iloc[:, 1],
                name=f'{df.columns[1]}',
                mode='lines+markers',
                hovertext=df['Summary_M1'],
                hoverinfo='text'
            ),
            secondary_y=True
        )

        # Plot model 2 score
        fig.add_trace(
            go.Scatter(
                x=df.index+1,
                y=df.iloc[:, 3],
                name=f'{df.columns[3]}',
                mode='lines+markers',
                hovertext=df['Summary_M2'],
                hoverinfo='text'
            ),
            secondary_y=True
        )

        fig.update_layout(
            title='Agreeability Coefficients and Model Scores Over Algorithm Iterations',
            xaxis_title='Iteration',
            yaxis_title='Agreeability',
            yaxis2_title='Model Scores',
            hovermode='closest'
        )

        fig.update_xaxes(type='category')
        fig.show()