from alice.metrics.regress import mse, rmse, mae
from alice.metrics.classify import accuracy, precision, recall, f1
from alice.agreeability.regress import pearson
from alice.agreeability.classify import cohen_kappa
import pandas as pd
import plotly.express as px


class BackEliminator():

    def __init__(self,
                 X=None,
                 y=None, 
                 validation_data=None,
                 task_type=None,
                 criterion=None,
                 agreeability=None
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
        self.initial_feature_list = list(self.X.columns)

        if task_type == 'classification':
            self.sort_scores = lambda scores: sorted(scores, key=lambda x: x[1], reverse=True)
        else:
            self.sort_scores = lambda scores: sorted(scores, key=lambda x: x[1])

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
            # Append
            score_per_dropped_feature.append((feature,score, y_preds))

        # At the end of loop, identify feature
        # which led to the worst score when 
        # feature dropped
        # Descending sort based on score, (x[1])
        score_per_dropped_feature = self.sort_scores(score_per_dropped_feature)

        # For ease of read
        worst_feature = score_per_dropped_feature[0][0]
        best_score = score_per_dropped_feature[0][1]
        best_preds = score_per_dropped_feature[0][2]

        # Free up memory
        del score_per_dropped_feature
        # Return feature name
        return worst_feature, best_score, best_preds
        ### TO DO ###
        # Add functionality to possibly save trained models 
        # Will take up large memory, may be unfeasible
        ### TO DO ###
    def compare_models(self,
                           m1,
                           m2,): 
        # Copy all features initially
        # for both models
        new_feature_list_m1 = self.initial_feature_list.copy()
        new_feature_list_m2 = self.initial_feature_list.copy()


        ### DEBUG
        print(f'Before loop m1:{new_feature_list_m1}')
        print(f'Before loop m2: {new_feature_list_m2}')
        ### DEBUG    


        # Aggreeability scores
        results = []
        # First fit models w/o any removed features
        m1.fit(self.X[new_feature_list_m1], self.y)
        m2.fit(self.X[new_feature_list_m2], self.y)
        # Predict on validation set
        if self.validation_data:
            # Model 1
            m1_preds = m1.predict(self.X_val[new_feature_list_m1])
            m1_score = self.criterion_registry[self.criterion](self.y_val, m1_preds)
            # Model 2
            m2_preds = m2.predict(self.X_val[new_feature_list_m2])
            m2_score = self.criterion_registry[self.criterion](self.y_val, m2_preds)
            # Aggreeability Score
            agreeability_coeff = self.agreeability_registry[self.agreeability](m1_preds, m2_preds)
        # Predict on training set
        else:
            # Model 1
            m1_preds = m1.predict(self.X[new_feature_list_m1])
            m1_score = self.criterion_registry[self.criterion](self.y, m1_preds)
            # Model 2
            m2_preds = m2.predict(self.X[new_feature_list_m2])
            m2_score = self.criterion_registry[self.criterion](self.y, m2_preds)
            # Agreeability score
            agreeability_coeff = self.agreeability_registry[self.agreeability](m1_preds, m2_preds)
        # Append to results
        results.append({
            'Model 1 Included Features': new_feature_list_m1.copy(),
            f'Model 1 {self.criterion.upper()}': m1_score,
            'Model 2 Included Features': new_feature_list_m2.copy(),
            f'Model 2 {self.criterion.upper()}': m2_score,
            f'Agreeability Coefficient ({self.agreeability})': agreeability_coeff
        })
        
        ### DEBUG
        counter = 0
        ### DEBUG

        # Begin loop to deselect and evaluate
        while len(new_feature_list_m1) > 1 and len(new_feature_list_m2) > 1:

            ### DEBUG
            counter += 1    
            ### DEBUG    

            # Obtain worst_feature, score and preds from deselect_feature functions
            worst_feature_m1, m1_score, m1_preds = self._deselect_feature(new_feature_list_m1, m1)
            worst_feature_m2, m2_score, m2_preds = self._deselect_feature(new_feature_list_m2, m2)
            # Update included feature lists
            new_feature_list_m1.remove(worst_feature_m1) 
            new_feature_list_m2.remove(worst_feature_m2)


            ### DEBUG
            print(f'At iteration {counter} m1:{new_feature_list_m1}')
            print(f'At iteration {counter} m2: {new_feature_list_m2}')
            ### DEBUG    


            # Compute agreeability
            agreeability_coeff = self.agreeability_registry[self.agreeability](m1_preds, m2_preds)
            # Append to results
            results.append({
                'Model 1 Included Features': new_feature_list_m1.copy(),
                f'Model 1 {self.criterion.upper()}': m1_score,
                'Model 2 Included Features': new_feature_list_m2.copy(),
                f'Model 2 {self.criterion.upper()}': m2_score,
                f'Agreeability Coefficient ({self.agreeability})': agreeability_coeff
            })
        # Save results
        self.results = results
        # Return results
        return results
    
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

        # Create a new column that combines the relevant information for hovering with HTML line breaks
        df['Summary'] = df.apply(lambda row: f"{df.columns[0]}: <br>{', '.join(row.iloc[0])}<br>{df.columns[1]}: {row.iloc[1]}<br>{df.columns[2]}: <br>{', '.join(row.iloc[2])}<br>{df.columns[3]}: {row.iloc[3]}", axis=1)

        # Plot 
        fig = px.line(
            df, 
            x=df.index+1, 
            y=df.iloc[:, 4], 
            hover_data=['Summary'], 
            labels={'y': f'{df.columns[4]}', 'x': 'Iteration'},
            title='Agreeability Coefficients Over Algorithm Iterations',
            markers=True
        )

        fig.update_xaxes(type='category')
        fig.update_layout(hovermode='closest')

        fig.show()