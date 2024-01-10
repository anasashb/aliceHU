import tensorflow as tf


class KerasParams:
    '''
    Object to hold additional training parameters required for Keras.
    '''
    def __init__(self, batch_size=32, epochs=10, validation_split=0.2, callbacks=None, verbose=1):
        '''
        Args:
            batch_size (int): Batch size.
            epochs (int): Training epochs.
            validation_split (float): Validation split ratio for the provided training set
            callbacks: Keras callback - defaults to early stopping
            verbose: Verbosity of keras logging.
        '''
        if callbacks is None:
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        
        self.params = {
            'batch_size': batch_size,
            'epochs': epochs, 
            'validation_split': validation_split,
            'callbacks': callbacks,
            'verbose': verbose
        }

    def as_dict(self):
        '''
        Returns:
            params (dict): Dictionary of parameters to provide to ModelTrainer.
        '''
        return self.params


class ModelTrainer:
    '''
    Utility to handle key differences between .fit() and .predict() functions between Scikit-learn and Keras.
    '''
    @staticmethod
    def is_keras(model):
        '''
        Checks if model instance is of type tf.keras.Model
        '''
        return isinstance(model, tf.keras.Model)
    
    @staticmethod
    def fit(model, X, y, keras_params=None):
        '''
        Training function wrapper to handle both Sklearn and Keras models.
        '''
        if ModelTrainer.is_keras(model):
            # Incorporate additional parameters to fitting - batch_size, epochs, validation_split, callbacks, verbose
            
             # If keras model provided without training config default to
            if keras_params:
                keras_params = keras_params.as_dict()
            else:
                keras_params = KerasParams(
                    batch_size=32,
                    epochs=100,
                    validation_split=0.2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)],
                    verbose=0
                ).as_dict()
                
            # Fit
            model.fit(X, y, **keras_params)
        # If sklearn model
        else:
            model.fit(X, y)

    # Same for 
    @staticmethod
    def predict(model, X, keras_params=None):
        '''
        Prediction function wrapper to handle both Sklearn and Keras models.
        '''
        # If keras model, use batch size
        if ModelTrainer.is_keras(model):
            batch_size = keras_params.as_dict().get('batch_size') if keras_params else 32
            preds = model.predict(X, batch_size=batch_size, verbose=0)

            return (preds >= 0.5).astype(int).flatten()
        # If sklearn just predict
        else:
            return model.predict(X)