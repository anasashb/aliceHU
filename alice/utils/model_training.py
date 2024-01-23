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

## Set up a keras model class
class KerasSequential:
    '''
    Docstring.
    '''
    def __init__(self):
        self.layer_info = []
    
    def add(self, layer, **kwargs):
        self.layer_info.append((layer, kwargs))
    
    def compile(self, **kwargs):
        self.compile_info = kwargs
    
    def build(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        for layer, kwargs in self.layer_info:
            model.add(layer(**kwargs))
        model.compile(**self.compile_info)
            # take this out
        return model
        

class ModelTrainer:
    '''
    Utility to handle key differences between .fit() and .predict() functions between Scikit-learn and Keras.
    '''
    @staticmethod
    def is_keras_seq(model):
        '''
        Checks if model instance is of type tf.keras.Model
        '''
        return isinstance(model, KerasSequential)
    
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
        if ModelTrainer.is_keras_seq(model):
            INPUT_SHAPE = X.shape[1]
            model = model.build(input_shape=X.shape[1])
            #model.summary()
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

            # Cast data to tensors
            X_tensor = tf.cast(X.values, dtype=tf.float32)
            y_tensor = tf.cast(y.values, dtype=tf.float32)

            # Fit
            model.fit(X_tensor, y_tensor, **keras_params)

            return model
        # If sklearn model
        else:
            model.fit(X, y)
            return model

    # Same for 
    @staticmethod
    def predict(model, X, keras_params=None):
        '''
        Prediction function wrapper to handle both Sklearn and Keras models.
        '''
        # If keras model, use batch size
        if ModelTrainer.is_keras(model):
            #print('is keras well done.') ##################### delet this debug line
            batch_size = keras_params.as_dict().get('batch_size') if keras_params else 32
            
            # Cast data to tensor
            X_tensor = tf.cast(X.values, dtype=tf.float32)

            preds = model.predict(X_tensor, batch_size=batch_size, verbose=0)

            return (preds >= 0.5).astype(int).flatten()
            #return preds
        # If sklearn just predict
        else:
            return model.predict(X)