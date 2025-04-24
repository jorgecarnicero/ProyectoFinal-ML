import numpy as np
import pandas as pd

def dividir_df_train_test(df:pd.DataFrame, test_size:float=0.3, random_state:int=0):
    """

    Divide un DataFrame en conjuntos de entrenamiento y test con mezcla aleatoria de las filas.

    Args:
        df (pd.DataFrame): El DataFrame original con los datos.
        test_size (float, optional): Porcentaje del tamaño que queremos de test
        random_state (int, optional): Semilla para poder repetir de forma aleatoria y que no cambien los datos

    Returns:
        X_train, y_train: variables para el entrenamiento del modelo, X features e Y variable objetivo
        X_test, y_test: variables para el test del modelo.

    """

    # Cambiar algún nombre más

    if test_size >= 0.9:
        test_size = 0.9

    # Mezclamos el DataFrame
    df_mezclado = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Vemos con el % indicado que tamaño tendrá el conjunto de test.
    test_limite = int(len(df_mezclado) * test_size)

    # Dividimos el dataframe en test y train
    df_test = df_mezclado.iloc[:test_limite]
    df_train = df_mezclado.iloc[test_limite:]
    
    # Dividimos entre la variable objetivo y el conjunto de features, para cada conjunto.
    X_train, y_train = df_train,df_train["T3"]
    X_test, y_test = df_test,df_test["T3"]

    # Quitamos la variable objetivo de la X
    X_train = X_train.drop(columns=["T3"])
    X_test = X_test.drop(columns=["T3"])



    # Devolvemos todos los subconjuntos obtenidos
    return X_train, y_train, X_test, y_test

def evaluate_regression(y_true:np.ndarray, y_pred:np.ndarray):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    RSS = np.sum(np.power(y_true-y_pred,2))
    TSS = np.sum(np.power(y_true-np.mean(y_true),2))
    # TODO: Calculate R^2
    # r_squared = np.power(np.cov(y_true,y_pred,ddof=0),2) / (np.var(y_true,ddof=0) * np.var(y_pred))
    r_squared = 1 - (RSS/TSS)
    # Root Mean Squared Error
    # TODO: Calculate RMSE
    rmse = np.sqrt((1/len(y_true)) * np.sum(np.power(y_true-y_pred,2)))

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = (1/len(y_true)) * np.sum(np.abs(y_true-y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}
