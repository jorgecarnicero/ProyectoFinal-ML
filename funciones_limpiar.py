import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


def computar_nulos(df:pd.DataFrame):
    """
    Imputar los valores nulos de nuestro DataFrame original

    Args:
        df (pd.DataFrame): DataFrame original

    Returns:
        df_copy(pd.DataFrame): copia del DataFrame original con los valores nulos imputados
    """

    # Hacemos una copia del df
    df_copy = df.copy()
    
    # Iteramos sobre las columnas
    for col in df_copy.columns:
        # Comprobamos si tiene valores nulos, y en caso de haber entra
        if df_copy[col].isna().sum() > 0:
            # Vemos que tipo de columna es, si es numérica o categórica
            tipo = df_copy[col].dtype

            if tipo == "object":
                # Imputamos por la moda
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

            elif tipo in ["float64","int64"]:
                # Imputamos por la mediana
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
                
    return df_copy

def computar_otras_en_razon(df:pd.DataFrame):
    """
    Reemplaza el valor "otros" por "otras" en la columna 'razon'.


    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        df_copy(pd.DataFrame): Copia del DataFrame con los cambios de otros por otras.
    """

    df_copy = df.copy()

    df_copy["razon"] = df_copy["razon"].replace("otros", "otras")

    return df_copy

def computar_outliers_faltas(df:pd.DataFrame,mediana:float):
    """
    Convertimos los otuliers de las faltas que consideremos, en nuestro caso cuando las faltas superen el valor de 100
    y lo cambiaremos por la mediana de las faltas.

    Args:
        df (pd.DataFrame): DataFrame original
        mediana (float): mediana del DataFrame orignal, en la columna de faltas

    Returns:
        df_copy(pd.DataFrame): Copia del DataFrame con los datos de las faltas outliers imputados.
    """
    # Copia del DataFrame
    df_copy = df.copy()

    # Cambiar los outliers
    df_copy.loc[df_copy["faltas"] > 100, "faltas"] = mediana

    return df_copy

def convertir_columnas_categoricas(df:pd.DataFrame, columnas_categoricas:list):
    """

    Hacemos dummy encoding sobre las variables categóricas del df original.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        columnas_categoricas (list): Lista con los nombres de columnas categóricas que queremos codificar.

    Returns:
        df_copy (pd.DataFrame): Copia del DataFrame original con las columnas ya encodeadas

    """

    df_copy = df.copy()

    for col in columnas_categoricas:

        # Obtenemos los valores de cada columna en orden
        valores = sorted(df_copy[col].unique())

        # Seleccionamos todas menos la primera opción, para ahorrarnos una variable
        categorias_finales = valores[1:]

        # Creamos una nueva columna por cada categoría
        for categoria in categorias_finales:
            nombre_columna = f"{col}_{categoria}"
            df_copy[nombre_columna] = (df_copy[col] == categoria).astype(int) 

        # Eliminamos la columna
        df_copy = df_copy.drop(columns=[col])

    
    return df_copy

def limpieza_total(df:pd.DataFrame):
    """
    Combina todas las técnicas de limpieza, para devolvernos el dataframe con todos los "errores" / correcciones que hemos querido hacer.

    Args:
        df (pd.DataFrame): dataframe que queremos limpiar

    Returns:
        df_copy(pd.DataFrame): copia del DataFrame original, ya con las cosas corregidas
    """
    
    df_copy = df.copy()

    # Limpiamos nulos
    df_copy = computar_nulos(df_copy)

    # Corregimos la columna de razon
    df_copy = computar_otras_en_razon(df_copy)

    # Corregimos outliers en la columna de faltas
    mediana = df_copy["faltas"].median()
    df_copy = computar_outliers_faltas(df_copy,mediana)

    # Encodeamos las columnas categóricas
    columas_categoricas =  df.select_dtypes(include=["object"])
    df_copy = convertir_columnas_categoricas(df_copy,columas_categoricas)

    return df_copy