# Análisis Predictivo del Rendimiento Académico 
**Autor**: Jorge Carnicero Príncipe

Este proyecto tiene como objetivo realizar una **predicción de la nota final** de los estudiantes a partir de distintos conjuntos de variables académicas y sociales. Además del análisis predictivo, se lleva a cabo una interpretación de las variables más influyentes, proponiendo posibles líneas de intervención desde el punto de vista de un **director académico**.

También contiene una interpretación de cómo influyen **variables sociales**, reflexionando sobre cómo estas afectan a nuestro día a día y cómo podríamos adaptarnos a ellas para lograr un mejor rendimiento académico.

---

## Contenido

- `main.ipynb`: notebook principal con el desarrollo del proyecto.
- `LinearRegressor.py`: contiene el modelo de regresión lineal, implementado por nosotros.
- `funciones_limpiar.py`: contiene las funciones que usaremos para limpiar el DataFrame.
- `funciones_prediccion.py`: contiene las funciones que usaremos para analizar las predicciones.
- `requirements.txt`: dependencias necesarias para ejecutar el proyecto.

---

## Requisitos y ejecución

### 1. Clona el repositorio o Descarga el Contenido 

```bash
git clone https://github.com/jorgecarnicero/ProyectoFinal-ML.git
cd ProyectoFinal-ML
```

### 2. Crea un entorno virtual (recomendado, pero no necesario)

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instala dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecuta el notebook
La implementación es en un solo notebook, dividido en subsecciones para mayor velocidad y facilidad a la hora de querer analizar los datos. Ahorrandonos múltiples archivos.

#### Opción 1. Desde Jupyter
Abre y ejecuta ``main.ipynb`` manualmente ( ▶ )

#### Opción 2: Desde terminal (automático)
```bash
pip install jupyter
jupyter nbconvert --to notebook --execute main.ipynb --output main_ejecutado.ipynb
```
Y nos aparecerá un `main_ejecutado.ipynb`, con el notebook ejecutado