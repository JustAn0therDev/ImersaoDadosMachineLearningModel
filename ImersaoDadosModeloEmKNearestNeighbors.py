import pandas as pd
from sklearn.metrics import mean_squared_error 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

"""
    Cada instrução em Python pode ser executada em uma mesma célula (Jupyter Notebook)
    ou em células separadas onde façam sentido.
"""

source = 'https://github.com/alura-cursos/imersao-dados-2-2020/blob/master/MICRODADOS_ENEM_2019_SAMPLE_43278.csv?raw=true'

data = pd.read_csv(source)

data['NU_NOTA_TOTAL'] = data[['NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_CN', 'NU_NOTA_REDACAO', 'NU_NOTA_MT']].sum(axis=1)
grades_from_people_who_actually_did_the_test = data.query('NU_NOTA_TOTAL != 0')[['NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_CN', 'NU_NOTA_REDACAO', 'NU_NOTA_MT', 'NU_NOTA_TOTAL']]
exit_math_test_data = grades_from_people_who_actually_did_the_test.dropna()['NU_NOTA_MT']
entry_data = grades_from_people_who_actually_did_the_test.dropna()[['NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_CN', 'NU_NOTA_REDACAO']]

# Seed para evitar resultados com maior aleatoriedade.
SEED = 4321

x_train, x_test, y_train, y_test = train_test_split(entry_data, exit_math_test_data, test_size=0.25, random_state=SEED)

"""
    Imaginando um scatterplot, escolhemos um ponto no gráfico e olhamos
    para os 1000 "vizinhos" mais próximos daquele ponto para calcular a média entre eles
    e a probabilidade da nota com base nisso.
"""
neighbor_regressor = KNeighborsRegressor(n_neighbors=1000)
neighbor_regressor.fit(x_train, y_train)

neighbor_regressor_predictions = list(neighbor_regressor.predict(x_test))
y_test_list = list(y_test)

dataframe_diff = pd.DataFrame()
dataframe_diff['Actual'] = y_test_list
dataframe_diff['Prediction'] = neighbor_regressor_predictions
dataframe_diff['Difference'] = (dataframe_diff['Actual'] - dataframe_diff['Prediction'])
dataframe_diff['Squared_Difference'] = (dataframe_diff['Prediction'] - dataframe_diff['Prediction'])**2

"""
   Utilizando o algoritmo K Nearest Neighbors
   Com Regressão Linear 81~82 -> com K Nearest Neighbors: 73~74
"""

print('Margem de erro em pontos (para mais ou para menos): {}'.format(mean_squared_error(y_test_list, neighbor_regressor_predictions)**(0.5)))