<h1> <center> Neural Assignment </center> </h1>
Goal: Data classification <br/>
Using: Self Organizing Map and SOM's vectors in a Radial Basis Function Neural Network <br/> 
Author: BOUKHELOUA Rémi Tej


```python
#Code I used to transform the data from matlab format to csv one, so I don't need to do it every time

# import scipy.io
# import numpy as np

# dataX = scipy.io.loadmat(r"data_train.mat")
# np.savetxt("data_train.csv", np.asarray(dataX["data_train"]), delimiter=",")
# dataY = scipy.io.loadmat(r"label_train.mat")
# np.savetxt("label_train.csv", np.asarray(dataY["label_train"]), delimiter=",")
# dataToPredictX = scipy.io.loadmat(r"data_test.mat")
# np.savetxt("data_test.csv", np.asarray(dataToPredictX["data_test"]), delimiter=",")
```

### Imports


```python
#Make sure you have all those packages available on your device
#If not you probably can do a conda install (Name of the package)
#For further details, please check: https://anaconda.org/conda-forge/repo

import scipy.io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random
import csv
from collections import Counter
```

## Exploration


```python
dataTrainFile = "data_train" #Train data to load
labelTrainFile = "label_train" #Label of train data to load

dataTestFile = "data_test" #Train data to load
```


```python
# Loading data in lists
dataX = [list(i) for i in scipy.io.loadmat(dataTrainFile + ".mat")[dataTrainFile]]
dataY = [i[0] for i in list(scipy.io.loadmat(labelTrainFile + ".mat")[labelTrainFile])]
```


```python
#Charging datas in pandas dataframe for analysis only. Don't use pandas outside, it would slow down computations
df = pd.DataFrame(dataX, dataY).reset_index().rename(columns={"index": "label"})
```


```python
print("nb of line:", len(df))
print("nb of columns:", len(df.columns))
```

    nb of line: 330
    nb of columns: 34
    


```python
#Shape
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>-2.865899</td>
      <td>0.720619</td>
      <td>2.164823</td>
      <td>-1.156208</td>
      <td>-0.251490</td>
      <td>-1.116596</td>
      <td>-0.229209</td>
      <td>-2.981564</td>
      <td>-2.441548</td>
      <td>...</td>
      <td>-0.684820</td>
      <td>0.139995</td>
      <td>0.887941</td>
      <td>-1.691672</td>
      <td>-2.393610</td>
      <td>2.023542</td>
      <td>-2.366672</td>
      <td>1.954525</td>
      <td>-2.581707</td>
      <td>2.104295</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.347937</td>
      <td>0.641677</td>
      <td>0.060050</td>
      <td>0.767382</td>
      <td>-0.158263</td>
      <td>0.913227</td>
      <td>-0.050370</td>
      <td>0.768820</td>
      <td>-0.481110</td>
      <td>...</td>
      <td>0.277699</td>
      <td>-0.011530</td>
      <td>0.075298</td>
      <td>0.086337</td>
      <td>0.205240</td>
      <td>0.030311</td>
      <td>0.280999</td>
      <td>0.007095</td>
      <td>0.328369</td>
      <td>-0.034804</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1</td>
      <td>-2.865899</td>
      <td>-3.297799</td>
      <td>2.164823</td>
      <td>-1.156208</td>
      <td>-0.251490</td>
      <td>-1.116596</td>
      <td>-0.229209</td>
      <td>-2.981564</td>
      <td>1.691956</td>
      <td>...</td>
      <td>-2.413575</td>
      <td>-1.826594</td>
      <td>0.887941</td>
      <td>-1.691672</td>
      <td>1.079303</td>
      <td>2.023542</td>
      <td>-2.366672</td>
      <td>1.954525</td>
      <td>-0.668430</td>
      <td>-0.030918</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 34 columns</p>
</div>




```python
#Proportion of -1 and 1
print("Repartition of labels")
df.groupby("label")[0].count()
```

    Repartition of labels
    




    label
    -1    116
     1    214
    Name: 0, dtype: int64




```python
#Box plot to see value ranges and repartition
df.boxplot(figsize = (10,10), grid = False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x9caf2b0>




![png](output_11_1.png)



```python
df.groupby("label").mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>-0.621755</td>
      <td>-0.736889</td>
      <td>-0.094129</td>
      <td>-0.712457</td>
      <td>-0.175886</td>
      <td>-0.633875</td>
      <td>-0.207343</td>
      <td>-0.412804</td>
      <td>-0.160251</td>
      <td>-0.193880</td>
      <td>...</td>
      <td>-0.236397</td>
      <td>-0.081370</td>
      <td>0.153250</td>
      <td>-0.102369</td>
      <td>-0.346642</td>
      <td>0.034486</td>
      <td>-0.437771</td>
      <td>0.01304</td>
      <td>-0.377915</td>
      <td>0.090371</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.347937</td>
      <td>0.374576</td>
      <td>0.106543</td>
      <td>0.371677</td>
      <td>0.135849</td>
      <td>0.314223</td>
      <td>0.188947</td>
      <td>0.195352</td>
      <td>0.123971</td>
      <td>0.097373</td>
      <td>...</td>
      <td>0.128322</td>
      <td>0.032896</td>
      <td>-0.089154</td>
      <td>0.059874</td>
      <td>0.189915</td>
      <td>0.027319</td>
      <td>0.219213</td>
      <td>0.00885</td>
      <td>0.192323</td>
      <td>-0.007589</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 33 columns</p>
</div>




```python
#Getting boxplot to compare the two groups on a column
df.boxplot(by = "label", column=[1,5,7], layout = (1,3))
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x000000001215D9B0>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x000000001217CB38>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x00000000121ABDD8>],
          dtype=object)




![png](output_13_1.png)



```python
#Correlation between columns
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(df.corr())
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb1765f8>




![png](output_14_1.png)


## Self Organizing Map Neural Network (FS)


```python
#Regular functions to calculate distances, find closest node ...

#Euclidian distance
def eucDist(v1, v2):
    return np.linalg.norm(v1 - v2) 

#Manhattan distance between 2D vectors (x1,y1) and (x2,y2)
def manhattanDist(x1, y1, x2, y2):
    return np.abs(x1-x2) + np.abs(y1-y2)

#Given a data (one row), returns the closest node in the Map (lowest euclidian distance)
def closestNode(data, map, mRows, mCols):
    # (row,col) of map node closest to data[t]
    result = (0,0)
    smallDist = 1.0e20
    for i in range(mRows):
        for j in range(mCols):
            ed = eucDist(map[i][j], data)
            if ed < smallDist:
                smallDist = ed
                result = (i, j)
    return result

#Returns most common value in a list
def mostCommon(lst):
    if len(lst) == 0: return -2
    return Counter(lst).most_common(1)[0][0]
```


```python
# Variables set up
np.random.seed(1) #Set seed to fix the random value generation (random but every time is the same sequence)
Dim = 33 #Nb of columns in data
Rows = 5; Cols = 4  #Dimmension of SOM
RangeMax = Rows + Cols #Maximum manhattan distance between two neurons
LearnMax = 0.1 #Initial learning rate
StepsMax = 5000 #Maximum nb of step
```

### Train


```python
#Trainsing the SOM Network
somNetwork = np.random.random_sample(size=(Rows,Cols,Dim)) #Centers
for s in range(StepsMax):
    if s % (StepsMax/10) == 0: print("step = ", str(s))
    pctLeft = 1.0 - ((s * 1.0) / StepsMax) #Small Pourcentage to make variables decrease depending on the amount of steps already done
    currRange = (int)(pctLeft * RangeMax) #Max distance between a neuron to update and the winner of the round
    currRate = pctLeft * LearnMax #Importance of the modification
    t = np.random.randint(len(dataX)) #Draw random element in data
    #1. Competition
    (bmuRow, bmuCol) = closestNode(dataX[t], somNetwork, Rows, Cols) #Find the winner
    #Update all neurons that are close enough to the winner
    for i in range(Rows):
        for j in range(Cols):
            if manhattanDist(bmuRow, bmuCol, i, j) < currRange: #2. Cooperation
                somNetwork[i][j] = somNetwork[i][j] + currRate * (dataX[t] - somNetwork[i][j]) #3. Adaptation

                
print("SOM construction complete \n")
```

    step =  0
    step =  500
    step =  1000
    step =  1500
    step =  2000
    step =  2500
    step =  3000
    step =  3500
    step =  4000
    step =  4500
    SOM construction complete 
    
    


```python
#List of the 20 vectors
somNetwork
```




    array([[[ 1.54727905e-01,  3.01551775e-01,  7.20702497e-01,
             -1.30774090e-01,  1.09862508e+00, -5.21957106e-01,
              1.32229467e+00, -1.05869258e+00,  1.33487315e+00,
             -1.41171592e+00,  1.15475150e+00, -1.54512790e+00,
              7.68289225e-01, -1.66517835e+00,  3.87149238e-01,
             -1.64727106e+00, -1.67580035e-01, -1.48224034e+00,
             -6.35880300e-01, -1.29927378e+00, -1.07104288e+00,
             -1.01342342e+00, -9.49888781e-01, -4.89473189e-01,
             -1.02474356e+00, -4.31756108e-01, -8.43851820e-01,
              5.99852753e-02, -6.19925458e-01,  2.89785601e-02,
             -3.12628865e-01,  4.66039396e-02,  1.55628399e-01],
            [-1.47502118e-01, -2.52800598e-01,  6.77303314e-01,
             -5.42604206e-01,  5.53423880e-01, -8.81346021e-01,
              8.74439686e-01, -1.16864783e+00,  6.13921263e-01,
             -1.23508526e+00,  6.89473558e-01, -1.19581834e+00,
              3.64099444e-01, -1.18628716e+00, -4.20672267e-02,
             -1.19118799e+00, -2.83672268e-01, -1.11009586e+00,
             -2.86705116e-01, -9.41469859e-01, -5.88537178e-01,
             -7.25929776e-01, -3.25907772e-01, -4.71008100e-01,
             -4.17985897e-01, -1.99428081e-01, -4.20005659e-01,
             -2.48034860e-01,  1.08724450e-01, -4.95793893e-01,
             -1.10275380e-02, -5.11670912e-01,  2.30256776e-01],
            [ 2.50413602e-02,  1.77049442e-01,  6.14854058e-01,
             -2.91625991e-01,  9.95044722e-02, -4.68335785e-01,
              3.22303203e-01, -3.39012508e-01,  5.30537709e-01,
             -4.01007997e-01,  7.23325761e-01, -6.94252080e-01,
              8.44061285e-01, -6.38186341e-01,  8.09012222e-01,
             -8.13660320e-01,  8.95078019e-01, -8.99319410e-01,
              9.00794822e-01, -1.13071340e+00,  5.71769761e-01,
             -1.19799022e+00,  5.13322061e-01, -1.11962042e+00,
              2.12206572e-01, -1.22337107e+00,  1.98430648e-01,
             -1.14402055e+00,  3.19876461e-02, -1.15790066e+00,
             -2.93951795e-01, -1.10514293e+00, -4.61525148e-01],
            [ 2.14731429e-01,  4.21445046e-01,  4.82923361e-01,
              3.93826749e-01,  4.71735141e-01,  2.62212454e-01,
              7.88191030e-01, -4.39156740e-02,  9.89666013e-01,
             -2.04137431e-01,  1.19599858e+00, -3.55723582e-01,
              1.38501471e+00, -5.06187431e-01,  1.48047915e+00,
             -8.64166484e-01,  1.41027892e+00, -1.11326778e+00,
              1.30438067e+00, -1.28709516e+00,  1.00693840e+00,
             -1.44114654e+00,  8.65715172e-01, -1.61220465e+00,
              5.48700082e-01, -2.01971806e+00,  3.83284390e-01,
             -1.62207098e+00,  6.98659310e-02, -1.47594461e+00,
             -2.91873601e-01, -1.49663987e+00, -4.82040284e-01]],
    
           [[-3.47057284e-01, -6.26096017e-01,  2.85139697e-01,
             -8.35157606e-01,  7.42485048e-01, -1.07532778e+00,
              7.65606771e-01, -1.27557178e+00,  6.78140594e-01,
             -1.30014813e+00,  4.95177381e-01, -1.28535823e+00,
             -7.07354049e-02, -1.32038816e+00, -3.95327652e-01,
             -9.97276282e-01, -6.57367537e-01, -7.38284267e-01,
             -9.52964748e-01, -6.71323373e-01, -9.69304771e-01,
             -4.74431454e-01, -6.25195530e-01, -1.24199641e-01,
             -4.91669064e-01, -1.03239449e-02, -4.64060205e-01,
             -1.22121412e-01,  6.16044119e-02, -5.01605753e-01,
              3.25454026e-01, -6.84341296e-01,  5.31250663e-01],
            [-1.27975637e+00, -1.37004638e+00,  6.33238486e-01,
             -1.21257836e+00, -2.72011441e-01, -1.56060118e+00,
              1.05919796e-01, -1.04373106e+00, -1.81180663e-01,
             -3.20608042e-01, -8.95944918e-02, -8.15193019e-01,
             -4.71306489e-01, -6.54950809e-01, -1.67574215e-01,
             -4.70833404e-01, -3.44468800e-01, -5.79725993e-01,
             -1.09240750e-01, -8.31825277e-01,  2.43736614e-02,
             -5.05305445e-01,  2.01726074e-01, -4.74607031e-01,
             -2.86283667e-01,  3.79686218e-01, -2.05406205e-01,
             -2.21389624e-01,  3.38070151e-01, -8.71095972e-01,
              1.27201540e-01, -8.85918239e-01, -1.10165096e-01],
            [-4.42552349e-01, -3.64916154e-01,  5.92207410e-01,
             -1.07409256e+00, -6.34587668e-01, -1.17566257e+00,
             -1.26328282e-01, -6.33353322e-01, -1.27040850e-01,
             -3.38422156e-01, -3.81673893e-02, -6.40549663e-01,
              3.12550892e-01, -6.14826094e-01, -2.08819456e-01,
             -2.99065888e-01,  1.78314653e-01, -4.56750952e-01,
              5.23936315e-01, -8.29092172e-01,  2.18782264e-01,
             -6.72428596e-01,  4.25594609e-01, -3.61094857e-01,
             -4.17249692e-02,  4.42411297e-02,  1.03931630e-01,
             -4.26165856e-01,  4.40273643e-01, -9.45388921e-01,
             -2.84266187e-01, -9.71254488e-01, -5.12438169e-02],
            [ 5.99463167e-02,  3.53084423e-01,  4.10943670e-01,
              9.40087204e-02,  7.29849582e-02, -3.40857639e-02,
              2.67830477e-01,  4.16901300e-02,  5.23737071e-01,
              6.48268829e-02,  6.25454307e-01, -1.82829035e-01,
              9.38192282e-01, -1.92961457e-01,  9.25125199e-01,
             -3.15352546e-01,  1.02876038e+00, -4.55525103e-01,
              1.16255023e+00, -6.99472538e-01,  9.59908250e-01,
             -7.78027750e-01,  9.45881461e-01, -7.64237607e-01,
              7.05338649e-01, -1.03827433e+00,  7.56895077e-01,
             -9.77672109e-01,  6.39631621e-01, -1.09453915e+00,
              2.73710425e-01, -1.17803510e+00,  1.88382747e-01]],
    
           [[-5.66720634e-01, -1.45093654e+00, -9.24109268e-01,
             -8.15201938e-01,  3.61242365e-01, -5.99943280e-01,
             -1.36712433e-01, -6.34941767e-01, -3.22544055e-01,
             -4.14347096e-01, -4.06504041e-01, -3.62662055e-01,
             -6.12638112e-01, -4.43784350e-01, -4.01679411e-01,
             -2.39714386e-01, -3.24445293e-01, -4.44469690e-01,
             -7.81167761e-01, -2.19767500e-02, -6.00512681e-01,
             -5.18777017e-01, -6.83585121e-01, -1.03450974e-01,
              6.37744971e-02, -2.73229259e-01, -7.24650926e-01,
             -7.17874007e-01, -7.76936445e-01, -5.94234226e-01,
              3.77507588e-01, -3.66062167e-01, -3.98239422e-02],
            [-2.37075900e-01, -1.21308791e+00, -3.21969124e-01,
             -8.99071856e-01, -3.33905145e-01, -9.38442642e-01,
              6.62123002e-02, -8.10137795e-01, -4.10212711e-01,
             -5.22928461e-01, -3.58522536e-01, -3.85998085e-01,
             -4.47114271e-01, -3.26705340e-01, -3.69784833e-01,
             -3.29399230e-01, -2.17277608e-01, -3.80416387e-01,
             -1.27344411e-01, -4.23313532e-01,  1.09586618e-01,
             -3.15519259e-01, -1.46317248e-01, -5.44950581e-01,
              2.34114940e-01, -3.21141866e-01,  5.72399363e-04,
             -6.16562796e-01,  5.29518041e-03, -4.66646936e-01,
              1.55071508e-01, -5.83839961e-01, -4.05420737e-02],
            [ 1.33736804e-02, -1.38058060e-01,  1.64195369e-01,
             -2.77519816e-01, -2.81712370e-01, -2.71350882e-01,
             -1.64723764e-01, -2.04678872e-01, -3.59422791e-01,
              1.21874531e-01, -8.83812513e-02,  1.64010553e-03,
             -1.09317540e-01,  5.41896826e-02,  1.00576076e-02,
              1.47412824e-01,  1.62365838e-01,  1.72222710e-01,
              5.01886858e-01, -7.88064555e-02,  4.09251018e-01,
              1.35663458e-01,  1.98657695e-01, -6.57599021e-02,
              2.86979172e-01,  7.30994061e-02,  5.11401662e-01,
             -1.69100521e-01,  4.63119308e-01, -3.18029681e-01,
              3.22188089e-01, -2.59189660e-01,  1.97109634e-01],
            [ 1.96645672e-01,  4.75712451e-01,  4.63988301e-02,
              5.01646801e-01,  5.94068509e-02,  5.67155739e-01,
              2.11915438e-01,  5.30841274e-01, -1.60163542e-02,
              5.30797657e-01,  4.13025386e-01,  5.68779129e-01,
              6.40233338e-01,  5.62682078e-01,  7.27105044e-01,
              3.64013508e-01,  7.01095866e-01,  4.44037010e-01,
              9.24909513e-01,  3.49226553e-01,  9.95870886e-01,
              3.04518915e-01,  9.38782679e-01,  2.39501436e-01,
              1.03642569e+00, -1.00807682e-01,  1.10763194e+00,
              3.85775009e-02,  1.07678990e+00, -2.67735963e-01,
              9.81686099e-01, -2.31743140e-01,  1.12826992e+00]],
    
           [[-7.85469620e-02, -3.36668703e-01, -7.84861853e-01,
             -1.60170148e-02, -5.12071000e-02, -1.08059095e-01,
             -7.86026224e-01,  2.38077995e-01, -2.13009332e-01,
              1.40809012e-01, -5.36422669e-01,  1.09875680e-01,
             -4.68422548e-01,  1.02537134e-01, -3.73950299e-01,
              3.72575952e-01, -5.66544830e-01,  2.23214869e-01,
             -9.20671349e-01,  2.52638479e-01, -3.18994963e-01,
              8.26848859e-02, -4.30295229e-01,  1.90043989e-01,
             -2.96032685e-01,  5.58741431e-02, -5.58803554e-01,
             -2.66370628e-01, -8.21257996e-01,  5.02008789e-02,
             -4.21271676e-01,  2.65731254e-02, -6.51642147e-01],
            [ 1.21247822e-01, -2.47826211e-01, -2.09065532e-01,
              1.86381095e-01, -3.25037805e-01,  8.33777339e-02,
             -2.39748078e-01,  1.12916388e-01, -3.92054298e-01,
              2.66155394e-01, -2.87972497e-01,  4.09860704e-01,
             -2.53268164e-01,  4.42797835e-01, -1.87070392e-01,
              3.99505655e-01, -2.96050844e-01,  2.84476857e-01,
             -7.19021696e-02,  3.46429583e-01,  6.58267280e-02,
              3.84468022e-01, -2.56843261e-02,  1.54156440e-01,
              6.60148235e-02,  1.89843378e-01, -9.57183654e-02,
              7.49807568e-02, -4.76044630e-02,  2.82469788e-01,
             -1.93869314e-01,  2.16009209e-01, -1.99276432e-01],
            [ 2.98081766e-01,  2.67799620e-01, -9.03331339e-03,
              4.36684430e-01, -2.41127826e-01,  5.39801002e-01,
             -1.26889853e-01,  5.14742408e-01, -3.51460791e-01,
              5.36964877e-01, -5.53938977e-02,  6.31432399e-01,
             -5.06240509e-02,  6.92600811e-01,  3.90657585e-02,
              5.89178758e-01,  7.21824316e-02,  6.28851953e-01,
              1.50863764e-01,  6.73835483e-01,  1.93003431e-01,
              5.94027306e-01,  2.42300610e-01,  5.42166246e-01,
              2.50244212e-01,  3.31588431e-01,  3.19807408e-01,
              4.40801432e-01,  2.10890672e-01,  5.15896654e-01,
              1.90282550e-01,  6.00791604e-01,  2.31628601e-01],
            [ 1.59141147e-01,  4.50799110e-01, -2.83358242e-01,
              6.33504628e-01,  1.77166489e-01,  6.80275061e-01,
              1.89664180e-01,  6.27270022e-01, -2.79808426e-01,
              5.92702964e-01,  1.56440805e-01,  6.97239134e-01,
              1.53376564e-01,  7.02619745e-01,  4.60903926e-01,
              5.25762760e-01,  5.96861247e-01,  7.23605937e-01,
              4.29272966e-01,  7.42365767e-01,  5.21900395e-01,
              7.35587299e-01,  3.69847438e-01,  5.30833114e-01,
              7.21166963e-01,  3.09820319e-01,  8.23208558e-01,
              4.78628959e-01,  4.54289123e-01,  3.06074953e-01,
              8.05972994e-01,  5.51519195e-01,  9.10058308e-01]],
    
           [[ 2.24750746e-01,  3.10963429e-01, -4.76756632e-01,
              4.47022287e-01, -2.49002277e-01,  6.38993137e-01,
             -7.79014693e-01,  7.65750616e-01, -4.60395552e-01,
              7.17966204e-01, -5.31247193e-01,  6.09097886e-01,
             -3.75358891e-01,  6.97336818e-01, -8.70113457e-01,
              6.24557367e-01, -4.47281015e-01,  6.42576165e-01,
             -6.97396793e-01,  5.96266175e-01, -2.66551808e-01,
              3.93662629e-01, -3.66976695e-01,  4.59380795e-01,
             -4.51993316e-01,  4.00193521e-01, -4.81866261e-01,
              4.27483633e-01, -7.78323382e-01,  4.76473947e-01,
             -9.60140043e-01,  4.00191968e-01, -1.11496625e+00],
            [-8.16563352e-02,  4.55486593e-01, -1.74961053e-01,
              4.02892237e-01, -2.86440880e-01,  6.88840944e-01,
             -6.35114634e-01,  7.18164585e-01, -4.56921703e-01,
              6.85153999e-01, -3.01697310e-01,  7.47094867e-01,
              9.69008069e-03,  8.23785132e-01, -7.40589917e-01,
              7.52812751e-01, -1.07871742e-01,  7.28912379e-01,
              6.23686153e-02,  8.55213369e-01, -5.68932739e-02,
              5.43987852e-01, -2.71308623e-01,  7.47197076e-01,
             -4.51318389e-01,  5.56644921e-01, -3.48486706e-01,
              6.92640574e-01, -1.59269352e-03,  6.48906553e-01,
             -4.61827577e-01,  7.89854019e-01, -5.22652786e-01],
            [ 2.93549301e-01,  5.69140880e-01,  1.60326280e-02,
              6.04271484e-01, -3.08459790e-01,  7.69384379e-01,
             -2.48913246e-01,  8.26882592e-01, -3.81613003e-01,
              7.86180006e-01, -1.94287943e-01,  8.58424335e-01,
             -1.97852159e-01,  8.93197909e-01, -2.23223167e-01,
              8.81883925e-01,  2.12786176e-02,  8.91266392e-01,
              5.44689651e-02,  9.55777174e-01, -2.14455582e-02,
              8.77386045e-01,  1.40477028e-01,  8.77359355e-01,
              7.06063990e-02,  7.11425976e-01,  1.20731476e-01,
              8.68862830e-01,  7.53668842e-02,  9.25800342e-01,
             -5.96331147e-02,  1.02855331e+00, -1.55337563e-01],
            [-2.47592446e-01,  6.01177734e-01, -3.69610459e-01,
              7.07931590e-01,  3.88911742e-01,  7.88045009e-01,
              4.72419967e-01,  8.29506511e-01, -1.50345522e-01,
              8.32582092e-01,  1.48175393e-01,  8.73283639e-01,
             -4.95336688e-01,  9.00483702e-01,  3.77185562e-01,
              7.39354415e-01,  3.20577594e-01,  9.15466230e-01,
             -1.73485913e-01,  9.84500712e-01,  2.59329889e-01,
              9.69565473e-01,  4.40097155e-02,  7.78191563e-01,
             -2.91750607e-02,  7.47769985e-01,  7.78045839e-01,
              9.01319120e-01, -3.52415241e-02,  8.70807673e-01,
              9.31429209e-01,  1.04470123e+00,  6.17684212e-01]]])



### Associating neurons to the cluster they mostly give


```python
# Associate each data label with a map node (closest one)
mapping = np.empty(shape=(Rows,Cols), dtype=object)

for i in range(Rows):
    for j in range(Cols):
        mapping[i][j] = []
        
sigma = [0 for i in range(Rows * Cols)] #Will be the mean of distances of every point related to a center (used in RBF)

#Associate to each data the closest node
for t in range(len(dataX)):
    (mRow, mCol) = closestNode(dataX[t], somNetwork, Rows, Cols)
    sigma[mRow*Cols + mCol] += eucDist(dataX[t], somNetwork[mRow][mCol]) #Adding the distance
    mapping[mRow][mCol].append(dataY[t]) #Association

#Knowing the class a node is the most related to
labelMap = np.zeros(shape=(Rows,Cols), dtype=np.int) #Most commun label of data associated to a node
for i in range(Rows):
    for j in range(Cols):
        labelMap[i][j] = mostCommon(mapping[i][j])

#Show on a map the repartition
plt.imshow(labelMap)
plt.colorbar()
plt.show()


```


![png](output_22_0.png)



```python
#Knowing how much data points are related to each neuron
countMap = np.zeros(shape=(Rows,Cols), dtype=np.int) 
for i in range(Rows):
    for j in range(Cols):
        tmp = 0
        label = labelMap[i][j]
        for k in mapping[i][j]:
            tmp += 1 if k == label else 0
        countMap[i][j] = tmp
```


```python
#Need to check if every neurons has been the winner once
#Othewise, when I calculate sigma, I would divid by nb of closest point (0) and generate error
print("Closest once:", len(np.argwhere(countMap > 0)))
print("Never closest:", Cols * Rows - len(np.argwhere(countMap > 0)))
```

    Closest once: 20
    Never closest: 0
    

## Radial Basis Function Neural Network (FS)


```python
#Basic Functions recquired for construction of the RBF

def gaussianFunction(d, sigma):
    return math.exp(- d ** 2 / (2 * sigma ** 2))

#inverse matrix m
def inverseMatrix(m):
    return np.matrix(m).I.A

#Transpose matrix m
def transposeMatrix(m):
    return np.matrix(m).T.A

#Giving a row, centers and weights, it returns the predicted value (any range, surely in ]-infinite;+infinite[)
def predict(x, c, weights):
    tmp = 0
    tmpO = []
    #Calculating every output o
    for i in range(len(c)):
        tmpO.append(gaussianFunction(eucDist(x, c[i]),sigma[i]))
    tmpO.append(1)
    return (tmpO * np.matrix(weights).T).A[0][0] #f = o * transposed(w)

#Calculate least square error between result and labels
def leastSquare(result, labels):
    tmp = 0
    for x in range(len(result)):
        tmp += (result[x] - labels[x]) ** 2
    return tmp / (len(result)*2)

#Calculate classification accuracy:
#TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative
#Accuracy = (TP + TN) / (TP + TN + FP + FN)
def accuracy(predicted, reality):
    cpt = 0
    for i in range(len(predicted)):
        if predicted[i] == reality[i]: cpt += 1
    return cpt/len(predicted)
```


```python
# Variable set up

#Flat it as we want one layer and not  a 2D list (grid)
c = [somNetwork[i][j] for i in range(len(somNetwork)) for j in range(len(somNetwork[i]))] 

#Tried different distances
distanceToUse = "Euclidian" 

```


```python
#Having a SIGMA per variables to have larger/smaller zones covered by a center
for i in range(len(countMap)):
    for j in range(len(countMap[i])):
        if countMap[i][j] != 0:
            sigma[i*Cols + j] = sigma[i*Cols + j]/countMap[i][j] #Divide the sum of distance to the point by the nb of points

#Deleting the centers that have a sigma equals to 0
#That means they don't have any points related and that would be unrelevant to have vectors not representatives of the datas
cpt = 0
for i in range(len(sigma)):
    if sigma[i-cpt] == 0:
        sigma.pop(i - cpt)
        c.pop(i - cpt)
        cpt += 1
        
sizeHidden = len(c)
```


```python
#Set train test dataset
#We separate them to have fresh data to choose the threshold to state what is -1 and what is 1
#As the dataset is pretty small (330 lines only), we will still use the training data for thresh determination
#Stating on a 66 population might not be the best choice
trainPart = 0.8

dataIndex = [i for i in range(len(dataX))]
shuffledIndex = random.sample(dataIndex, len(dataIndex)) #Randomize Indexes

#Train Set
trainSetX = []
trainSetY = []
for i in shuffledIndex[:int(trainPart * len(dataX))]:
    trainSetX.append(dataX[i])
    trainSetY.append(dataY[i])

#Test Set
testSetX = []
testSetY = []
for i in shuffledIndex[int(trainPart * len(dataX)):]:
    testSetX.append(dataX[i])
    testSetY.append(dataY[i])

```

### Train: Weights calculation


```python
#Calculation of o for each data in train set
o = np.zeros(shape=(len(trainSetX), sizeHidden + 1)) 
for i in range(len(trainSetX)):
    for j in range(sizeHidden):
        if distanceToUse == "Euclidian":
            o[i][j] = gaussianFunction(eucDist(trainSetX[i], c[j]),sigma[j])
    o[i][sizeHidden] = 1

```


```python
#Calculation of all weights minimizing to least square cost function 
weights = np.dot( 
    inverseMatrix( np.dot(transposeMatrix(o) , o) ),
    np.dot( transposeMatrix(o) , trainSetY )
)
```

### Threshold selection and Test


```python
resultTrain = []
for i in trainSetX:
    resultTrain.append( predict(i, c, weights))
```


```python
#Calculate maximal accuracy and corresponding threshold on TEST SET ONLY
resultTest = []
for i in testSetX:
    resultTest.append( predict(i, c, weights))

accuracyTest = 0
threshTest = 0
for i in range(-1000,1000):
    r = [-1 if e < i/1000 else 1 for e in resultTest]
    tmp = accuracy(r, testSetY)
    if tmp > accuracyTest:
        accuracyTest = tmp
        threshTest = i/1000

print("Accuracy on test data:", accuracyTest, "with a thresh of:", threshTest)
```

    Accuracy on test data: 0.9848484848484849 with a thresh of: -0.217
    


```python
#Calculate maximal accuracy and corresponding threshold the whole dataset
resultO = []
for i in dataX:
    resultO.append( predict(i, c, weights))

accuracyOverall = 0
threshOverall = 0
for i in range(-1000,1000):
    r = [-1 if e < i/1000 else 1 for e in resultO]
    tmp = accuracy(r, dataY)
    if tmp > accuracyOverall:
        accuracyOverall = tmp
        threshOverall = i/1000

print("Accuracy on all data:", accuracyOverall, "with a thresh of:", threshOverall)
```

    Accuracy on all data: 0.9575757575757575 with a thresh of: -0.119
    


```python
print("Final results with the overall threshold (" + str(threshOverall) + "): ")
print(" - Accuracy on train data:", accuracy([-1 if e < threshOverall else 1 for e in resultTrain], trainSetY) *100)
print(" - Accuracy on test data: ", accuracy([-1 if e < threshOverall else 1 for e in resultTest], testSetY)*100)
print(" - Accuracy on all data:  ", accuracy([-1 if e < threshOverall else 1 for e in resultO],dataY)*100)
```

    Final results with the overall threshold (-0.119): 
     - Accuracy on train data: 95.45454545454545
     - Accuracy on test data:  96.96969696969697
     - Accuracy on all data:   95.75757575757575
    

## Prediction


```python
#Predict on the dataset provided as "Test Set" with threshold previously selected
dataToPredictX = [list(i) for i in scipy.io.loadmat(dataTestFile + ".mat")[dataTestFile]] 
#[list(i) for i in np.loadtxt(dataTestFile, delimiter=",", dtype=np.float64)]
prediction = []
for i in dataToPredictX:
    prediction.append( -1 if predict(i, c, weights) < threshOverall else 1 )

prediction
```




    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1]




```python
#Write results in a csv
with open("prediction.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    for i in zip(prediction):
        writer.writerow(i)
```

## To try / To go further
 - Bottom up/Top down ?
     - Train bigger SOM?
     - Building up to optimal one?
