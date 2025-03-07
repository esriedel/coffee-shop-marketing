```python
# Research questions
# Are there groups of coffee shops based on size, local traffic, and hours?
# Do these groups benefit differently from marketing spend?
```


```python
!pip install seaborn
%pip install seaborn
!pip install scikit-learn
%pip install scikit-learn
!pip install statsmodels
%pip install statsmodels
```

    Requirement already satisfied: seaborn in /Applications/anaconda3/lib/python3.12/site-packages (0.13.2)
    Requirement already satisfied: numpy!=1.24.0,>=1.20 in /Applications/anaconda3/lib/python3.12/site-packages (from seaborn) (1.26.4)
    Requirement already satisfied: pandas>=1.2 in /Applications/anaconda3/lib/python3.12/site-packages (from seaborn) (2.2.2)
    Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in /Applications/anaconda3/lib/python3.12/site-packages (from seaborn) (3.9.2)
    Requirement already satisfied: contourpy>=1.0.1 in /Applications/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /Applications/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /Applications/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.51.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /Applications/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.4)
    Requirement already satisfied: packaging>=20.0 in /Applications/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (24.1)
    Requirement already satisfied: pillow>=8 in /Applications/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (10.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /Applications/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.1.2)
    Requirement already satisfied: python-dateutil>=2.7 in /Applications/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /Applications/anaconda3/lib/python3.12/site-packages (from pandas>=1.2->seaborn) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /Applications/anaconda3/lib/python3.12/site-packages (from pandas>=1.2->seaborn) (2023.3)
    Requirement already satisfied: six>=1.5 in /Applications/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.16.0)
    Requirement already satisfied: seaborn in /opt/anaconda3/lib/python3.12/site-packages (0.13.2)
    Requirement already satisfied: numpy!=1.24.0,>=1.20 in /opt/anaconda3/lib/python3.12/site-packages (from seaborn) (2.2.1)
    Requirement already satisfied: pandas>=1.2 in /opt/anaconda3/lib/python3.12/site-packages (from seaborn) (2.2.3)
    Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in /opt/anaconda3/lib/python3.12/site-packages (from seaborn) (3.10.0)
    Requirement already satisfied: contourpy>=1.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.55.3)
    Requirement already satisfied: kiwisolver>=1.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (24.1)
    Requirement already satisfied: pillow>=8 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (10.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.2.1)
    Requirement already satisfied: python-dateutil>=2.7 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas>=1.2->seaborn) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas>=1.2->seaborn) (2024.2)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: scikit-learn in /Applications/anaconda3/lib/python3.12/site-packages (1.5.1)
    Requirement already satisfied: numpy>=1.19.5 in /Applications/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.26.4)
    Requirement already satisfied: scipy>=1.6.0 in /Applications/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.13.1)
    Requirement already satisfied: joblib>=1.2.0 in /Applications/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /Applications/anaconda3/lib/python3.12/site-packages (from scikit-learn) (3.5.0)
    Requirement already satisfied: scikit-learn in /opt/anaconda3/lib/python3.12/site-packages (1.6.1)
    Requirement already satisfied: numpy>=1.19.5 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (2.2.1)
    Requirement already satisfied: scipy>=1.6.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.15.0)
    Requirement already satisfied: joblib>=1.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (3.5.0)
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: statsmodels in /Applications/anaconda3/lib/python3.12/site-packages (0.14.2)
    Requirement already satisfied: numpy>=1.22.3 in /Applications/anaconda3/lib/python3.12/site-packages (from statsmodels) (1.26.4)
    Requirement already satisfied: scipy!=1.9.2,>=1.8 in /Applications/anaconda3/lib/python3.12/site-packages (from statsmodels) (1.13.1)
    Requirement already satisfied: pandas!=2.1.0,>=1.4 in /Applications/anaconda3/lib/python3.12/site-packages (from statsmodels) (2.2.2)
    Requirement already satisfied: patsy>=0.5.6 in /Applications/anaconda3/lib/python3.12/site-packages (from statsmodels) (0.5.6)
    Requirement already satisfied: packaging>=21.3 in /Applications/anaconda3/lib/python3.12/site-packages (from statsmodels) (24.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /Applications/anaconda3/lib/python3.12/site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /Applications/anaconda3/lib/python3.12/site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /Applications/anaconda3/lib/python3.12/site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2023.3)
    Requirement already satisfied: six in /Applications/anaconda3/lib/python3.12/site-packages (from patsy>=0.5.6->statsmodels) (1.16.0)
    Requirement already satisfied: statsmodels in /opt/anaconda3/lib/python3.12/site-packages (0.14.4)
    Requirement already satisfied: numpy<3,>=1.22.3 in /opt/anaconda3/lib/python3.12/site-packages (from statsmodels) (2.2.1)
    Requirement already satisfied: scipy!=1.9.2,>=1.8 in /opt/anaconda3/lib/python3.12/site-packages (from statsmodels) (1.15.0)
    Requirement already satisfied: pandas!=2.1.0,>=1.4 in /opt/anaconda3/lib/python3.12/site-packages (from statsmodels) (2.2.3)
    Requirement already satisfied: patsy>=0.5.6 in /opt/anaconda3/lib/python3.12/site-packages (from statsmodels) (1.0.1)
    Requirement already satisfied: packaging>=21.3 in /opt/anaconda3/lib/python3.12/site-packages (from statsmodels) (24.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/lib/python3.12/site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2024.2)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas!=2.1.0,>=1.4->statsmodels) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.



```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
```


```python
coffee=pd.read_csv('coffee_shop_revenue.csv')
```


```python
coffee.rename(columns={'Number_of_Customers_Per_Day': 'Customers', 'Average_Order_Value': 'Order', 'Operating_Hours_Per_Day': 'Hours', 'Number_of_Employees': 'Employees', 'Marketing_Spend_Per_Day': 'Marketing', 'Location_Foot_Traffic': 'Traffic', 'Daily_Revenue': 'Revenue'}, inplace=True)
print(coffee)
```

          Customers  Order  Hours  Employees  Marketing  Traffic  Revenue
    0           152   6.74     14          4     106.62       97  1547.81
    1           485   4.50     12          8      57.83      744  2084.68
    2           398   9.09      6          6      91.76      636  3118.39
    3           320   8.48     17          4     462.63      770  2912.20
    4           156   7.44     17          2     412.52      232  1663.42
    ...         ...    ...    ...        ...        ...      ...      ...
    1995        372   6.41     11          4     466.11      913  2816.85
    1996        105   3.01     11          7      12.62      235   337.97
    1997         89   5.28     16          9     376.64      310   951.34
    1998        403   9.41      7         12     452.49      577  4266.21
    1999         89   6.88     13         14      78.46      322   914.24
    
    [2000 rows x 7 columns]



```python
coffee.describe()
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
      <th>Customers</th>
      <th>Order</th>
      <th>Hours</th>
      <th>Employees</th>
      <th>Marketing</th>
      <th>Traffic</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>274.296000</td>
      <td>6.261215</td>
      <td>11.667000</td>
      <td>7.947000</td>
      <td>252.614160</td>
      <td>534.893500</td>
      <td>1917.325940</td>
    </tr>
    <tr>
      <th>std</th>
      <td>129.441933</td>
      <td>2.175832</td>
      <td>3.438608</td>
      <td>3.742218</td>
      <td>141.136004</td>
      <td>271.662295</td>
      <td>976.202746</td>
    </tr>
    <tr>
      <th>min</th>
      <td>50.000000</td>
      <td>2.500000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>10.120000</td>
      <td>50.000000</td>
      <td>-58.950000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>164.000000</td>
      <td>4.410000</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>130.125000</td>
      <td>302.000000</td>
      <td>1140.085000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>275.000000</td>
      <td>6.300000</td>
      <td>12.000000</td>
      <td>8.000000</td>
      <td>250.995000</td>
      <td>540.000000</td>
      <td>1770.775000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>386.000000</td>
      <td>8.120000</td>
      <td>15.000000</td>
      <td>11.000000</td>
      <td>375.352500</td>
      <td>767.000000</td>
      <td>2530.455000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>499.000000</td>
      <td>10.000000</td>
      <td>17.000000</td>
      <td>14.000000</td>
      <td>499.740000</td>
      <td>999.000000</td>
      <td>5114.600000</td>
    </tr>
  </tbody>
</table>
</div>




```python
coffee.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 7 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Customers  2000 non-null   int64  
     1   Order      2000 non-null   float64
     2   Hours      2000 non-null   int64  
     3   Employees  2000 non-null   int64  
     4   Marketing  2000 non-null   float64
     5   Traffic    2000 non-null   int64  
     6   Revenue    2000 non-null   float64
    dtypes: float64(3), int64(4)
    memory usage: 109.5 KB



```python
missing_values=coffee.isnull().sum()
missing_values
```




    Customers    0
    Order        0
    Hours        0
    Employees    0
    Marketing    0
    Traffic      0
    Revenue      0
    dtype: int64




```python
coffee.hist(figsize=(10, 8))  
plt.show()
```


    
![png](output_8_0.png)
    



```python
correlation_matrix=coffee.corr(method='spearman')
correlation_matrix
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
      <th>Customers</th>
      <th>Order</th>
      <th>Hours</th>
      <th>Employees</th>
      <th>Marketing</th>
      <th>Traffic</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Customers</th>
      <td>1.000000</td>
      <td>-0.014640</td>
      <td>-0.000548</td>
      <td>0.000234</td>
      <td>0.027216</td>
      <td>-0.000989</td>
      <td>0.751988</td>
    </tr>
    <tr>
      <th>Order</th>
      <td>-0.014640</td>
      <td>1.000000</td>
      <td>-0.016621</td>
      <td>0.010852</td>
      <td>0.019729</td>
      <td>0.017450</td>
      <td>0.515663</td>
    </tr>
    <tr>
      <th>Hours</th>
      <td>-0.000548</td>
      <td>-0.016621</td>
      <td>1.000000</td>
      <td>-0.030340</td>
      <td>0.019644</td>
      <td>0.014904</td>
      <td>-0.005695</td>
    </tr>
    <tr>
      <th>Employees</th>
      <td>0.000234</td>
      <td>0.010852</td>
      <td>-0.030340</td>
      <td>1.000000</td>
      <td>0.026757</td>
      <td>-0.041735</td>
      <td>-0.000293</td>
    </tr>
    <tr>
      <th>Marketing</th>
      <td>0.027216</td>
      <td>0.019729</td>
      <td>0.019644</td>
      <td>0.026757</td>
      <td>1.000000</td>
      <td>-0.012289</td>
      <td>0.252762</td>
    </tr>
    <tr>
      <th>Traffic</th>
      <td>-0.000989</td>
      <td>0.017450</td>
      <td>0.014904</td>
      <td>-0.041735</td>
      <td>-0.012289</td>
      <td>1.000000</td>
      <td>0.013800</td>
    </tr>
    <tr>
      <th>Revenue</th>
      <td>0.751988</td>
      <td>0.515663</td>
      <td>-0.005695</td>
      <td>-0.000293</td>
      <td>0.252762</td>
      <td>0.013800</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(correlation_matrix, cmap='coolwarm')
```




    <Axes: >




    
![png](output_10_1.png)
    



```python
sns.pairplot(coffee)
plt.show()
```


    
![png](output_11_0.png)
    



```python
scaler = StandardScaler()

shop = scaler.fit_transform(coffee[['Customers','Order', 'Hours', 'Employees', 'Traffic']])

scaler = StandardScaler()

scaler_df = scaler.fit_transform(coffee[['Customers', 'Order', 'Hours', 'Employees', 'Traffic']])

shop = pd.DataFrame(scaler_df, columns=['Customers', 'Order', 'Hours', 'Employees', 'Traffic'])

shop.describe()

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
      <th>Customers</th>
      <th>Order</th>
      <th>Hours</th>
      <th>Employees</th>
      <th>Traffic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.506706e-17</td>
      <td>1.154632e-17</td>
      <td>7.283063e-17</td>
      <td>-1.243450e-17</td>
      <td>-6.394885e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.733226e+00</td>
      <td>-1.729065e+00</td>
      <td>-1.648463e+00</td>
      <td>-1.589562e+00</td>
      <td>-1.785359e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-8.523017e-01</td>
      <td>-8.510206e-01</td>
      <td>-7.757986e-01</td>
      <td>-7.876979e-01</td>
      <td>-8.575049e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.440092e-03</td>
      <td>1.782982e-02</td>
      <td>9.686574e-02</td>
      <td>1.416627e-02</td>
      <td>1.880194e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.631819e-01</td>
      <td>8.545006e-01</td>
      <td>9.695301e-01</td>
      <td>8.160304e-01</td>
      <td>8.546072e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.736379e+00</td>
      <td>1.718754e+00</td>
      <td>1.551306e+00</td>
      <td>1.617895e+00</td>
      <td>1.708822e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
shop.rename(columns={'Customers': 'Customers_scaled', 'Order': 'Order_scaled', 'Hours': 'Hours_scaled', 'Employees': 'Employees_scaled', 'Traffic':'Traffic_scaled'}, inplace=True)

```


```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = [] 

for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(shop)  
    wcss.append(kmeans.inertia_) 


plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()

```


    
![png](output_14_0.png)
    



```python
silhouette_scores = []
k_range = range(2, 11)  # Test cluster numbers from 2 to 10

for k in k_range:

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(shop)
    silhouette_avg = silhouette_score(shop, labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(k_range, silhouette_scores)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette analysis for optimal k")
plt.show()
```


    
![png](output_15_0.png)
    



```python
coffee = coffee.reset_index(drop=True)
shop = shop.reset_index(drop=True)

coffee2 = pd.concat([coffee, shop], axis=1)
coffee2.describe()
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
      <th>Customers</th>
      <th>Order</th>
      <th>Hours</th>
      <th>Employees</th>
      <th>Marketing</th>
      <th>Traffic</th>
      <th>Revenue</th>
      <th>Customers_scaled</th>
      <th>Order_scaled</th>
      <th>Hours_scaled</th>
      <th>Employees_scaled</th>
      <th>Traffic_scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>274.296000</td>
      <td>6.261215</td>
      <td>11.667000</td>
      <td>7.947000</td>
      <td>252.614160</td>
      <td>534.893500</td>
      <td>1917.325940</td>
      <td>5.506706e-17</td>
      <td>1.154632e-17</td>
      <td>7.283063e-17</td>
      <td>-1.243450e-17</td>
      <td>-6.394885e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>129.441933</td>
      <td>2.175832</td>
      <td>3.438608</td>
      <td>3.742218</td>
      <td>141.136004</td>
      <td>271.662295</td>
      <td>976.202746</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>50.000000</td>
      <td>2.500000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>10.120000</td>
      <td>50.000000</td>
      <td>-58.950000</td>
      <td>-1.733226e+00</td>
      <td>-1.729065e+00</td>
      <td>-1.648463e+00</td>
      <td>-1.589562e+00</td>
      <td>-1.785359e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>164.000000</td>
      <td>4.410000</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>130.125000</td>
      <td>302.000000</td>
      <td>1140.085000</td>
      <td>-8.523017e-01</td>
      <td>-8.510206e-01</td>
      <td>-7.757986e-01</td>
      <td>-7.876979e-01</td>
      <td>-8.575049e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>275.000000</td>
      <td>6.300000</td>
      <td>12.000000</td>
      <td>8.000000</td>
      <td>250.995000</td>
      <td>540.000000</td>
      <td>1770.775000</td>
      <td>5.440092e-03</td>
      <td>1.782982e-02</td>
      <td>9.686574e-02</td>
      <td>1.416627e-02</td>
      <td>1.880194e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>386.000000</td>
      <td>8.120000</td>
      <td>15.000000</td>
      <td>11.000000</td>
      <td>375.352500</td>
      <td>767.000000</td>
      <td>2530.455000</td>
      <td>8.631819e-01</td>
      <td>8.545006e-01</td>
      <td>9.695301e-01</td>
      <td>8.160304e-01</td>
      <td>8.546072e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>499.000000</td>
      <td>10.000000</td>
      <td>17.000000</td>
      <td>14.000000</td>
      <td>499.740000</td>
      <td>999.000000</td>
      <td>5114.600000</td>
      <td>1.736379e+00</td>
      <td>1.718754e+00</td>
      <td>1.551306e+00</td>
      <td>1.617895e+00</td>
      <td>1.708822e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
kmeans = KMeans(n_clusters=9, random_state=42, n_init='auto') 
kmeans.fit(shop)

clusters = kmeans.labels_

coffee2['cluster'] = clusters

```


```python
coffee2.describe()
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
      <th>Customers</th>
      <th>Order</th>
      <th>Hours</th>
      <th>Employees</th>
      <th>Marketing</th>
      <th>Traffic</th>
      <th>Revenue</th>
      <th>Customers_scaled</th>
      <th>Order_scaled</th>
      <th>Hours_scaled</th>
      <th>Employees_scaled</th>
      <th>Traffic_scaled</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
      <td>2.000000e+03</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>274.296000</td>
      <td>6.261215</td>
      <td>11.667000</td>
      <td>7.947000</td>
      <td>252.614160</td>
      <td>534.893500</td>
      <td>1917.325940</td>
      <td>5.506706e-17</td>
      <td>1.154632e-17</td>
      <td>7.283063e-17</td>
      <td>-1.243450e-17</td>
      <td>-6.394885e-17</td>
      <td>4.028500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>129.441933</td>
      <td>2.175832</td>
      <td>3.438608</td>
      <td>3.742218</td>
      <td>141.136004</td>
      <td>271.662295</td>
      <td>976.202746</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
      <td>1.000250e+00</td>
      <td>2.556943</td>
    </tr>
    <tr>
      <th>min</th>
      <td>50.000000</td>
      <td>2.500000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>10.120000</td>
      <td>50.000000</td>
      <td>-58.950000</td>
      <td>-1.733226e+00</td>
      <td>-1.729065e+00</td>
      <td>-1.648463e+00</td>
      <td>-1.589562e+00</td>
      <td>-1.785359e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>164.000000</td>
      <td>4.410000</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>130.125000</td>
      <td>302.000000</td>
      <td>1140.085000</td>
      <td>-8.523017e-01</td>
      <td>-8.510206e-01</td>
      <td>-7.757986e-01</td>
      <td>-7.876979e-01</td>
      <td>-8.575049e-01</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>275.000000</td>
      <td>6.300000</td>
      <td>12.000000</td>
      <td>8.000000</td>
      <td>250.995000</td>
      <td>540.000000</td>
      <td>1770.775000</td>
      <td>5.440092e-03</td>
      <td>1.782982e-02</td>
      <td>9.686574e-02</td>
      <td>1.416627e-02</td>
      <td>1.880194e-02</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>386.000000</td>
      <td>8.120000</td>
      <td>15.000000</td>
      <td>11.000000</td>
      <td>375.352500</td>
      <td>767.000000</td>
      <td>2530.455000</td>
      <td>8.631819e-01</td>
      <td>8.545006e-01</td>
      <td>9.695301e-01</td>
      <td>8.160304e-01</td>
      <td>8.546072e-01</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>499.000000</td>
      <td>10.000000</td>
      <td>17.000000</td>
      <td>14.000000</td>
      <td>499.740000</td>
      <td>999.000000</td>
      <td>5114.600000</td>
      <td>1.736379e+00</td>
      <td>1.718754e+00</td>
      <td>1.551306e+00</td>
      <td>1.617895e+00</td>
      <td>1.708822e+00</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
coffee2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 13 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Customers         2000 non-null   int64  
     1   Order             2000 non-null   float64
     2   Hours             2000 non-null   int64  
     3   Employees         2000 non-null   int64  
     4   Marketing         2000 non-null   float64
     5   Traffic           2000 non-null   int64  
     6   Revenue           2000 non-null   float64
     7   Customers_scaled  2000 non-null   float64
     8   Order_scaled      2000 non-null   float64
     9   Hours_scaled      2000 non-null   float64
     10  Employees_scaled  2000 non-null   float64
     11  Traffic_scaled    2000 non-null   float64
     12  cluster           2000 non-null   int32  
    dtypes: float64(8), int32(1), int64(4)
    memory usage: 195.4 KB



```python
sns.boxplot(x="cluster", y="Hours", data=coffee2)

mean_value = coffee2['Hours'].mean()

plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

plt.legend()

plt.show()

```


    
![png](output_20_0.png)
    



```python
sns.boxplot(x="cluster", y="Employees", data=coffee2)

mean_value = coffee2['Employees'].mean()

plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

plt.legend()

plt.show()
```


    
![png](output_21_0.png)
    



```python
sns.boxplot(x="cluster", y="Customers", data=coffee2)

mean_value = coffee2['Customers'].mean()

plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

plt.legend()

plt.show()

```


    
![png](output_22_0.png)
    



```python
sns.boxplot(x="cluster", y="Order", data=coffee2)

mean_value = coffee2['Order'].mean()

plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

plt.legend()

plt.show()

```


    
![png](output_23_0.png)
    



```python
sns.boxplot(x="cluster", y="Traffic", data=coffee2)

mean_value = coffee2['Traffic'].mean()

plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

plt.legend()

plt.show()
```


    
![png](output_24_0.png)
    



```python
median_pivot = result.pivot_table(values=['Customers', 'Order', 'Hours', 'Employees', 'Traffic'], index='cluster', aggfunc='median')
print("\nMedians by Category (using pivot_table()):\n", median_pivot)
```

    
    Medians by Category (using pivot_table()):
              Customers  Employees  Hours  Order  Traffic
    cluster                                             
    0            162.5        5.0   14.0  8.630    341.5
    1            388.0        7.0   14.0  4.250    245.0
    2            161.5        4.0   13.0  4.140    756.0
    3            377.0       11.0   10.0  8.015    307.5
    4            200.0       12.0   15.0  5.380    641.0
    5            408.5        9.0    9.0  4.330    766.0
    6            184.0        9.0    8.0  8.320    782.0
    7            402.5        5.0   14.0  7.805    771.0
    8            166.5        7.0    8.0  4.870    270.0



```python
mean=coffee2.mean()
print(mean)
```

    Customers           2.742960e+02
    Order               6.261215e+00
    Hours               1.166700e+01
    Employees           7.947000e+00
    Marketing           2.526142e+02
    Traffic             5.348935e+02
    Revenue             1.917326e+03
    Customers_scaled    5.506706e-17
    Order_scaled        1.154632e-17
    Hours_scaled        7.283063e-17
    Employees_scaled   -1.243450e-17
    Traffic_scaled     -6.394885e-17
    cluster             4.028500e+00
    dtype: float64



```python
mean_by_cluster = coffee2.groupby("cluster")[['Customers_scaled', 'Order_scaled', 'Hours_scaled', 'Employees_scaled', 'Traffic_scaled']].mean()
print(mean_by_category)


```

             Customers_scaled  Order_scaled  Hours_scaled  Employees_scaled  \
    cluster                                                                   
    0               -0.784315      0.967888      0.637277         -0.661410   
    1                0.773254     -0.861459      0.526212         -0.325488   
    2               -0.815282     -0.879191      0.423269         -0.910464   
    3                0.705752      0.723088     -0.434840          0.841226   
    4               -0.532417     -0.372994      0.778215          1.049393   
    5                0.919299     -0.765818     -0.763090          0.204901   
    6               -0.638339      0.840566     -0.921889          0.163848   
    7                0.910791      0.650923      0.602222         -0.666512   
    8               -0.763510     -0.604507     -0.980805         -0.116932   
    
             Traffic_scaled  
    cluster                  
    0             -0.639678  
    1             -1.009832  
    2              0.803402  
    3             -0.775974  
    4              0.335564  
    5              0.799682  
    6              0.808509  
    7              0.735848  
    8             -0.884296  



```python

df = pd.DataFrame(mean_by_cluster)


def recode_integer_to_string(x):
    if pd.isna(x):
        return "Out of Range"
    if isinstance(x, (int, float)):  # Check if the value is numeric
        if x <= -0.5:
            return "Low"
        elif -0.5 < x <= 0.5:
            return "Medium"
        elif x > 0.5:
            return "High"
    return "Invalid"


columns_to_recode = ['Customers_scaled', 'Order_scaled', 'Hours_scaled', 'Employees_scaled', 'Traffic_scaled']

for column in columns_to_recode:
    df[f'{column}_recode'] = df[column].apply(recode_integer_to_string)

df2=df[['Customers_scaled_recode', 'Order_scaled_recode', 'Hours_scaled_recode', 'Employees_scaled_recode', 'Traffic_scaled_recode']]


print(df2)

```

            Customers_scaled_recode Order_scaled_recode Hours_scaled_recode  \
    cluster                                                                   
    0                           Low                High                High   
    1                          High                 Low                High   
    2                           Low                 Low              Medium   
    3                          High                High              Medium   
    4                           Low              Medium                High   
    5                          High                 Low                 Low   
    6                           Low                High                 Low   
    7                          High                High                High   
    8                           Low                 Low                 Low   
    
            Employees_scaled_recode Traffic_scaled_recode  
    cluster                                                
    0                           Low                   Low  
    1                        Medium                   Low  
    2                           Low                  High  
    3                          High                   Low  
    4                          High                Medium  
    5                        Medium                  High  
    6                        Medium                  High  
    7                           Low                  High  
    8                        Medium                   Low  



```python
counts = coffee2['cluster'].value_counts()
print(counts)
```

    cluster
    4    260
    3    244
    7    236
    1    229
    6    225
    0    218
    8    210
    5    206
    2    172
    Name: count, dtype: int64



```python


# Fit separate models
model_A = sm.OLS(result[result['cluster'] == 0]['Revenue'], sm.add_constant(result[result['cluster']== 0]['Marketing'])).fit()
model_B = sm.OLS(result[result['cluster'] == 1]['Revenue'], sm.add_constant(result[result['cluster']== 1]['Marketing'])).fit()
model_C = sm.OLS(result[result['cluster'] == 2]['Revenue'], sm.add_constant(result[result['cluster']== 2]['Marketing'])).fit()
model_D = sm.OLS(result[result['cluster'] == 3]['Revenue'], sm.add_constant(result[result['cluster']== 3]['Marketing'])).fit()
model_E = sm.OLS(result[result['cluster'] == 4]['Revenue'], sm.add_constant(result[result['cluster']== 4]['Marketing'])).fit()
model_F = sm.OLS(result[result['cluster'] == 5]['Revenue'], sm.add_constant(result[result['cluster']== 5]['Marketing'])).fit()
model_G = sm.OLS(result[result['cluster'] == 6]['Revenue'], sm.add_constant(result[result['cluster']== 6]['Marketing'])).fit()
model_H = sm.OLS(result[result['cluster'] == 7]['Revenue'], sm.add_constant(result[result['cluster']== 7]['Marketing'])).fit()
model_I = sm.OLS(result[result['cluster'] == 8]['Revenue'], sm.add_constant(result[result['cluster']== 8]['Marketing'])).fit()

# Print model summaries
print("Model A Summary:")
print(model_A.summary())
print("Model B Summary:")
print(model_B.summary())
print("Model C Summary:")
print(model_C.summary())
print("Model D Summary:")
print(model_D.summary())
print("Model E Summary:")
print(model_E.summary())
print("Model F Summary:")
print(model_F.summary())
print("Model G Summary:")
print(model_G.summary())
print("Model H Summary:")
print(model_H.summary())
print("Model I Summary:")
print(model_I.summary())

```

    Model A Summary:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Revenue   R-squared:                       0.038
    Model:                            OLS   Adj. R-squared:                  0.034
    Method:                 Least Squares   F-statistic:                     8.636
    Date:                Fri, 07 Mar 2025   Prob (F-statistic):            0.00365
    Time:                        13:40:07   Log-Likelihood:                -1735.0
    No. Observations:                 218   AIC:                             3474.
    Df Residuals:                     216   BIC:                             3481.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       1454.4886     96.819     15.023      0.000    1263.658    1645.319
    Marketing      0.9925      0.338      2.939      0.004       0.327       1.658
    ==============================================================================
    Omnibus:                       10.618   Durbin-Watson:                   2.195
    Prob(Omnibus):                  0.005   Jarque-Bera (JB):               10.991
    Skew:                           0.547   Prob(JB):                      0.00411
    Kurtosis:                       3.122   Cond. No.                         589.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Model B Summary:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Revenue   R-squared:                       0.101
    Model:                            OLS   Adj. R-squared:                  0.097
    Method:                 Least Squares   F-statistic:                     25.42
    Date:                Fri, 07 Mar 2025   Prob (F-statistic):           9.44e-07
    Time:                        13:40:07   Log-Likelihood:                -1806.6
    No. Observations:                 229   AIC:                             3617.
    Df Residuals:                     227   BIC:                             3624.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       1491.4231     85.097     17.526      0.000    1323.741    1659.105
    Marketing      1.4926      0.296      5.042      0.000       0.909       2.076
    ==============================================================================
    Omnibus:                       10.484   Durbin-Watson:                   2.014
    Prob(Omnibus):                  0.005   Jarque-Bera (JB):               11.169
    Skew:                           0.539   Prob(JB):                      0.00376
    Kurtosis:                       2.903   Cond. No.                         571.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Model C Summary:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Revenue   R-squared:                       0.198
    Model:                            OLS   Adj. R-squared:                  0.193
    Method:                 Least Squares   F-statistic:                     41.91
    Date:                Fri, 07 Mar 2025   Prob (F-statistic):           9.85e-10
    Time:                        13:40:07   Log-Likelihood:                -1273.3
    No. Observations:                 172   AIC:                             2551.
    Df Residuals:                     170   BIC:                             2557.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        671.1674     64.540     10.399      0.000     543.764     798.570
    Marketing      1.4743      0.228      6.474      0.000       1.025       1.924
    ==============================================================================
    Omnibus:                       10.137   Durbin-Watson:                   2.030
    Prob(Omnibus):                  0.006   Jarque-Bera (JB):               10.641
    Skew:                           0.609   Prob(JB):                      0.00489
    Kurtosis:                       3.050   Cond. No.                         601.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Model D Summary:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Revenue   R-squared:                       0.063
    Model:                            OLS   Adj. R-squared:                  0.059
    Method:                 Least Squares   F-statistic:                     16.35
    Date:                Fri, 07 Mar 2025   Prob (F-statistic):           7.07e-05
    Time:                        13:40:07   Log-Likelihood:                -1966.7
    No. Observations:                 244   AIC:                             3937.
    Df Residuals:                     242   BIC:                             3944.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       2559.6101    110.483     23.167      0.000    2341.978    2777.242
    Marketing      1.4825      0.367      4.044      0.000       0.760       2.205
    ==============================================================================
    Omnibus:                       12.536   Durbin-Watson:                   1.846
    Prob(Omnibus):                  0.002   Jarque-Bera (JB):                5.480
    Skew:                           0.050   Prob(JB):                       0.0646
    Kurtosis:                       2.273   Cond. No.                         676.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Model E Summary:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Revenue   R-squared:                       0.182
    Model:                            OLS   Adj. R-squared:                  0.179
    Method:                 Least Squares   F-statistic:                     57.48
    Date:                Fri, 07 Mar 2025   Prob (F-statistic):           6.18e-13
    Time:                        13:40:07   Log-Likelihood:                -2023.0
    No. Observations:                 260   AIC:                             4050.
    Df Residuals:                     258   BIC:                             4057.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        893.9233     72.581     12.316      0.000     750.997    1036.850
    Marketing      1.9340      0.255      7.581      0.000       1.432       2.436
    ==============================================================================
    Omnibus:                       14.807   Durbin-Watson:                   1.873
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):               16.273
    Skew:                           0.602   Prob(JB):                     0.000293
    Kurtosis:                       2.768   Cond. No.                         573.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Model F Summary:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Revenue   R-squared:                       0.073
    Model:                            OLS   Adj. R-squared:                  0.069
    Method:                 Least Squares   F-statistic:                     16.08
    Date:                Fri, 07 Mar 2025   Prob (F-statistic):           8.53e-05
    Time:                        13:40:07   Log-Likelihood:                -1650.9
    No. Observations:                 206   AIC:                             3306.
    Df Residuals:                     204   BIC:                             3313.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       1662.7274    100.402     16.561      0.000    1464.768    1860.687
    Marketing      1.4289      0.356      4.010      0.000       0.726       2.132
    ==============================================================================
    Omnibus:                       13.257   Durbin-Watson:                   1.977
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):               14.673
    Skew:                           0.650   Prob(JB):                     0.000651
    Kurtosis:                       2.852   Cond. No.                         552.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Model G Summary:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Revenue   R-squared:                       0.129
    Model:                            OLS   Adj. R-squared:                  0.125
    Method:                 Least Squares   F-statistic:                     33.02
    Date:                Fri, 07 Mar 2025   Prob (F-statistic):           2.97e-08
    Time:                        13:40:07   Log-Likelihood:                -1805.7
    No. Observations:                 225   AIC:                             3615.
    Df Residuals:                     223   BIC:                             3622.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       1283.5284     98.924     12.975      0.000    1088.583    1478.474
    Marketing      1.9847      0.345      5.747      0.000       1.304       2.665
    ==============================================================================
    Omnibus:                       10.657   Durbin-Watson:                   1.787
    Prob(Omnibus):                  0.005   Jarque-Bera (JB):               11.034
    Skew:                           0.514   Prob(JB):                      0.00402
    Kurtosis:                       2.652   Cond. No.                         572.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Model H Summary:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Revenue   R-squared:                       0.134
    Model:                            OLS   Adj. R-squared:                  0.130
    Method:                 Least Squares   F-statistic:                     36.07
    Date:                Fri, 07 Mar 2025   Prob (F-statistic):           7.22e-09
    Time:                        13:40:07   Log-Likelihood:                -1896.7
    No. Observations:                 236   AIC:                             3797.
    Df Residuals:                     234   BIC:                             3804.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       2523.4494    103.213     24.449      0.000    2320.104    2726.795
    Marketing      2.0803      0.346      6.006      0.000       1.398       2.763
    ==============================================================================
    Omnibus:                       13.254   Durbin-Watson:                   2.043
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):                7.905
    Skew:                           0.287   Prob(JB):                       0.0192
    Kurtosis:                       2.311   Cond. No.                         629.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Model I Summary:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Revenue   R-squared:                       0.078
    Model:                            OLS   Adj. R-squared:                  0.074
    Method:                 Least Squares   F-statistic:                     17.71
    Date:                Fri, 07 Mar 2025   Prob (F-statistic):           3.83e-05
    Time:                        13:40:07   Log-Likelihood:                -1600.1
    No. Observations:                 210   AIC:                             3204.
    Df Residuals:                     208   BIC:                             3211.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        898.9097     69.050     13.018      0.000     762.781    1035.038
    Marketing      0.9974      0.237      4.208      0.000       0.530       1.465
    ==============================================================================
    Omnibus:                       19.984   Durbin-Watson:                   1.946
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               23.248
    Skew:                           0.811   Prob(JB):                     8.95e-06
    Kurtosis:                       3.152   Cond. No.                         588.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python

```
