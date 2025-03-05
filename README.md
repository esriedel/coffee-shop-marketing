# coffee-shop-marketing

## Introduction

This analysis is based on a Kaggle dataset of 2,000 coffee shops with data on the coffee shop itself, its daily operations, and revenue. The analysis focuses on two questions:
1. Are there meaningful types of coffee shops?
2. Does these groups benefit differently from marketing?

### Variables
* Customers: Number of Customers Per Day 

* Order: Average Order Value ($) 

* Hours: Operating Hours Per Day

* Employees: Number of Employees

* Marketing: Marketing Spend Per Day ($)

* Traffic: Location Foot Traffic (people/hour)

* Revenue: Daily Revenue ($)

## Description of Data

A quick review of the variables showed no missing variables and each variable was either a float or integer. The tables and graphs below show the distribution properties of each variable. Only Revenue approximates a normal distribution with the other variables having a roughly uniform distrbution. 

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


    
![png](output_8_0.png)
    

## Correlations between Variables

Taking into account the distribution and differences in scales between the variables, Spearman rank correlations are used to examine the associations between variables. The heatmap below shows the level of correlation with most pairs having low levels of correlation (blue and dark blue). Not unexpectedly, the strongest relationshipss are between revenue, the number of customers, and the average order per customer. A modest relationship does exist with the amount of daily marketing spend and daily revenue, however, as well. A further review of scatter plots between variables shows that the relationships that do exist appear to be linear. 


### Spearman Rank Correlations Among Variables

![png](output_10_1.png)
    

### Scatter Plots Between Variables 

![png](output_11_0.png)
    

## Cluster Analysis of Data

A k-means cluster analysis is used to see if there are distinct groups of coffee shops based on qualities about the shop and its operations. To avoid the disproportionate influence of one variable over another due to scale, a standard scaler is applied to the following variables used in the cluster analysis:
* Customers
* Order
* Hours
* Employees
* Traffic



```python
shop.rename(columns={'Customers': 'Customers_scaled', 'Order': 'Order_scaled', 'Hours': 'Hours_scaled', 'Employees': 'Employees_scaled', 'Traffic':'Traffic_scaled'}, inplace=True)

```


```python
wcss = [] # Within-cluster sum of squares
for i in range(1, 11): # Test k from 1 to 10
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(shop)
        wcss.append(kmeans.inertia_)
    
# Plot the elbow graph
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

result = pd.concat([coffee, shop], axis=1)
result.describe()

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
# Initialize and fit the K-means model
kmeans = KMeans(n_clusters=9, random_state=42, n_init='auto') 
kmeans.fit(shop)

# Get cluster assignments for each data point
clusters = kmeans.labels_

# Add cluster assignments to the original dataframe
result['cluster'] = clusters

```


```python
result.describe()
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
result.info()
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
sns.boxplot(x="cluster", y="Hours", data=result)

mean_value = result['Hours'].mean()

plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

plt.legend()

plt.show()

```


    
![png](output_20_0.png)
    



```python
sns.boxplot(x="cluster", y="Employees", data=result)

mean_value = result['Employees'].mean()

plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

plt.legend()

plt.show()


```


    
![png](output_21_0.png)
    



```python
sns.boxplot(x="cluster", y="Customers", data=result)

mean_value = result['Customers'].mean()

# Add a horizontal line at the mean value
plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

# Optionally, add a legend
plt.legend()

# Show the plot
plt.show()


```


    
![png](output_22_0.png)
    



```python
sns.boxplot(x="cluster", y="Order", data=result)

mean_value = result['Order'].mean()

# Add a horizontal line at the mean value
plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

# Optionally, add a legend
plt.legend()

# Show the plot
plt.show()

```


    
![png](output_23_0.png)
    



```python
sns.boxplot(x="cluster", y="Traffic", data=result)

mean_value = result['Traffic'].mean()

# Add a horizontal line at the mean value
plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

# Optionally, add a legend
plt.legend()

# Show the plot
plt.show()

```


    
![png](output_24_0.png)
    



```python
means_pivot = result.pivot_table(values=['Customers', 'Order', 'Hours', 'Employees', 'Traffic'], index='cluster', aggfunc='mean')
print("\nMeans by Category (using pivot_table()):\n", means_pivot)
```

    
    Means by Category (using pivot_table()):
               Customers  Employees      Hours     Order     Traffic
    cluster                                                        
    0        172.798165   5.472477  13.857798  8.366651  361.160550
    1        374.362445   6.729258  13.475983  4.387293  260.628821
    2        168.790698   4.540698  13.122093  4.348721  753.093023
    3        365.627049  11.094262  10.172131  7.834139  324.143443
    4        205.396154  11.873077  14.342308  5.449846  626.030769
    5        393.262136   8.713592   9.043689  4.595340  752.082524
    6        191.688889   8.560000   8.497778  8.089689  754.480000
    7        392.161017   5.453390  13.737288  7.677161  734.745763
    8        175.490476   7.509524   8.295238  4.946238  294.723810



```python
# Cluster 0: Few employees, high hours, low traffic (The Optimist)
# Cluster 1: Low employees, Medium hours, high traffic (The Morning Rush)
# Cluster 2: Low employees, medium hours, low traffic (The 9 to 5)
# Cluster 3: High employees, medium hours, High traffic (High touch)
# cluster 4: High employees, high hours, low traffic (Downtown base)
# cluster 5: High employees, high hours, high traffic (The boutique)
# cluster 6: Low emloyees, High hours, high traffic (Exhausted)
```


```python
counts = result['cluster'].value_counts()
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

```

    Model A Summary:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Revenue   R-squared:                       0.038
    Model:                            OLS   Adj. R-squared:                  0.034
    Method:                 Least Squares   F-statistic:                     8.636
    Date:                Wed, 05 Mar 2025   Prob (F-statistic):            0.00365
    Time:                        13:09:48   Log-Likelihood:                -1735.0
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
    Date:                Wed, 05 Mar 2025   Prob (F-statistic):           9.44e-07
    Time:                        13:09:48   Log-Likelihood:                -1806.6
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
    Date:                Wed, 05 Mar 2025   Prob (F-statistic):           9.85e-10
    Time:                        13:09:48   Log-Likelihood:                -1273.3
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
    Date:                Wed, 05 Mar 2025   Prob (F-statistic):           7.07e-05
    Time:                        13:09:48   Log-Likelihood:                -1966.7
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
    Date:                Wed, 05 Mar 2025   Prob (F-statistic):           6.18e-13
    Time:                        13:09:48   Log-Likelihood:                -2023.0
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
    Date:                Wed, 05 Mar 2025   Prob (F-statistic):           8.53e-05
    Time:                        13:09:48   Log-Likelihood:                -1650.9
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
    Date:                Wed, 05 Mar 2025   Prob (F-statistic):           2.97e-08
    Time:                        13:09:48   Log-Likelihood:                -1805.7
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
    Date:                Wed, 05 Mar 2025   Prob (F-statistic):           7.22e-09
    Time:                        13:09:48   Log-Likelihood:                -1896.7
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

