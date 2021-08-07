# step-select
![build](https://travis-ci.com/chris-santiago/steps.svg?branch=master)
[![codecov](https://codecov.io/gh/chris-santiago/steps/branch/master/graph/badge.svg?token=RIB2YFGWFX)](https://codecov.io/gh/chris-santiago/steps)

A SciKit-Learn style feature selector using best subsets and stepwise regression.

## Install

Create a virtual environment with Python 3.8 and install from git:

```bash
pip install git+https://github.com/chris-santiago/steps.git
```

## Use

### Preliminaries

*Note: this example requires two additional packages*: `pandas` and `statsmodels`.

In this example we'll show how the `ForwardSelector` and `SubsetSelector` classes can be used on their own or in conjuction with a Scikit-Learn `Pipeline` object.


```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import statsmodels.datasets
from statsmodels.api import OLS
from statsmodels.tools import add_constant

from steps.forward import ForwardSelector
from steps.subset import SubsetSelector
```

We'll download the `auto` dataset via `Statsmodels`; we'll use `mpg` as the endogenous variable and the remaining variables as exongenous.  We won't use `make`, as that will create several dummies and increase the number of paramters to 12+, which is too many for the `SubsetSelector` class; we'll also drop `price`.


```python
data = statsmodels.datasets.webuse('auto')
data['foreign'] = pd.Series([x == 'Foreign' for x in data['foreign']]).astype(int)
data.fillna(0, inplace=True)
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>make</th>
      <th>price</th>
      <th>mpg</th>
      <th>rep78</th>
      <th>headroom</th>
      <th>trunk</th>
      <th>weight</th>
      <th>length</th>
      <th>turn</th>
      <th>displacement</th>
      <th>gear_ratio</th>
      <th>foreign</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AMC Concord</td>
      <td>4099</td>
      <td>22</td>
      <td>3.0</td>
      <td>2.5</td>
      <td>11</td>
      <td>2930</td>
      <td>186</td>
      <td>40</td>
      <td>121</td>
      <td>3.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AMC Pacer</td>
      <td>4749</td>
      <td>17</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>11</td>
      <td>3350</td>
      <td>173</td>
      <td>40</td>
      <td>258</td>
      <td>2.53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AMC Spirit</td>
      <td>3799</td>
      <td>22</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>12</td>
      <td>2640</td>
      <td>168</td>
      <td>35</td>
      <td>121</td>
      <td>3.08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Buick Century</td>
      <td>4816</td>
      <td>20</td>
      <td>3.0</td>
      <td>4.5</td>
      <td>16</td>
      <td>3250</td>
      <td>196</td>
      <td>40</td>
      <td>196</td>
      <td>2.93</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Buick Electra</td>
      <td>7827</td>
      <td>15</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>20</td>
      <td>4080</td>
      <td>222</td>
      <td>43</td>
      <td>350</td>
      <td>2.41</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = data.iloc[:, 3:]
y = data['mpg']
```

### Forward Stepwise Selection

The `ForwardSelector` follows the standard stepwise regression algorithm: begin with a null model, iteratively test each variable and select the one that gives the most statistically significant improvement of the fit, and repeat. This greedy algorithm continues until the fit no longer improves.

The `ForwardSelector` is instantiated with two parameters: `normalize` and `metric`. `Normalize` defaults to `False`, assuming that this class is part of a larger pipeline; `metric` defaults to AIC.

|Parameter|Type|Description|
|---------|----|-----------|
|normalize|bool|Whether to normalize features; default `False`|
|metric|str|Optimization metric to use; must be one of `aic` or `bic`; default `aic`|

The `ForwardSelector` class follows the Scikit-Learn API.  After fitting the selector using the `.fit()` method, the selected features can be accessed using the boolean mask under the `.best_support_` attribute.


```python
selector = ForwardSelector(normalize=True, metric='aic')
selector.fit(X, y)
```




    ForwardSelector(normalize=True)




```python
X.loc[:, selector.best_support_]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rep78</th>
      <th>weight</th>
      <th>length</th>
      <th>gear_ratio</th>
      <th>foreign</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>2930</td>
      <td>186</td>
      <td>3.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>3350</td>
      <td>173</td>
      <td>2.53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>2640</td>
      <td>168</td>
      <td>3.08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>3250</td>
      <td>196</td>
      <td>2.93</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>4080</td>
      <td>222</td>
      <td>2.41</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>4.0</td>
      <td>2160</td>
      <td>172</td>
      <td>3.74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>70</th>
      <td>5.0</td>
      <td>2040</td>
      <td>155</td>
      <td>3.78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>4.0</td>
      <td>1930</td>
      <td>155</td>
      <td>3.78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>72</th>
      <td>4.0</td>
      <td>1990</td>
      <td>156</td>
      <td>3.78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>73</th>
      <td>5.0</td>
      <td>3170</td>
      <td>193</td>
      <td>2.98</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 5 columns</p>
</div>



### Best Subset Selection

The `SubsetSelector` follows a very simple algorithm: compare all possible models with $k$ predictors, and select the model that minimizes our selection criteria. This algorithm is only appropriate for $k<=12$ features, as it becomes computationally expensive: there are $\frac{k!}{(p-k)!}$possible models, where $p$ is the total number of paramters and $k$ is the number of features included in the model.

The `SubsetSelector` is instantiated with two parameters: `normalize` and `metric`. `Normalize` defaults to `False`, assuming that this class is part of a larger pipeline; `metric` defaults to AIC.

|Parameter|Type|Description|
|---------|----|-----------|
|normalize|bool|Whether to normalize features; default `False`|
|metric|str|Optimization metric to use; must be one of `aic` or `bic`; default `aic`|

The `SubsetSelector` class follows the Scikit-Learn API.  After fitting the selector using the `.fit()` method, the selected features can be accessed using the boolean mask under the `.best_support_` attribute.


```python
selector = SubsetSelector(normalize=True, metric='aic')
selector.fit(X, y)
```




    SubsetSelector(normalize=True)




```python
X.loc[:, selector.get_support()]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rep78</th>
      <th>weight</th>
      <th>length</th>
      <th>gear_ratio</th>
      <th>foreign</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>2930</td>
      <td>186</td>
      <td>3.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>3350</td>
      <td>173</td>
      <td>2.53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>2640</td>
      <td>168</td>
      <td>3.08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>3250</td>
      <td>196</td>
      <td>2.93</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>4080</td>
      <td>222</td>
      <td>2.41</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>4.0</td>
      <td>2160</td>
      <td>172</td>
      <td>3.74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>70</th>
      <td>5.0</td>
      <td>2040</td>
      <td>155</td>
      <td>3.78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>4.0</td>
      <td>1930</td>
      <td>155</td>
      <td>3.78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>72</th>
      <td>4.0</td>
      <td>1990</td>
      <td>156</td>
      <td>3.78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>73</th>
      <td>5.0</td>
      <td>3170</td>
      <td>193</td>
      <td>2.98</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 5 columns</p>
</div>



### Comparing the full model 

Using the `SubsetSelector` selected features yields a model with 4 fewer parameters and slightly improved AIC and BIC metrics. The summaries indicate possible multicollinearity in both models, likely caused by `weight`, `length`, `displacement` and other features that are all related to the weight of a vehicle. 

*Note: Selection using BIC as the optimization metric yields a model where `weight` is the only selected feature. Bayesian information criteria penalizes additional parameters more then AIC.*


```python
mod = OLS(endog=y, exog=add_constant(X)).fit()
mod.summary()
```





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.720</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.681</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   18.33</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 07 Aug 2021</td> <th>  Prob (F-statistic):</th> <td>1.29e-14</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:37:36</td>     <th>  Log-Likelihood:    </th> <td> -187.23</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    74</td>      <th>  AIC:               </th> <td>   394.5</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    64</td>      <th>  BIC:               </th> <td>   417.5</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     9</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>        <td>   39.0871</td> <td>    9.100</td> <td>    4.295</td> <td> 0.000</td> <td>   20.907</td> <td>   57.267</td>
</tr>
<tr>
  <th>rep78</th>        <td>    1.0021</td> <td>    0.357</td> <td>    2.809</td> <td> 0.007</td> <td>    0.290</td> <td>    1.715</td>
</tr>
<tr>
  <th>headroom</th>     <td>   -0.0167</td> <td>    0.611</td> <td>   -0.027</td> <td> 0.978</td> <td>   -1.237</td> <td>    1.204</td>
</tr>
<tr>
  <th>trunk</th>        <td>   -0.0772</td> <td>    0.154</td> <td>   -0.503</td> <td> 0.617</td> <td>   -0.384</td> <td>    0.230</td>
</tr>
<tr>
  <th>weight</th>       <td>   -0.0037</td> <td>    0.002</td> <td>   -1.928</td> <td> 0.058</td> <td>   -0.008</td> <td>    0.000</td>
</tr>
<tr>
  <th>length</th>       <td>   -0.0752</td> <td>    0.061</td> <td>   -1.229</td> <td> 0.223</td> <td>   -0.197</td> <td>    0.047</td>
</tr>
<tr>
  <th>turn</th>         <td>   -0.1762</td> <td>    0.187</td> <td>   -0.941</td> <td> 0.350</td> <td>   -0.550</td> <td>    0.198</td>
</tr>
<tr>
  <th>displacement</th> <td>    0.0131</td> <td>    0.011</td> <td>    1.180</td> <td> 0.243</td> <td>   -0.009</td> <td>    0.035</td>
</tr>
<tr>
  <th>gear_ratio</th>   <td>    3.7067</td> <td>    1.751</td> <td>    2.116</td> <td> 0.038</td> <td>    0.208</td> <td>    7.206</td>
</tr>
<tr>
  <th>foreign</th>      <td>   -4.4633</td> <td>    1.385</td> <td>   -3.222</td> <td> 0.002</td> <td>   -7.230</td> <td>   -1.696</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>28.364</td> <th>  Durbin-Watson:     </th> <td>   2.523</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  52.945</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.389</td> <th>  Prob(JB):          </th> <td>3.18e-12</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.074</td> <th>  Cond. No.          </th> <td>7.55e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 7.55e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
mod = OLS(endog=y, exog=add_constant(X.loc[:, selector.best_support_])).fit()
mod.summary()
```





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.710</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.688</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   33.25</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 07 Aug 2021</td> <th>  Prob (F-statistic):</th> <td>5.22e-17</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:37:40</td>     <th>  Log-Likelihood:    </th> <td> -188.63</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    74</td>      <th>  AIC:               </th> <td>   389.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    68</td>      <th>  BIC:               </th> <td>   403.1</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td>   40.3703</td> <td>    7.860</td> <td>    5.136</td> <td> 0.000</td> <td>   24.687</td> <td>   56.054</td>
</tr>
<tr>
  <th>rep78</th>      <td>    0.9040</td> <td>    0.342</td> <td>    2.647</td> <td> 0.010</td> <td>    0.223</td> <td>    1.586</td>
</tr>
<tr>
  <th>weight</th>     <td>   -0.0030</td> <td>    0.002</td> <td>   -1.770</td> <td> 0.081</td> <td>   -0.006</td> <td>    0.000</td>
</tr>
<tr>
  <th>length</th>     <td>   -0.1058</td> <td>    0.053</td> <td>   -1.990</td> <td> 0.051</td> <td>   -0.212</td> <td>    0.000</td>
</tr>
<tr>
  <th>gear_ratio</th> <td>    2.6905</td> <td>    1.511</td> <td>    1.780</td> <td> 0.079</td> <td>   -0.325</td> <td>    5.706</td>
</tr>
<tr>
  <th>foreign</th>    <td>   -4.0123</td> <td>    1.320</td> <td>   -3.040</td> <td> 0.003</td> <td>   -6.646</td> <td>   -1.379</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>24.257</td> <th>  Durbin-Watson:     </th> <td>   2.442</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  39.774</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.252</td> <th>  Prob(JB):          </th> <td>2.31e-09</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.576</td> <th>  Cond. No.          </th> <td>6.59e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 6.59e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### Use in Scikit-Learn Pipeline

Both `ForwardSelector` and `SubsetSelector` objects are compatible with Scikit-Learn `Pipeline` objects, and can be used as feature selection steps:


```python
pl = Pipeline([
    ('feature_selection', SubsetSelector(normalize=True)),
    ('regression', LinearRegression())
])
pl.fit(X, y)
```




    Pipeline(steps=[('feature_selection', SubsetSelector(normalize=True)),
                    ('regression', LinearRegression())])




```python
pl.score(X, y)
```




    0.7097132531085899


