<a href="https://colab.research.google.com/github/jakecupani/meteorite-landings/blob/main/Jake_Cupani_Final_Analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Imports


```python
import pandas as pd
import numpy as np
import requests
import json

!pip install geopandas
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame


import math
import scipy.stats as stats
from math import sqrt

import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
%matplotlib inline 
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['font.size'] = 12
```

    Requirement already satisfied: geopandas in /usr/local/lib/python3.6/dist-packages (0.8.1)
    Requirement already satisfied: pandas>=0.23.0 in /usr/local/lib/python3.6/dist-packages (from geopandas) (1.1.5)
    Requirement already satisfied: fiona in /usr/local/lib/python3.6/dist-packages (from geopandas) (1.8.18)
    Requirement already satisfied: shapely in /usr/local/lib/python3.6/dist-packages (from geopandas) (1.7.1)
    Requirement already satisfied: pyproj>=2.2.0 in /usr/local/lib/python3.6/dist-packages (from geopandas) (3.0.0.post1)
    Requirement already satisfied: numpy>=1.15.4 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.23.0->geopandas) (1.18.5)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.23.0->geopandas) (2018.9)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.23.0->geopandas) (2.8.1)
    Requirement already satisfied: cligj>=0.5 in /usr/local/lib/python3.6/dist-packages (from fiona->geopandas) (0.7.1)
    Requirement already satisfied: click-plugins>=1.0 in /usr/local/lib/python3.6/dist-packages (from fiona->geopandas) (1.1.1)
    Requirement already satisfied: attrs>=17 in /usr/local/lib/python3.6/dist-packages (from fiona->geopandas) (20.3.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.6/dist-packages (from fiona->geopandas) (2020.12.5)
    Requirement already satisfied: click<8,>=4.0 in /usr/local/lib/python3.6/dist-packages (from fiona->geopandas) (7.1.2)
    Requirement already satisfied: six>=1.7 in /usr/local/lib/python3.6/dist-packages (from fiona->geopandas) (1.15.0)
    Requirement already satisfied: munch in /usr/local/lib/python3.6/dist-packages (from fiona->geopandas) (2.5.0)
    

# About the Data

https://www.kaggle.com/nasa/meteorite-landings

https://data.nasa.gov/Space-Science/Meteorite-Landings/gh4g-9sfh


# Data Collection and Cleaning


```python
# Read in the data
df = pd.read_csv("/content/drive/My Drive/INST627/meteorite-landings.csv",low_memory=False).dropna()

# Some data cleaning to make sure year values are for relevant years (a couple of rows had weird years)
df = df[(df['year'] <= 2016) & (df['year'] > 1800)]
df = df[df['mass'] > 0]
df = df[(df['reclat'] != 0) & (df['reclong'] != 0)]

drop_cols = ['name','id','recclass','fall','nametype']

# Create new column for hemisphere based on lat lon
df['hemisphere'] = df['reclat'].apply(lambda x: "Northern" if x > 0 else "Southern")

df = df.drop(drop_cols,axis=1)
df
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
      <th>mass</th>
      <th>year</th>
      <th>reclat</th>
      <th>reclong</th>
      <th>GeoLocation</th>
      <th>hemisphere</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21.0</td>
      <td>1880.0</td>
      <td>50.77500</td>
      <td>6.08333</td>
      <td>(50.775000, 6.083330)</td>
      <td>Northern</td>
    </tr>
    <tr>
      <th>1</th>
      <td>720.0</td>
      <td>1951.0</td>
      <td>56.18333</td>
      <td>10.23333</td>
      <td>(56.183330, 10.233330)</td>
      <td>Northern</td>
    </tr>
    <tr>
      <th>2</th>
      <td>107000.0</td>
      <td>1952.0</td>
      <td>54.21667</td>
      <td>-113.00000</td>
      <td>(54.216670, -113.000000)</td>
      <td>Northern</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1914.0</td>
      <td>1976.0</td>
      <td>16.88333</td>
      <td>-99.90000</td>
      <td>(16.883330, -99.900000)</td>
      <td>Northern</td>
    </tr>
    <tr>
      <th>4</th>
      <td>780.0</td>
      <td>1902.0</td>
      <td>-33.16667</td>
      <td>-64.95000</td>
      <td>(-33.166670, -64.950000)</td>
      <td>Southern</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45711</th>
      <td>172.0</td>
      <td>1990.0</td>
      <td>29.03700</td>
      <td>17.01850</td>
      <td>(29.037000, 17.018500)</td>
      <td>Northern</td>
    </tr>
    <tr>
      <th>45712</th>
      <td>46.0</td>
      <td>1999.0</td>
      <td>13.78333</td>
      <td>8.96667</td>
      <td>(13.783330, 8.966670)</td>
      <td>Northern</td>
    </tr>
    <tr>
      <th>45713</th>
      <td>3.3</td>
      <td>1939.0</td>
      <td>49.25000</td>
      <td>17.66667</td>
      <td>(49.250000, 17.666670)</td>
      <td>Northern</td>
    </tr>
    <tr>
      <th>45714</th>
      <td>2167.0</td>
      <td>2003.0</td>
      <td>49.78917</td>
      <td>41.50460</td>
      <td>(49.789170, 41.504600)</td>
      <td>Northern</td>
    </tr>
    <tr>
      <th>45715</th>
      <td>200.0</td>
      <td>1976.0</td>
      <td>33.98333</td>
      <td>-115.68333</td>
      <td>(33.983330, -115.683330)</td>
      <td>Northern</td>
    </tr>
  </tbody>
</table>
<p>31640 rows × 6 columns</p>
</div>



# What years had the highest and lowest frequency of meteorite landings in past and present?

<b>Past: any year between 1900 and 1950 <br>
Present: any year after 1990</b>

Bar plots were used in this case to show the frequency of meteorite landings since we want to see how many times they happened over the years.


```python
# Present

df['year'][df['year'] >= 1990].value_counts().plot(kind = 'bar', title = 'Present Frequencies of Meteorite Landings by Year')

print("As we can see from the graph, 2003 had the highest frequency of meteorite landings, while 2013 had the least.")
```

    As we can see from the graph, 2003 had the highest frequency of meteorite landings, while 2013 had the least.
    


    
![png](output_7_1.png)
    



```python
present = df['year'][df['year'] >= 1990].value_counts()
present = pd.DataFrame(present).reset_index().rename(columns={"index": "year", "year": "frequency"}).sort_values(by=['year'])
present.plot(x ='year', y='frequency', kind = 'line')
plt.suptitle("Present Frequencies of Meteorite Landings (Fig. 1)")
```




    Text(0.5, 0.98, 'Present Frequencies of Meteorite Landings (Fig. 1)')




    
![png](output_8_1.png)
    



```python
df['year'][(df['year'] <= 1950) & (df['year'] >= 1900)].value_counts().plot(kind = 'bar', title = 'Bar plot of Year')

print("As we can see from the graph, 1937 had the highest frequency of meteorite landings, while 1901 had the least.")
```

    As we can see from the graph, 1937 had the highest frequency of meteorite landings, while 1901 had the least.
    


    
![png](output_9_1.png)
    



```python
past = df['year'][(df['year'] <= 1950) & (df['year'] >= 1900)].value_counts()
past = pd.DataFrame(past).reset_index().rename(columns={"index": "year", "year": "frequency"}).sort_values(by=['year'])
past.plot(x ='year', y='frequency', kind = 'line')
plt.suptitle("Past Frequencies of Meteorite Landings (Fig. 2)")
```




    Text(0.5, 0.98, 'Past Frequencies of Meteorite Landings (Fig. 2)')




    
![png](output_10_1.png)
    


# In what areas of the world were meteorite landings most common? Does the distribution change over the years?

By using a scatter plot on top of a world map, it allows us to easily see where a majority of the meteorite landings happened.


```python
def plot_meteorites(df,title):
  geometry = [Point(xy) for xy in zip(df['reclong'], df['reclat'])]
  gdf = GeoDataFrame(df, geometry=geometry)   

  world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
  gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=3)

  plt.suptitle(title)
  # Source: https://stackoverflow.com/questions/53233228/plot-latitude-longitude-from-csv-in-python-3-6
```


```python
# All meteorites
plot_meteorites(df,"All Meteorites (Fig. 3)")
```


    
![png](output_13_0.png)
    



```python
# Past Meteorites

past_m = df[(df['year'] <= 1950) & (df['year'] >= 1900)]
plot_meteorites(past_m, "Past Meteorites (Fig. 3)")
```


    
![png](output_14_0.png)
    



```python
# Present Meteorites

present_m = df[df['year'] >= 1990]
plot_meteorites(present_m, "Present Meteorites (Fig. 4)")
```


    
![png](output_15_0.png)
    



```python
# 90s Meteorites
df_90 = df[(df['year'] >= 1990) & (df['year'] <= 1999)]
print("90s Meteorites")
plot_meteorites(df_90, "90s Meteorites")
```

    90s Meteorites
    


    
![png](output_16_1.png)
    



```python
# 00s Meteorites
df_00 = df[(df['year'] >= 2000) & (df['year'] <= 2009)]
print("00s Meteorites")
plot_meteorites(df_00,"2000s Meteorites")
```

    00s Meteorites
    


    
![png](output_17_1.png)
    



```python
# 10s Meteorites
df_10 = df[(df['year'] >= 2010) & (df['year'] <= 2019)]
print("10s Meteorites")
plot_meteorites(df_10,"2010s Meteorites")
```

    10s Meteorites
    


    
![png](output_18_1.png)
    


As we can see, there's a general trend of meteorites landing in the western US, Peru, Northern Africa, Southern Australia, and parts of Europe. Another trend that I noticed is that there seem to be less meteorites in the 2010s than in the other decades.

# What is the distribution of masses amongst all of the meteorites?

> Indented block




```python
df['mass'][df['mass'] < 250].plot(kind='box')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f483d5efeb8>




    
![png](output_21_1.png)
    



```python
print("As we can see in the distribution plot, the distribution is highly skewed to the right with a majority of the values below 500 grams.")
print("We can also say that the distribution is unimodal.")
print("There are",len(df[df['mass'] > 250]),"clear outliers that I will remove from my histogram to show the general distribution.")
print()

# sns.displot(df[df['mass'] < 2000], x="mass")
sns.displot(df[df['mass'] < 250], x="mass")

plt.title("Distribution of Meteorite Masses (Fig. 5)")

```

    As we can see in the distribution plot, the distribution is highly skewed to the right with a majority of the values below 500 grams.
    We can also say that the distribution is unimodal.
    There are 7182 clear outliers that I will remove from my histogram to show the general distribution.
    
    




    Text(0.5, 1.0, 'Distribution of Meteorite Masses (Fig. 5)')




    
![png](output_22_2.png)
    



```python
print("Looks like the more 'normal' mass meteorites are distributed evenly across the world.")
print()

mass_df = df[df['mass'] < 250]

plot_meteorites(mass_df,"Normal Mass Meteorites (<250g) (Fig. 6)")
```

    Looks like the more 'normal' mass meteorites are distributed evenly across the world.
    
    


    
![png](output_23_1.png)
    



```python
print("There seem to be more large mass meteorites in Southern Africa, the US, and parts of South America.")
print()

mass_df = df[df['mass'] > 2000]

plot_meteorites(mass_df,"Large Mass Meteorites (>2000g) (Fig. 7)")
```

    There seem to be more large mass meteorites in Southern Africa, the US, and parts of South America.
    
    


    
![png](output_24_1.png)
    


# What differences are there in the masses of Northern vs. Southern Hemisphere meteorites?

The bar chart shows the average mass of meteorites for the northern and southern hemispheres respectively.


```python
print("Here we show the average mass of Northern vs Southern Hemisphere meteorites.")
print("It is interesting to see that Northern Hemisphere meteorites have a higher average mass.")
print()

hemisphere_bar = sns.barplot(x="hemisphere", y="mass", data=df,ci=None)
plt.title("Northern vs. Southern Hemisphere Meteorite Masses (Fig. 8)")
```

    Here we show the average mass of Northern vs Southern Hemisphere meteorites.
    It is interesting to see that Northern Hemisphere meteorites have a higher average mass.
    
    




    Text(0.5, 1.0, 'Northern vs. Southern Hemisphere Meteorite Masses (Fig. 8)')




    
![png](output_26_2.png)
    



```python
# Is there any statistically significant difference in the masses of Northern vs Southern Hemisphere meteorites?

# H0: There is no difference between the masses of meteorites in the hemispheres. (avg difference 0).
# HA: There is a difference between the masses of meteorites in the hemispheres. (avg difference != 0).

northern = df[df['hemisphere'] == 'Northern']
southern = df[df['hemisphere'] == 'Southern']

north_mass = northern['mass']
south_mass = southern['mass']

# Point Estimate
mass_pe = north_mass.mean() - south_mass.mean()
print("The point estimate is:",mass_pe)

# Degree of Freedom
deg_freedom = len(north_mass) if len(north_mass) < len(south_mass) else len(south_mass)
deg_freedom = deg_freedom - 1
print("The degrees of freedom is:",deg_freedom)

# Standard Error
north_se = north_mass.std()**2/len(north_mass)
south_se = south_mass.std()**2/len(south_mass)

mass_se = sqrt(north_se + south_se)
print("The standard error is:",mass_se)

# 5. Calculate the t score and the p-value.
mass_t = (mass_pe - 0)/mass_se
print("The t score is:",mass_t)

mass_p = stats.distributions.t.sf(mass_t,deg_freedom)
print("The p value is:",mass_p)

print("Therefore, since the p-value is less than an alpha of 0.05, we can reject the null hypothesis and conclude that there is a significant difference in the masses of meteorites in the Northern and Southern Hemisphere.")
```

    The point estimate is: 32648.79079167894
    The degrees of freedom is: 8258
    The standard error is: 10687.521940131997
    The t score is: 3.054851346698214
    The p value is: 0.001129450442007376
    Therefore, since the p-value is less than an alpha of 0.05, we can reject the null hypothesis and conclude that there is a significant difference in the masses of meteorites in the Northern and Southern Hemisphere.
    

# Key Points

As we can see from the analysis done above, there are some noteable points that we have learned from this dataset.



*   1998 had the highest frequency of meteorite landings, while 2013 had the least in the last three decades.
*   1937 had the highest frequency of meteorite landings, while 1901 had the least in the early 1900s.
*   The distribution of masses is unimodal and highly skewed to the right with a majority of the values below 250 grams.
*   Meteorites are mainly found along the 25 and -25 degree latitude lines.
*   There's a general trend of meteorites landing in the western US, Southern America, Norther Africa, Southern Austrailia, and parts of Europe. Another trend that I noticed is that there seem to be less meteorites in the 2010s than in the other decades.




```python

```
