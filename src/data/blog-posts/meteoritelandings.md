---
title: NASA Meteorite Landings Analysis ☄️
# publishDate: Friday, June 9th, 2023
description: Statistical analysis of meteorite landings dataset to identify the frequency of meteorite landings throughout different time periods, most common geographic regions of meteorites, and the differences between meteorite landings in the norhtern and southern hemispheres respectively.
tags: ['Data Visualization 📊','Data Analytics 📈']
---
## Imports

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

## About the Data

https://www.kaggle.com/nasa/meteorite-landings

https://data.nasa.gov/Space-Science/Meteorite-Landings/gh4g-9sfh

## Data Collection and Cleaning

```python
# Read in the data
df = pd.read_csv("/content/drive/My Drive/INST627/meteorite-landings.csv",
low_memory=False).dropna()

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

## What years had the highest and lowest frequency of meteorite landings in past and present?

**Past: any year between 1900 and 1950**
**Present: any year after 1990**

Bar plots were used in this case to show the frequency of meteorite landings since we want to see how many times they happened over the years.

```python
# Present
df['year'][df['year'] >= 1990].value_counts().plot(kind = 'bar', title = 'Present Frequencies of Meteorite Landings by Year')
```

As we can see from the graph, 2003 had the highest frequency of meteorite landings, while 2013 had the least.

<img src="">

As we can see from the graph, 2003 had the highest frequency of meteorite landings, while 2013 had the least.

```python
present = df['year'][df['year'] >= 1990].value_counts()
present = pd.DataFrame(present).reset_index()
.rename(columns={"index": "year", "year": "frequency"}).sort_values(by=['year'])

present.plot(x ='year', y='frequency', kind = 'line')
plt.suptitle("Present Frequencies of Meteorite Landings (Fig. 1)")
```

<img src="">

```python
df['year'][(df['year'] <= 1950) & (df['year'] >= 1900)].value_counts()
.plot(kind='bar', title='Bar plot of Year')
```

As we can see from the graph, 1937 had the highest frequency of meteorite landings, while 1901 had the least.

<img src="">

```python
past = df['year'][(df['year'] <= 1950) & (df['year'] >= 1900)].value_counts()
past = pd.DataFrame(past).reset_index().rename(columns={"index": "year", "year": "frequency"}).sort_values(by=['year'])
past.plot(x ='year', y='frequency', kind = 'line')
plt.suptitle("Past Frequencies of Meteorite Landings (Fig. 2)")
```

<img src="">

## In what areas of the world were meteorite landings most common? Does the distribution change over the years?

```python
def plot_meteorites(df,title):
  geometry = [Point(xy) for xy in zip(df['reclong'], df['reclat'])]
  gdf = GeoDataFrame(df, geometry=geometry)   

  world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
  gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=3)

  plt.suptitle(title)

# All Meteorites
plot_meteorites(df,"All Meteorites (Fig. 3)")
```

<img src="">

```python
# Past Meteorites
past_m = df[(df['year'] <= 1950) & (df['year'] >= 1900)]
plot_meteorites(past_m, "Past Meteorites (Fig. 3)")
```

<img src="">

```python
# Present Meteorites
present_m = df[df['year'] >= 1990]
plot_meteorites(present_m, "Present Meteorites (Fig. 4)")
```

<img src="">

```python
# 90s Meteorites
df_90 = df[(df['year'] >= 1990) & (df['year'] <= 1999)]
plot_meteorites(df_90, "90s Meteorites")
```

<img src="">

```python
# 00s Meteorites
df_00 = df[(df['year'] >= 2000) & (df['year'] <= 2009)]
plot_meteorites(df_00,"2000s Meteorites")
```

<img src="">

```python
# 10s Meteorites
df_10 = df[(df['year'] >= 2010) & (df['year'] <= 2019)]
plot_meteorites(df_10,"2010s Meteorites")
```

<img src="">

As we can see, there's a general trend of meteorites landing in the western US, Peru, Northern Africa, Southern Australia, and parts of Europe. Another trend that I noticed is that there seem to be less meteorites in the 2010s than in the other decades.

## What is the distribution of masses amongst all of the meteorites?

```python
df['mass'][df['mass'] < 250].plot(kind='box')
```

<img src="">

As we can see in the distribution plot, the distribution is highly skewed to the right with a majority of the values below 500 grams.

We can also say that the distribution is unimodal. There are 7182 clear outliers that I will remove from my histogram to show the general distribution.

<img src="">

```python
mass_df = df[df['mass'] < 250]
plot_meteorites(mass_df,"Normal Mass Meteorites (<250g) (Fig. 6)")
```

Looks like the more 'normal' mass meteorites are distributed evenly across the world.

<img src="">

```python
mass_df = df[df['mass'] > 2000]
plot_meteorites(mass_df,"Large Mass Meteorites (>2000g) (Fig. 7)")
```

There seem to be more large mass meteorites in Southern Africa, the US, and parts of South America.

<img src="">

## What differences are there in the masses of Northern vs. Southern Hemisphere meteorites?

The bar chart shows the average mass of meteorites for the northern and southern hemispheres respectively.

```python
hemisphere_bar = sns.barplot(x="hemisphere", y="mass", data=df,ci=None)
plt.title("Northern vs. Southern Hemisphere Meteorite Masses (Fig. 8)")
```

Here we show the average mass of Northern vs Southern Hemisphere meteorites. It is interesting to see that Northern Hemisphere meteorites have a higher average mass.

<img src="">

## Final Analysis

> Is there any statistically significant difference in the masses of Northern vs Southern Hemisphere meteorites?

**H0: There is no difference between the masses of meteorites in the hemispheres. (avg difference 0).**

</br>

**HA: There is a difference between the masses of meteorites in the hemispheres. (avg difference != 0).**

```python
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
```

- The **point estimate** is: **32,648.79**
- The **degrees of freedom** is: **8,258**
- The **standard error** is: **10,687.52**
- The **t score** is: **3.055**
- The **p value** is: **0.001**


> Therefore, since the p-value is less than an alpha of 0.05, we can reject the null hypothesis and conclude that there is a significant difference in the masses of meteorites in the Northern and Southern Hemisphere.

</br>

## Conclusion

As we can see from the analysis done above, there are some noteable points that we have learned from this dataset.

- 1998 had the highest frequency of meteorite landings, while 2013 had the least in the last three decades.

- 1937 had the highest frequency of meteorite landings, while 1901 had the least in the early 1900s.

- The distribution of masses is unimodal and highly skewed to the right with a majority of the values below 250 grams.

- Meteorites are mainly found along the 25 and -25 degree latitude lines.

- There's a general trend of meteorites landing in the western US, Southern America, Norther Africa, Southern Austrailia, and parts of Europe. Another trend that I noticed is that there seem to be less meteorites in the 2010s than in the other decades.












