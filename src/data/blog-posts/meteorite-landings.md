---
title:  Global Meteorite Landings Statistical Analysis ☄️
publishDate: 
description: Statistical analysis of meteorite landings dataset to identify the frequency of meteorite landings throughout different time periods, most common geographic regions of meteorites, and the differences between meteorite landings in the norhtern and southern hemispheres respectively.
tags: ['Data Visualization','Machine Learning', 'NASA']
---

# Background

The data is available on Kaggle and NASA's own data portal:

[Kaggle Data](https://www.kaggle.com/nasa/meteorite-landings) or
[NASA Data Portal](https://data.nasa.gov/Space-Science/Meteorite-Landings/gh4g-9sfh)

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

# Data Collection and Cleaning
```python
# Read in the data
df = pd.read_csv("<FILEPATH>",low_memory=False).dropna()

# Some data cleaning to make sure year values are
# for relevant years (a couple of rows had weird years)
df = df[(df['year'] <= 2016) & (df['year'] > 1800)]
df = df[df['mass'] > 0]
df = df[(df['reclat'] != 0) & (df['reclong'] != 0)]

drop_cols = ['name','id','recclass','fall','nametype']

# Create new column for hemisphere based on lat lon
df['hemisphere'] = df['reclat']
.apply(lambda x: "Northern" if x > 0 else "Southern")

df = df.drop(drop_cols,axis=1)
```

--- 
Graph
---

# Data Exploration

--- 
Graph
---

## What years had the highest and lowest frequency of meteorite landings in the past and present?
**Past: any year between 1900 and 1950**
**Present: any year after 1990**

Bar plots were used in this case to show the frequency of meteorite landings since we want to see how many times they happened over the years.

```python
df['year'][df['year'] >= 1990].value_counts().plot(kind = 'bar', 
title = 'Present Frequencies of Meteorite Landings by Year')
```

--- 
Graph
---
As we can see from the graph, 2003 had the highest frequency of meteorite landings, while 2013 had the least.