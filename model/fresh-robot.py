import os
import pandas as pd
import numpy as np

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
#get_ipython().magic('matplotlib inline')
import seaborn as sns

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures,     load_robot_execution_failures
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()

# The first column is the DataFrame index and has no meaning here.
# There are six different time series (a-f) for the different sensors.
# The different robots are denoted by the ids column.

print(timeseries.head())
print(y.head())

print ('{} failures out of {} '.format(len(y[y]),len(y)))

ts1 = timeseries[timeseries.id == 1]
ts20 = timeseries[timeseries.id == 20]
fig, ax = plt.subplots()
ax.plot(ts1.time,ts1.F_z, 'b',label='proper robot' )
ax.plot(ts20.time,ts20.F_z, 'g' ,label='failure robot')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

plt.show()


# You can already see some differences by eye - but for successful machine learning we have to put these differences into numbers.
#
# For this, tsfresh comes into place. It allows us to automatically extract over 1200 features from those six different time series for each robot.
#
# For extracting all features, we do:
#

from tsfresh import extract_features
extracted_features = extract_features(timeseries, column_id="id", column_sort="time")

# You end up with a DataFrame extracted_features with all more than 1200 different extracted features.
extracted_features.head()

# We will now remove all NaN values (that were created by feature calculators, than can not be used on the given data, e.g. because it has too low statistics) and select only the relevant features next:
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
impute(extracted_features)
features_filtered = select_features(extracted_features, y)


print('{} features after feature selection '.format(len(features_filtered.columns)))


# Further, you can even perform the extraction, imputing and filtering at the same time with the tsfresh.extract_relevant_features() function:

# In[18]:


from tsfresh import extract_relevant_features

features_filtered_direct = extract_relevant_features(timeseries, y,
                                                     column_id='id', column_sort='time')


# In[19]:


len(features_filtered_direct.columns)


