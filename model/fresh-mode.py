
# Try tsfresh feature extraction on mode detection data
import os
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import make_scorer

## consts :
dataSource = r'../fresh-data'
SAMPLE_FREQ = 50 
FILE_MARGINES = 2* SAMPLE_FREQ  ## number of samples to ignore in the  start and in the end of the file (5 seconds )  
WINDOW_SIZE = 2 * 128  ## sliding window size 
PEAKS_WINDOW_SIZE = 5*WINDOW_SIZE  ## sliding window size for peaks count feature

DEVICE_MODE_LABELS = ['pocket','swing','texting','talking','whatever']

data_limit = 10000


def loadFile(root,file):
    data=pd.read_csv(os.path.join(root,file))
    
    print('loading : ' , file) 

    print('loading : ' , len(data) , ' samples from ', file) 
    
    ## usefull property : 
    data['source']=file  

    ## default label values in case file name not contains label  
    data['devicemodeDescription']=DEVICE_MODE_LABELS[-1] ## 'whatever' label 
    data['devicemode'] = len(DEVICE_MODE_LABELS)

    ## search device mode label in file name and add as new properties :
    for label in DEVICE_MODE_LABELS:
        if label.lower() in file.lower():  
            data['devicemodeDescription']=label         ## label name 
            data['devicemode'] = DEVICE_MODE_LABELS.index(label)    ## label index 
            break
           
    ## crop samples from start and from the end of the file :
    margin = min(len(data) / 2 - 1 , FILE_MARGINES)
    data.drop(data.index[range(0,margin)],axis=0,inplace=True)
    data.drop(data.index[range(-margin,-1)],axis=0,inplace=True)   
    ##  print(len(data) , ' samples after cropping ' , margin , 'samples from start-end of the file  ')
    return data 

def loadFiles(inputDir):
    print ('loading files from : ' , inputDir )
    return pd.concat([loadFile(inputDir,f) for f in os.listdir(inputDir) if f.lower().endswith('.csv')])  

rdf = loadFiles(dataSource)
print('=========================================================')
print( 'total train samples ' , len(rdf) , ' from ' ,len(rdf.source.unique()),  ' files ')

rdf.drop(u'Unnamed: 11',inplace=True,axis=1)
rdf.drop(u'Unnamed: 12',inplace=True,axis=1)
# rdf.drop(u'Unnamed: 17',inplace=True,axis=1)

rdf.groupby('devicemodeDescription').devicemode.count()

# add id for each data window
windowsize = 128
rdf['id'] = [int(i/windowsize) for i in range(0,len(rdf))] # rdf['source']

rdf.columns

timeseries = rdf[['id','gFy', 'gFz', 'gfx','time', 'wx', 'wy', 'wz']].copy()

# calc norm for each vector
timeseries['g-norm'] = np.sqrt(timeseries['gfx']**2 + timeseries['gFy']**2 + timeseries['gFz']**2)
timeseries['w-norm'] = np.sqrt(timeseries['wx']**2 + timeseries['wy']**2 + timeseries['wz']**2)

y = rdf.groupby('id')['devicemode'].agg(np.mean)
# fill nan values
timeseries.fillna(0,inplace=True)

#y = rdf.groupby('source')['devicemode'].agg(np.mean)


extracted_features = extract_features(timeseries, column_id='id', column_sort="time")
print ('{} features extracted '.format(len(extracted_features)))

impute(extracted_features)

features_filtered = select_features(extracted_features, y)

k= 4 ## len(rdf.source.unique())
x_train= extracted_features
y_train = y
forest = RandomForestClassifier()
# knn5 = KNeighborsClassifier(n_neighbors=5)

def sourceFold(): 
    print ('list of source files for each kfold : ' )    
    _kfold = GroupKFold(n_splits=k) 
    _itr = _kfold.split(x_train, y_train, groups=rdf.source)  
    c = 0 
    sf = {}
    for i in _itr:
        ## print (c , str(rdf.iloc[i[1]].source.unique()))
        sf[c]= str(rdf.iloc[i[1]].source.unique())
        c = c+1
    return sf 


def CalcKFoldAccuracy(classifier,X,Y,k):
    group_kfold = GroupKFold(n_splits=k)     
    groups_itr = group_kfold.split(X, Y, groups=rdf.source)    
    return cross_val_score(classifier, X, Y, cv=groups_itr, scoring='accuracy')


print ('RF : ')
print (CalcKFoldAccuracy(forest,x_train,y_train,k))
