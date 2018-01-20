
# Try tsfresh feature extraction on mode detection data
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tsfresh.transformers import RelevantFeatureAugmenter

## consts :
dataSource = r'../fresh-data'
testSource = r'../raw-data/validation/utf8'
SAMPLE_FREQ = 50 
FILE_MARGINES = 2* SAMPLE_FREQ  ## number of samples to ignore in the  start and in the end of the file (5 seconds )  
WINDOW_SIZE = 64 ## sliding window size


DEVICE_MODE_LABELS = ['pocket','swing','texting','talking','whatever']

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

def loadSensorData(inputDir):
    rdf = loadFiles(inputDir)
    print('=========================================================')
    print( 'total train samples ' , len(rdf) , ' from ' ,len(rdf.source.unique()),  ' files ')

    rdf.drop(u'Unnamed: 11',inplace=True,axis=1)
    rdf.drop(u'Unnamed: 12',inplace=True,axis=1)
    # rdf.drop(u'Unnamed: 17',inplace=True,axis=1)

    rdf.groupby('devicemodeDescription').devicemode.count()

    # add id for each data window
    rdf['id'] = [int(i/WINDOW_SIZE) for i in range(0,len(rdf))] # rdf['source']

    rdf.columns

    timeseries = rdf[['id','gFy', 'gFz', 'gfx','time', 'wx', 'wy', 'wz']].copy()

    # calc norm for each vector
    timeseries['g-norm'] = np.sqrt(timeseries['gfx']**2 + timeseries['gFy']**2 + timeseries['gFz']**2)
    timeseries['w-norm'] = np.sqrt(timeseries['wx']**2 + timeseries['wy']**2 + timeseries['wz']**2)

    y = rdf.groupby('id')['devicemode'].agg(np.mean)> 1
    # fill nan values
    return timeseries.fillna(0,inplace=True)



pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
            ('classifier', RandomForestClassifier())])

df_ts, y = loadSensorData(dataSource)
X = pd.DataFrame(index=y.index)

pipeline.set_params(augmenter__timeseries_container=df_ts)
pipeline.fit(X, y)
pred = pipeline.predict(X)
test = loadSensorData(testSource)
pred.head()