
# Try tsfresh feature extraction on mode detection data
import os
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from tsfresh.transformers import RelevantFeatureAugmenter

## consts :
dataSource = r'../fresh-data'
testSource = r'../fresh-data/validation'
SAMPLE_FREQ = 50 
FILE_MARGINES = 2* SAMPLE_FREQ  ## number of samples to ignore in the  start and in the end of the file (5 seconds )  
WINDOW_SIZE = 64 ## sliding window size
SENSOR_COMPONENTS = ['time','gfx','gFy','gFz','wx','wy','wz']

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
    rdf.groupby('devicemodeDescription').devicemode.count()

    # add id for each data window
    rdf['id'] = [int(i/WINDOW_SIZE) for i in range(0,len(rdf))]
    ts_df = rdf[['id']+SENSOR_COMPONENTS].copy()

    # calc norm for each vector
    ts_df['g-norm'] = np.sqrt(ts_df['gfx']**2 + ts_df['gFy']**2 + ts_df['gFz']**2)
    ts_df['w-norm'] = np.sqrt(ts_df['wx']**2 +  ts_df['wy']**2 +  ts_df['wz']**2)
    y = rdf.groupby('id')['devicemode'].agg(np.max)

    # fill nan values
    return ts_df.fillna(0) , y


## main :
def main():
    pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
                         ('classifier', RandomForestClassifier())])

    ts_df, y = loadSensorData(dataSource)
    X = pd.DataFrame(index=y.index)

    pipeline.set_params(augmenter__timeseries_container=ts_df)
    pipeline.fit(X, y)

    val_ts_df, val_y = loadSensorData(testSource)

    val_X = pd.DataFrame(index=val_y.index)

    pipeline.set_params(augmenter__timeseries_container=val_ts_df)

    print('score : {}'.format(pipeline.score(val_X,val_y))) # pipeline.score(val_X,val_y)

    # scores = cross_val_scores(pipeline, X, y, cv=4, scoring='f1_micro')


if __name__ == "__main__" :
    main()


