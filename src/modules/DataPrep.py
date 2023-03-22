import tarfile
import pandas as pd
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

def PrepData(path='./../'):
    #LABELS
    df = pd.read_csv(path+'data/raw/DF20-train_metadata_PROD-2.csv')
    small_df = df[['image_path','species','Substrate']]
    small_df = small_df[df.species.isin(df.species.value_counts().head(30).index.tolist())]
    small_df.image_path = small_df.image_path.str.lower()
    
    # balance classes
    class_counts = Counter(small_df['species'])
    under_sampler = RandomUnderSampler(sampling_strategy='all')
    X_under, y_under = under_sampler.fit_resample(small_df.drop('species', axis=1), small_df['species'])
    small_df = pd.concat([X_under, y_under], axis=1)
    
    
    # IMAGES
    print('Extracting...', end='\r')
    with tarfile.open(path+'data/raw/DF20-300px.tar.gz', 'r:gz') as tar:
        tar.extractall(path+'data/train_tar')
    print('Finished', end='\r')
        
    image_array = np.zeros((len(small_df), 300, 300, 3), dtype=np.uint8)

    for i, image_path in enumerate(small_df.image_path.values.tolist()):
        with Image.open(path+'data/train_tar/DF20_300/'+image_path) as img:
            img = img.resize((300, 300))
            img = np.array(img)
            image_array[i]=img
            print(f'Processed image #{i+1}', end='\r')
    
    
    # Split and save
    y = small_df['species']

    X_train, X_test, y_train, y_test = train_test_split(image_array, y, test_size=0.1, stratify=y)
    
    np.save(path+'data/cherrypicked/TRAIN/train_images',X_train)
    np.save(path+'data/cherrypicked/TEST_DO_NOT_TOUCH/test_images',X_test)
    
    np.save(path+'data/cherrypicked/TRAIN/train_labels',y_train)
    np.save(path+'data/cherrypicked/TEST_DO_NOT_TOUCH/test_labels',y_test)
    
    print('Data saved in "data/cherrypicked"')

def TrimFat(df):
    drop_cols = ['gbifID', 'eventDate','taxonID','specificEpithet', 'infraspecificEpithet','kingdom', 'phylum', 'class',
       'order', 'family', 'genus','scientificName','taxonRank','level0Gid', 'level0Name', 'level1Gid',
       'level1Name', 'level2Gid', 'level2Name','ImageUniqueID','Latitude', 'Longitude','class_id','image_path','CoorUncert']
    df = df.drop(drop_cols,axis=1)
    df2 = df[df.species.isin(df.species.value_counts().head(30).index.tolist())]
    class_counts = Counter(df2['species'])
    under_sampler = RandomUnderSampler(sampling_strategy='all')
    X_under, y_under = under_sampler.fit_resample(df2.drop('species', axis=1), df2['species'])
    df2 = pd.concat([X_under, y_under], axis=1)
    
    return df, df2

def RemoveOutliers(X,Y,threshold=101):
    classes = len(np.unique(Y))
    cleanedImgs = []
    cleanedLabels = []
    removed = 0
    print('loaded data. Cleaning by class.',end='\r')
    
    for class_num in range(classes):
        selectedIndexes = []
        avg_r = np.average(X[Y==class_num][:,:,:,0])
        avg_g = np.average(X[Y==class_num][:,:,:,1])
        avg_b = np.average(X[Y==class_num][:,:,:,2])
        avg_row = np.array([avg_r,avg_g,avg_b])
        
        for i, img in enumerate(X[Y==class_num]):
            r = np.average(img[:,:,0])
            g = np.average(img[:,:,1])
            b = np.average(img[:,:,2])
            row = np.array([r,g,b])
            distance = np.linalg.norm(avg_row - row)
            if distance < threshold:
                selectedIndexes.append(i)
            else:
                removed+=1
                
        cleanedImgs.append(X[Y==class_num][selectedIndexes])
        cleanedLabels.append(Y[Y==class_num][selectedIndexes])
        print(f'Class {class_num} finished!\n',end='\r')
    print(f'finished all cleaning. Concatenating data.\nThis may take a minute or two.\n{removed} images have been removed.')
    cleanedImgs = np.concatenate(cleanedImgs,axis=0)
    cleanedLabels = np.concatenate(cleanedLabels,axis=0)
    
    return cleanedImgs, cleanedLabels
        
    
if __name__ == '__main__':
    PrepData(path='./../../')