from modules import DataPrep as setup
import numpy as np

print('Creating Data',end='\r')
setup.PrepData()

print('Loading Created Data\n',end='\r')
imgs = np.load('./../data/cherrypicked/TRAIN/train_images.npy')
labels = np.load('./../data/cherrypicked/TRAIN/train_labels.npy', allow_pickle=True)
labels_int = LabelEncoder().fit_transform(labels)

print('Removing Outliers: Pass 1\n')
imgs, labels_int = setup.RemoveOutliers(X=imgs,Y=labels_int)

print('Removing Outliers: Pass 2\n')
imgs, labels_int = setup.RemoveOutliers(X=cleanImgs,Y=cleanLabels,threshold=91)

print('Removing Outliers: Pass 3\n')
imgs, labels_int = setup.RemoveOutliers(X=cleanImgs,Y=cleanLabels,threshold=81)

print('Saving Cleaned Data in "data/cherrypicked/TRAIN"',end='\r')
np.save('./../data/cherrypicked/TRAIN/train_images_clean',imgs)
np.save('./../data/cherrypicked/TRAIN/train_labels_clean',labels_int)

print('Complete')
print('"data/train_tar" can be deleted to save space')