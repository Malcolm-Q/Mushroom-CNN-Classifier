# Run this script to setup the downloaded data as I have if you want to tweak my model or build your own.
# It's a little messy but gets the job done...
from modules import DataPrep as setup
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import pickle

# load a list of list of indices we'll manually remove
with open('data.pickle', 'rb') as f:
    manual_removals = pickle.load(f)

# Extract data, get the classes, get labels
print('Creating Data',end='\r')
setup.PrepData()

# Load gathered data and encode classes
print('Loading Created Data\n',end='\r')
imgs = np.load('./../data/cherrypicked/TRAIN/train_images.npy')
labels = np.load('./../data/cherrypicked/TRAIN/train_labels.npy', allow_pickle=True)
labels_int = LabelEncoder().fit_transform(labels)

# Remove outliers (images furthest away from average image until there's only 625 remaining)
# We'll also remove our selected indices now
from modules.ImgExploration import AvgPixelOutliersBalancer
num_classes = len(np.unique(labels_int))
cleaned_classes = []
for class_label in range(num_classes):
    new_imgs = AvgPixelOutliersBalancer(imgs=imgs[labels_int==class_label],balance_to=625)
    cleaned_classes.append([new_imgs[i] for i in range(len(new_imgs)) if i not in manual_removals[class_label]])
    print(f"Finished cleaning class {class_label}", end='\r')
labels_int = np.concatenate([np.repeat(i,len(cleaned_classes[i])) for i in range(num_classes)])
print(f"Concatenating {class_label}\t\t", end='\r')
imgs = np.concatenate(cleaned_classes,axis=0)
print(f"Done {class_label}\t\t\t", end='\r')
del cleaned_classes

# balance all classes to 600 with augmentation
num_classes = len(np.unique(labels_int))
target_samples_per_class = 600
augmented_data = []
for class_label in range(num_classes):
    X_class = imgs[labels_int == class_label]
    num_samples = X_class.shape[0]
    num_augmented_samples = 0
    while num_augmented_samples < target_samples_per_class:
        num_samples_to_augment = min(target_samples_per_class - num_augmented_samples, num_samples)
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(X_class[:num_samples_to_augment])
        augmented_samples = datagen.flow(X_class[:num_samples_to_augment], batch_size=num_samples_to_augment, shuffle=False)
        augmented_data.append(augmented_samples.next())
        num_augmented_samples += num_samples_to_augment
    print(f"Finished augmenting class {class_label}", end='\r')

print(f"Concatenating...\t\t", end='\r')
imgs = np.concatenate(augmented_data, axis=0)
print(f"Freeing RAM...\t\t", end='\r')
del augmented_data
print(f"Shuffling...\t\t", end='\r')
labels_int = np.repeat(np.arange(num_classes), target_samples_per_class)
shuffle_indices = np.random.permutation(target_samples_per_class*num_classes)
imgs = imgs[shuffle_indices]
labels_int = labels_int[shuffle_indices]
print(f"Done...\t\t", end='\r')

print('Saving Cleaned Data in "data/cherrypicked/TRAIN"',end='\r')
np.save('./../data/cherrypicked/TRAIN/train_images_clean',imgs)
np.save('./../data/cherrypicked/TRAIN/train_labels_clean',labels_int)

print('Complete')
print('"data/train_tar" can be deleted to save space')