import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def ShowAllClasses(imgs=None,label_names=None,labels_int=None,figsize=(20,15), rows=None, cols=None):
    '''
    Displays grid of images, one for each class in your dataset.
    Params:
        imgs (array) : Ideally numpy array of images (s, x, y, c).
        label_names (obj) : Object array of class names (optional).
        labels_int (int) : Label encoded array.
        figsize (tup) : plt.subplots figsize. Default (20,15).
        rows (int) : plt.subplots nrows.
        cols (int) : plt.subplots ncols.
    Returns:
        plt.show()
    '''
    assert imgs != None, 'You need to pass a numpy array of images into "imgs="'
    assert labels_int != None, 'You need to pass classes as integers (label encoding) into "labels_int="'
    assert rows != None, 'Please provide the number of rows in "rows="'
    assert cols != None, 'Please provide the number of cols in "cols="'
    
    fig, ax = plt.subplots(nrows=rows,ncols=cols,figsize=figsize)

    for i, ax in enumerate(ax.flat):
        index = np.where(labels_int == i)[0][0]
        ax.imshow(imgs[index],cmap='gray')
        if label_names != None: ax.set_title(label_names[index])
        else:ax.set_title(index)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
    
    
    
def CycleImages(imgs=None,label_names=None,figsize=(20,15), rows=6, cols=5):
    '''
    Displays grid of images, type 'w' to move forward throughout the dataset, 's' to come back, and 'q' to exit the loop.
    Params:
        imgs (array) : Ideally numpy array of images (s, x, y, c).
        label_names (obj) : Object array of class names (optional).
        figsize (tup) : plt.subplots figsize. Default (20,15).
        rows (int) : plt.subplots nrows.
        cols (int) : plt.subplots ncols.
    Returns:
        plt.show()
    '''
    assert imgs != None, 'You need to pass a numpy array of images into "imgs="'
    assert rows != None, 'Please provide the number of rows in "rows="'
    assert cols != None, 'Please provide the number of cols in "cols="'
    
    fig, ax = plt.subplots(nrows=rows,ncols=cols,figsize=figsize)
    starting_index=0
    while(True):
        for i, ax in enumerate(ax.flat):
            ax.imshow(imgs[i+starting_index],cmap='gray')
            if label_names != None: ax.set_title(label_names[index])
            else:ax.set_title(index)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
        
        response = input('"w" to see more images, "q" to quit, "s" to go back')
        if response == 'w': starting_index += rows * cols
        elif response == 's': starting_index -= rows * cols
        elif response == 'q': break
        else: print('please enter "w", "s", "q"')
        
def AvgPixelOutliers(imgs=None,threshold=100):
    outliers = []

    avg_r = np.average(imgs[:,:,:,0])
    avg_g = np.average(imgs[:,:,:,1])
    avg_b = np.average(imgs[:,:,:,2])
    avg_row = np.array([avg_r,avg_g,avg_b])
    for i, img in enumerate(imgs):
        r = np.average(img[:,:,0])
        g = np.average(img[:,:,1])
        b = np.average(img[:,:,2])
        row = np.array([r,g,b])
        distance = np.linalg.norm(avg_row - row)
        if distance > threshold:
            outliers.append(i)

    starting_index=0
    tmp = imgs[outliers]
    while(True):
        clear_output()
        fig, axs = plt.subplots(nrows=4,ncols=5,figsize=(20,10))
        for i, ax in enumerate(axs.flat):
            ax.imshow(tmp[i+starting_index],cmap='gray')
            ax.set_title(str(i+starting_index))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show();
        response = input('type "w" to see more images, "q" to quit, "s" to go back')
        if response == 'w': starting_index += 20
        elif response == 's': starting_index -= 20
        elif response == 'q': break
        else: print('please enter "w", "s", "q"')
        plt.close()