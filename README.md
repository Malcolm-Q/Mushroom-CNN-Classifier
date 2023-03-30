# Mushroom-CNN-Classifier
LHL Jan-09 Cohort Final Project using [DanishFungi](https://sites.google.com/view/danish-fungi-dataset/samples) dataset.

## <u>The Problem</u>
While mushroom classification with CNNs has been done a lot. All the architectures/solutions I've seen have utilized very traditional single input CNNs. Due to the nature of mushroom classification and how foragers often take two main perspectives into account (top/side and underneath/gills) when identifying a mushroom I think this problem would benefit from using a siamese architecture. I'm going to prove that this is the case by constructing a baseline model, converting it to a siamese model, and overcoming the failures of the single model.

## <u> Quickstart </u>
In the src folder you can run setup_data.py to prep the downloaded dataset as I have if you want to tweak the architecture or follow along.

[Download the Images (6GB).](http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-300px.tar.gz)

[Download the Labels (100MB).](http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-metadata.zip)

These need to be extracted into the 'data/raw' folder, then you can run setup_data.py.

The model architectures can be found in 'outputs/Deployments/ModelLoader.py'