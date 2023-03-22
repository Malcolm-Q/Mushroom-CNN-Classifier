# Mushroom-CNN-Classifier
LHL Jan-09 Cohort Final Project using [DanishFungi](https://sites.google.com/view/danish-fungi-dataset/samples) dataset.

## <u> The Contents </u>
* [src](src)
    * [EDA Notebook](src/EDA.ipynb)
    * [modules](src/modules/)
* [data](data)

## <u>The Problem</u>
While mushroom classification with CNNs has been done a lot. All the architectures/solutions I've seen have utilized very traditional CNNs (Maxpooled CNN layers doubling in size). Due to the nature of mushroom classification and how foragers often take two main perspectives into account (top/side and underneath/gills) when identifying a mushroom I think this problem would benefit from using a siamese architecture. I'm going to be expirementing to see if that's the case.