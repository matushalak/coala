# include Pytorch dataset scripts (ideally including downloading the dataset, and making the appropriate Torch dataset and dataloader)


# The goal is to use these datasets for SSL, possibly with training a small classification / segmentation / reconstruction head on top of the learned representations

# Datasets to include in order of difficulty:
# images
# easy
- EMNIST
- KMNIST
- MovingMNIST
# medium
- STL-10
- TinyImagenet
# hard
- Caltech-101, Caltech-256
- COCO2017
- PASCAL VOC
- Imagenet1k / Imagenet100 / Imagenette
- Open-images-v7
- PASS

# It would be great to directly have datasets that automatically download into their respective folder in data/_DATASETNAME_/. If torchvision.datasets doesn't expose the download API directly, possibly wire it in with huggingface. 

# main ones I want are
- STL-10, and TinyImagenet