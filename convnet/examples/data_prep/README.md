### Data Preparation

ConvNet is designed to run on data stored in different formats. Storing the data in an
HDF5 file is probably the fastest way for the code to access it during training. This is also the default format used for writing out features and storing model parameters.

#### Storing images in HDF5 files.
ConvNet provides `image2hdf` : a useful tool to extract JPEG images into an HDF5 file.

For example, 
```
image2hdf --input=test_images.txt --output=test_images.h5 --resize=256 --crop=224
```

This takes the image files listed in `test_images.txt` and writes them out to an HDF5 dataset. Each image will be resized so that the shorter side is 256. A central 224 x 224 crop will then be taken. The data is written out in a HDF5 dataset called `data`.
```
$ h5ls test_images.h5
data                     Dataset {10, 150528}
```
The number of rows is the number of images. The number of columns is 224 x 224 x 3. The data is written out with the R, G and B channels separated  [RRR ...(224x224) .. GGG ... BB ...]. The fastest changing dimension is along the width of the image, followed by height, followed by color channel.

To prepare the ILSVRC data - 
```
image2hdf --input=train_images.txt --output=imagenet_train.h5 --resize=256 --crop=256
image2hdf --input=valid_images.txt --output=imagenet_valid.h5 --resize=256 --crop=256
```
where the `train_images.txt` file lists a random permutation of the training JPEG images.
We take the central 256 x 256 crop of the image. Smaller random crops of this image will be taken later, on the fly, during training.


#### Writing Data Protocol Buffers
A data protocol buffer written in a text form looks something like this - 
```
data_config {
  file_pattern: "/path/to/data/data.h5"
  layer_name: "input"
  dataset_name: "train"
  data_type: HDF5
}
data_config {
  file_pattern: "/path/to/data/data.h5"
  layer_name: "output"
  dataset_name: "train_labels"
  data_type: HDF5
}
batch_size : 100
randomize_cpu: true
randomize_gpu: true
```
It describes which data streams are involved, which layers of the neural network they correspond to, where the data is stored, which format it is stored in, ..  etc.
The full details can be found in `convnet/proto/convnet_config.proto` under `message DatasetConfig`.
