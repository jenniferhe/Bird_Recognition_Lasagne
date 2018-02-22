#BirdID_recognition_lasagne

Bird image classification using convolutional neural networks in Python

![Sample Bird Images](http://i.imgur.com/R2rdTBe.png)



## How to use:

文件包含两部分，图片库和代码文件。图片是经过调整downsample和augmentation到128x128大小的9类鸟的图片，代码文件包含training代码和configuration部分

The folder with the images to classify has to be structured as such:

```
.
|-- path_to_folders_with_images
|    |-- class1
|    |    |-- some_image1.jpg
|    |    |-- some_image2.jpg
|    |    |-- some_image3.jpg
|    |    └ ...
|    |-- class2
|    |    └ ...
|    |-- class3
    ...
|    └-- classN
```

Two kinds of python files are provided here: Configuration and Training



### Configuration:

The parameters to be configured are:

| Name                   | Values             | Description                                                  |
| ---------------------- | ------------------ | ------------------------------------------------------------ |
| RATIO                  | [0,1]              | The ratio of the dataset to use for training. The remainder will be used for validation |
| PER_CATEGORY           | integer            | The number of images per category to be used for classification. Can not be higher than the number of images in any given category. |
| CATEGORIES             | integer            | The number of categories--used to assign labels              |
| DIR                    | String             | The directory containing the images, can be relative to the working directory |
| TYPE                   | String             | The extension for the images located in the folders, e.g. ".jpg" |
| DIM                    | integer            | Size of network input images. e.g. "128" will mean that input images are 128x128 pixels. Images will be resized as needed |
| PREAUG_DIM             | integer            | The dimension of the images prior to augmentation through random crops. Set this value equal to DIM above to avoid random crops |
| EPOCHS                 | integer            | Maximum number of epochs to train                            |
| BATCH_SIZE             | integer            | Batch size                                                   |
| SEED1                  | int or RandomState | The seed used to pick PER_CATEGORY number of images from each directory. Set to None for a random pick. |
| SEED2                  | int or RandomState | The seed used to generate stratified data splits based on RATIO. Set to None for a random split. |
| SAVE                   | boolean            | Save the network state or not--can be set to false either way (see description for the training files) |
| l2_regularization_rate | [0,1]              | L2 regularization constant                                   |
| learning_rate          | [0,1]              | Learning rate                                                |
| algorithm              | String             | The adaptive learning algorithm to use. Options are "rmpsprop", "adagrad", "adam". |

Additionally, all configuration files must have a network architecture specified within their build_model() methods, and this method must return a tuple of the input and output layers of this network for the training file to use.For available layers and such, see documentation for [Lasagne](#).

### Training:

首先将图片文档分为Test, validation 和training部分，然后对training data通过进行随机的裁剪和旋转完成augmentation。

然后进行training部分并打印出当前准确度

Files named train_net*.py are used for training networks based on configurations. The recommended one to use is train_net_args.py. The training scripts accept a few command line arguments:

| flag | alternative | description                                                  |
| ---- | ----------- | ------------------------------------------------------------ |
| -c   | --config    | Name of the configuration file. e.g. sx3_b32_random (do not include the extension) |
| -s   | --save      | Name that will be given to the .npy containing network parameters. If no name is specified, the network parameters are not saved after training. |
| -r   | --resume    | Name of the npy file to use to load a network to resume training. Make sure that a matching configuration file is used (and a low learning rate might be preferred) |

## Results:

These networks were used to classify photos of 9 species of birds. The dataset had a minimum of 98 images per category.

Images are resized to 140x140, and then augmented using random horizontal flips and crops to 128x128 with random offsets. The validation set goes through the exact same method for augmentation. 

The networks were trained using stochastic gradient descent(SGD), utilizing an adaptive subgradient method to change the learning rate over time. 

Rectified linear units were used as the activation function for both the convolutional and fully connected layers.

"Same" convolutions were used through zero-padding to keep the input and output dimensions the same.

The optimal initial learning rate and adaptive algorithm were determined using [simple_spearmint](#).The script used for hyperparameter optimization is included, see optimize.py