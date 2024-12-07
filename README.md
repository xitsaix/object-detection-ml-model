# Machine Learning Model for Furni-Scan Project

Utilize **Google Colab** to create a Machine Learning model that can identify and classify various types of furniture from images. Colab is a hosted Jupyter Notebook service that requires no setup to use and provides a seamless experience with no setup required and offers free access to robust compute resources, including GPUs and TPUs.


## Prerequisites

 install:

- **Python 3.6 or higher**
- **TensorFlow 2.x**
- **Keras (included with TensorFlow 2.x)**


## Steps to Create a Model

**1. Import Libraries:**

- `ImageDataGenerator` from `tensorflow.keras.preprocessing.image` is used for data preprocessing and augmentation, helps in enhancing the robustness and generalizability of our model by providing various transformations on the training images.=.

**2. Set Parameters:**

- `img_height` and `img_width` define the dimensions to resize all images to standardization is essential for consistent input sizes into the model.
- `batch_size` specifies the number of images to be yielded from the generator per batch.

**3. Create Training Data Generator:**

- `train_datagen` is configured with various augmentation techniques such as rescaling, rotation, shifting, shearing, zooming, and flipping, for creating a more diverse set of training data, which helps prevent overfitting and improves the model's ability to generalize.

**4. Create Validation Data Generator:**

- `val_datagen` is configured to only rescale the images, prepares the validation dataset without applying augmentations, ensuring that we evaluate the model performance on clean, unaltered data.

**5. Load Training Data:**

- `train_generator` loads training data from the train directory, applies augmentations, and prepares the data in batches, making the training process more efficient and less memory-intensive.

**6. Load Validation Data:**

- `validation_generator` loads validation data from the validation directory, rescaling the images without further augmentations, providing a standard evaluation dataset to test the model's performance during training.

**7. Training the Model**

- Compile and train the model using the `train_generator` and `validation_generator`. Adjust the model architecture, optimizer, and loss function as needed to best fit the problem. The training process involves iterating over batches of data, optimizing the model's weights to minimize the loss function.

**8. Evaluate and Save the Model**

- After training, evaluate the model's performance on the validation set to ensure it generalizes well. Save the trained model for future use or deployment.
