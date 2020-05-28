# assignment-9

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    file_data   = "mnist.csv"
    handle_file = open(file_data, "r")
    data        = handle_file.readlines()
    handle_file.close()


    size_row    = 28    # height of the image
    size_col    = 28    # width of the image

    num_image   = len(data)
    count       = 0     # count for the number of images


#
# normalize the values of the input data to be [0, 1]
#
    def normalize(data):

    data_normalized = (data - min(data)) / (max(data) - min(data))

    return(data_normalized)
 # 1.2.plot the loss curve and plot the accuracy curve

    def plot_history(net_history):
     history = net_history.history
     import matplotlib.pyplot as plt
      losses = history['loss']
      val_losses = history['val_loss']
      accuracies = history['accuracy']
      val_accuracies = history['val_accuracy']
    
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.plot(losses, 'r')
      plt.plot(val_losses, 'b')
      plt.legend(['loss', 'val_loss'])
    
      plt.figure()
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.plot(accuracies, 'b')
      plt.plot(val_accuracies, 'r')
      plt.legend(['acc', 'val_acc'])

# Load data
    train_images,test_images, train_labels, test_labels= train_test_split(x,y,train_size=0.5,test_size=0.5,random_state=123)


# Data attributes 
    print("train_images dimentions: ", train_images.ndim)
    print("train_images shape: ", train_images.shape)
    print("train_images type: ", train_images.dtype)

    X_train = train_images.reshape(5000, 784)
    X_test = test_images.reshape(5000, 784)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    from keras.utils import np_utils
    Y_train = np_utils.to_categorical(train_labels)
    Y_test = np_utils.to_categorical(test_labels)

#==================================================
# Creating our model
    from keras.models import Sequential
    from keras.layers import Dense, Dropout,Conv2D
    from keras.optimizers import SGD
    from keras.losses import categorical_crossentropy

    myModel = Sequential()
    myModel.add(Dense(196, activation='relu', input_shape=(784,)))
    myModel.add(Dropout(20))
    myModel.add(Dense(49, activation='relu'))
    myModel.add(Dropout(20))
    myModel.add(Dense(10, activation='softmax'))


    myModel.summary()
    myModel.compile(optimizer=SGD(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])


#==================================================
