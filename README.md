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
    
#
# make a matrix each column of which represents an images in a vector form
#
     list_image  = np.empty((size_row * size_col, num_image), dtype=float)
     list_label  = np.empty(num_image, dtype=int)

    for line in data:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])
    im_vector   = normalize(im_vector)

    list_label[count]       = label
    list_image[:, count]    = im_vector

    count += 1
    
    
    y, x= list_label, list_image
    x = x.transpose()

