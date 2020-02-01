from tensorflow.keras.layers import Add, Dense, Activation, ZeroPadding3D
from tensorflow.keras.layers import BatchNormalization, Flatten, Conv3D
from tensorflow.keras.layers import AveragePooling3D, MaxPooling3D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform


def identity_block(X, f, filters, stage, block):
    """
    From Coursera shamelessly coppied...
    Implementation of the identity block as defined in Figure 4
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main
     path
    filters -- python list of integers, defining the number of filters in the
    CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in
    the network
    block -- string/character, used to name the layers, depending on their
    position in the network
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2, F3 = filters

    X_shortcut = X

    # First component of main path
    X = Conv3D(filters=F1, kernel_size=(1, 1, 1), strides=(1, 1, 1),
               padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)

    # axis3 could be an issue
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv3D(filters=F2, kernel_size=(f, f, f), strides=(1, 1, 1),
               padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1),
               padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU
    # activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


# GRADED FUNCTION: convolutional_block


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the
    main path
    filters -- python list of integers, defining the number of filters in the
    CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in
    the network
    block -- string/character, used to name the layers, depending on their
    position in the network
    s -- Integer, specifying the stride to be used
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X
    # MAIN PATH #####
    # First component of main path n

    X = Conv3D(F1, (1, 1, 1), strides=(s, s, s), padding='valid',
               name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv3D(F2, (f, f, f), strides=(1, 1, 1), padding='same',
               name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    # Third component of main path (≈2 lines)
    X = Conv3D(F3, (1, 1, 1), strides=(1, 1, 1), padding='valid',
               name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    # SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv3D(F3, (1, 1, 1), strides=(s, s, s), padding='valid',
                        name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(
        axis=3, name=bn_name_base + '1')(X_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU
    # activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


# GRADED FUNCTION: ResNet50

def ResNet50(input_shape=(64, 64, 64, 1), classes=6):
    # Classes need to be fixed
    """
    Implementation of the popular ResNet50 the following architecture:
    Conv3D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 ->
    CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes
    Returns:
    model -- a Model() instance in Keras
    """
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding3D((3, 3, 3))(X_input)
    # Stage 1
    X = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), name='conv1',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(X)
    # Stage 2
    X = convolutional_block(
        X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(
        X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    X = convolutional_block(
        X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    # Stage 5
    X = convolutional_block(
        X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    # AVGPOOL
    X = AveragePooling3D(pool_size=(2, 2), name="avg_pool")(X)
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)
    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    return model


plot_model()
