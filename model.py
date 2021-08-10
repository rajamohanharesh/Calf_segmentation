from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras





def conv_block(m, dim):
    conv = Conv2D(dim, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(m)
    conv = BatchNormalization()(conv)
    conv = Activation(activations.relu)(conv)
    conv = Conv2D(dim, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(activations.relu)(conv)
    return conv


# In[18]:


def up_conv_block(m, dim):
    upconv = Conv2DTranspose(dim, 2, strides=(2,2), padding='same', kernel_initializer = 'he_normal')(m)    
    upconv = BatchNormalization()(upconv)
    upconv = Activation(activations.relu)(upconv)
    return upconv


# In[19]:


def unet(pretrained_weights = None,input_size = (192,192,1), NUM_CLASSES=6,DROPOUT_RATE=0.1):
    inputs = Input(input_size)

    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512)
    drop4 = Dropout(DROPOUT_RATE)(conv4)

    up7 = up_conv_block(drop4, 256)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = conv_block(merge7, 256)

    up8 = up_conv_block(conv7, 126)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = conv_block(merge8, 128)

    up9 = up_conv_block(conv8, 64)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = conv_block(merge9, 64)
    conv10 = Conv2D(NUM_CLASSES, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)

    return Model(input = inputs, outputs = conv10)