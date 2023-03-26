

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout , Activation 
from keras.layers import GlobalAveragePooling2D,Dense,Reshape,AveragePooling2D,Multiply,Add

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


def conv_block(inputs, filters):
    x = inputs

    x = Conv2D(filters, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x)

    return x


def rconv_block(x, n_filter):
    x_init = x
    ## Conv 1
    x = Conv2D(n_filter, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ## Conv 2
    x = Conv2D(n_filter, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ## Conv 3
    x = Conv2D(n_filter, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ## Shortcut
    s = Conv2D(n_filter, (1, 1), padding="same")(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    return x


def resnet_block(x, n_filter, pool=True):
    x1 = rconv_block(x, n_filter)
    c = x1

    ## Pooling
    if pool == True:
        x = MaxPooling2D((2, 2), (2, 2))(x1)
        return c, x
    else:
        return c


def encoder1(inputs):
    num_filters = [8,16, 32, 64, 128]
    c1, p1 = resnet_block(inputs, num_filters[0])
    c2, p2 = resnet_block(p1, num_filters[1])
    c3, p3 = resnet_block(p2, num_filters[2])
    c4, p4 = resnet_block(p3, num_filters[3])
    c5, p5 = resnet_block(p4, num_filters[4])
    skip_connections = [p1, p2, p3, p4, p5]
    print("Skip1", skip_connections)
    return p5, skip_connections


def decoder1(inputs, skip_connections):
    num_filters = [128, 64, 32, 16, 8]
    skip_connections.reverse()
    x = inputs
    print(x)

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = concatenate([x, skip_connections[i]])
        x = conv_block(x, f)

    return x


def encoder2(inputs):
    num_filters = [8, 16, 32, 64, 128]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPooling2D((2, 2))(x)

    print("Skip2", skip_connections)
    return x, skip_connections


def decoder2(inputs, skip_1, skip_2):
    num_filters = [128, 64, 32, 16, 8]
    skip_2.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = concatenate([x, skip_1[i], skip_2[i]])
        x = conv_block(x, f)

    return x


def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = Activation('sigmoid')(x)
    return x

def ASPP(x, filter):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = concatenate([y1, y2, y3, y4, y5])

    y = MaxPooling2D(pool_size=(2, 2))(y)

    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def build_model(shape):
    inputs = Input(shape)
    x, skip_1 = encoder1(inputs)
    x = ASPP(x,256)
    x = decoder1(x, skip_1)

    outputs1 = output_block(x)

    x = inputs * outputs1
    print("Multiply", x)

    x = MaxPooling2D((2, 2))(x)

    x, skip_2 = encoder2(x)
    x= conv_block(x,256)
    x = decoder2(x, skip_1, skip_2)

    outputs2 = output_block(x)

    outputs = outputs1*outputs2

    model = Model(inputs, outputs, name="unet")
    return model


model = build_model((256, 256, 3))
model.summary()