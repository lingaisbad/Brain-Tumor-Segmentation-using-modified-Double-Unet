from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout , Activation 
from keras.layers import GlobalAveragePooling2D,Dense,Reshape,AveragePooling2D,Multiply,Add


def conv_block(inputs, filters):
    x = inputs

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def encoder(inputs):
  num_filters = [16, 32, 64, 128,256]
  skip_connections = []
  x = inputs
  for i, f in enumerate(num_filters):
    x = conv_block(x, f)
    skip_connections.append(x)
    x = MaxPooling2D((2, 2))(x)

  print("Skip", skip_connections)
 
  return x, skip_connections


def decoder1(inputs, skip_connections):
    num_filters = [256, 128, 64, 32, 16]
    skip_connections.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = concatenate([x, skip_connections[i]])
        x = conv_block(x, f)
    
    return x


def decoder2(inputs, skip_1, skip_2):
    num_filters = [256,128, 64, 32, 16]
    skip_2.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = concatenate([x, skip_1[i], skip_2[i]])
        x = conv_block(x, f)
   
    return x


def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    
    return x


def build_model(shape):
    inputs = Input(shape)

    x, skip_1 = encoder(inputs)
    x = conv_block(x,512)
    x = decoder1(x, skip_1)
    outputs1 = output_block(x)

    x = inputs * outputs1
    print("Multiply", x)

    x, skip_2 = encoder(x)
    x = conv_block(x,512)
    x = decoder2(x, skip_1, skip_2)
    outputs2 = output_block(x)
    
    outputs = outputs1*outputs2
    #outputs = concatenate([outputs1, outputs2])

    model = Model(inputs, outputs, name="double_unet")
    return model


model = build_model((256, 256, 3))
model.summary()