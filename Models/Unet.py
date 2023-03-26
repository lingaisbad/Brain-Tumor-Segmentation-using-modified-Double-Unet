
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout , Activation

def build_model(shape):
    inputs = Input(shape)

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters,(3,3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters,(3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    s = conv_block(inputs, num_filters)
    p = MaxPooling2D((2, 2))(s)
    return s, p

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = concatenate([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_model(input_shape):
    """ Input """
    inputs = Input(input_shape)
 
    """ Encoder """
    s1,p1 = encoder_block(inputs,16)
    s2,p2 = encoder_block(p1, 32)        
    s3,p3 = encoder_block(p2, 64)          
    s4,p4 = encoder_block(p3, 128)           
    s5,p5 = encoder_block(p4, 256)           

    """ Bridge """
    b1 = conv_block(p5, 512)  

    """ Decoder """
    d1 = decoder_block(b1, s5, 256)
    d2 = decoder_block(d1, s4, 128)
    d3 = decoder_block(d2, s3, 64) 
    d4 = decoder_block(d3, s2, 32)                     
    d5 = decoder_block(d4, s1, 16)                     
                    
                   

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d5)

     
    model = Model(inputs=[inputs], outputs=[outputs])
       
    return model
    
model = build_model((256, 256, 3))
model.summary()