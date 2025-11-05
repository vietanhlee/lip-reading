from tensorflow import keras
from keras import layers, Layer, Sequential
from keras.optimizers import Adam  

class TCN(Layer):
    def __init__(self, filters, dilated_rate, dropout, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilated_rate = dilated_rate
        
        self.conv1 = layers.Conv1D(filters= filters, kernel_size= kernel_size,
                                   dilation_rate= dilated_rate, padding= 'causal')
        self.relu1 = layers.ReLU()
        self.drop1 = layers.Dropout(rate= dropout, seed= 42)
        
        self.conv2 = layers.Conv1D(filters= filters, kernel_size= kernel_size,
                                   dilation_rate= dilated_rate, padding= 'causal')
        self.relu2 = layers.ReLU()
        self.drop2 = layers.Dropout(rate= dropout, seed= 42)
        
        self.downsample = None
        self.relu_end = layers.ReLU()
    

    def build(self, input_shape): 
        if input_shape[-1] != self.filters:
            self.downsample = layers.Conv1D(filters= self.filters, kernel_size= 1,
                                            padding= 'same')
    def call(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        
        residual = x if self.downsample == None else self.downsample(x)
        out = self.relu_end(out + residual)
        return out
    
class TCN_bulid(keras.Layer):
    def __init__(self, list_filters):
        super().__init__()
        self.blocks_tcn = []
        for i, filters in enumerate(list_filters):
            self.blocks_tcn.append(TCN(filters= filters, dilated_rate= 2 ** i,
                                       kernel_size= 3, dropout= 0.2))
        
    def call(self, x):
        for block in self.blocks_tcn:
            x = block(x)
        return x    
    
def build(num_of_classes):
    model = Sequential([
        layers.Input(shape=(32, 80)), 
        TCN_bulid(list_filters= [64, 128, 256]),
        layers.Flatten(),
        layers.Dense(num_of_classes, activation='softmax')
    ])
    

    model.compile(optimizer= Adam(learning_rate= 0.0001), loss= 'categorical_crossentropy',
                metrics= ['accuracy'])
    
    return model

