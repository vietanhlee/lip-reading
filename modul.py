import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
class TemporalBlock(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation_rate, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = layers.Conv1D(filters=n_outputs, kernel_size=kernel_size,
                                   padding=padding, dilation_rate=dilation_rate, 
                                   kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.dropout1 = layers.Dropout(dropout)

        self.conv2 = layers.Conv1D(filters=n_outputs, kernel_size=kernel_size,
                                   padding=padding, dilation_rate=dilation_rate,
                                   kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.dropout2 = layers.Dropout(dropout)

        self.downsample = None
        if n_inputs != n_outputs:
            self.downsample = layers.Conv1D(n_outputs, kernel_size=1, padding="same")

        self.final_relu = layers.ReLU()

    def call(self, x, training=None):
        out = self.conv1(x)
        # out = self.bn1(out, training=training)
        out = self.relu1(out)
        out = self.dropout1(out, training=training)

        out = self.conv2(out)
        # out = self.bn2(out, training=training)
        out = self.relu2(out)
        out = self.dropout2(out, training=training)

        res = x if self.downsample is None else self.downsample(x)
        out = tf.keras.layers.Add()([out, res])
        return self.final_relu(out)

class TemporalConvNet(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.3, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)  # Chuyển tiếp tất cả các kwargs (bao gồm 'trainable')
        
        layers_list = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers_list.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation_size, padding="causal", dropout=dropout)
            )

        self.network = tf.keras.Sequential(layers_list)

    def call(self, x, training=None):
        return self.network(x, training=training)
    
    def get_config(self):
        # Lấy cấu hình từ lớp cha (tf.keras.layers.Layer)
        config = super().get_config()
        # Thêm các tham số cần thiết vào config
        config.update({
            "num_inputs": self.network[0].n_inputs if self.network else None,
            "num_channels": [layer.n_outputs for layer in self.network],
            "kernel_size": self.network[0].kernel_size if self.network else None,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Tái tạo đối tượng từ cấu hình đã lưu
        return cls(
            num_inputs=config["num_inputs"],
            num_channels=config["num_channels"],
            kernel_size=config["kernel_size"]
        )
