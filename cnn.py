from cnn_layers import ConvLayer
from cnn_layers import MaxPoolLayer
from cnn_layers import FlattenLayer
from cnn_layers import FullyConnected

from activations import softmax


class CNN:

    def __init__(self):

        # Convolution Layer
        self.conv = ConvLayer(
            num_filters=8,
            filter_size=3
        )

        # MaxPool Layer
        self.pool = MaxPoolLayer(
            pool_size=2
        )

        # Flatten Layer
        self.flatten = FlattenLayer()

        # Fully Connected Layer
        self.fc = FullyConnected(
            input_len=13 * 13 * 8,
            output_len=10
        )

    
    # FORWARD PASS
    def forward(self, image):

        # Normalize
        image = image / 255.0

        # Convolution
        output = self.conv.forward(image)

        # MaxPooling
        output = self.pool.forward(output)

        # Flatten
        output = self.flatten.forward(output)

        # Fully Connected
        output = self.fc.forward(output)

        # Softmax
        probabilities = softmax(
            output.reshape(1, -1)
        )

        return probabilities