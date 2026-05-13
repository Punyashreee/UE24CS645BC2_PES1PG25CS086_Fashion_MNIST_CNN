import numpy as np

from activations import relu
from activations import relu_derivative


# CONVOLUTION LAYER
class ConvLayer:

    def __init__(self, num_filters, filter_size):

        self.num_filters = num_filters
        self.filter_size = filter_size

        # Random filters
        self.filters = np.random.randn(
            num_filters,
            filter_size,
            filter_size
        ) / (filter_size * filter_size)

   
    # FORWARD PASS
    def forward(self, input):

        self.input = input

        h, w = input.shape

        output_dim = h - self.filter_size + 1

        output = np.zeros(
            (
                self.num_filters,
                output_dim,
                output_dim
            )
        )

        for f in range(self.num_filters):

            current_filter = self.filters[f]

            for i in range(output_dim):
                for j in range(output_dim):

                    region = input[
                        i:i+self.filter_size,
                        j:j+self.filter_size
                    ]

                    output[f, i, j] = np.sum(
                        region * current_filter
                    )

        # Store before activation
        self.conv_output = output

        # Apply ReLU
        activated = relu(output)

        return activated

    # BACKWARD PASS
    def backward(self, dL_dout, learning_rate):

        # Apply ReLU derivative
        dL_dout = dL_dout * relu_derivative(
            self.conv_output
        )

        dL_dfilters = np.zeros(self.filters.shape)

        dL_dinput = np.zeros(self.input.shape)

        output_dim = dL_dout.shape[1]

        for f in range(self.num_filters):

            for i in range(output_dim):
                for j in range(output_dim):

                    region = self.input[
                        i:i+self.filter_size,
                        j:j+self.filter_size
                    ]

                    dL_dfilters[f] += (
                        dL_dout[f, i, j] * region
                    )

                    dL_dinput[
                        i:i+self.filter_size,
                        j:j+self.filter_size
                    ] += (
                        dL_dout[f, i, j]
                        * self.filters[f]
                    )

        # Update filters
        self.filters -= learning_rate * dL_dfilters

        return dL_dinput


# MAXPOOL LAYER
class MaxPoolLayer:

    def __init__(self, pool_size):

        self.pool_size = pool_size

   
    # FORWARD PASS
    def forward(self, input):

        self.input = input

        num_filters, h, w = input.shape

        output_dim = h // self.pool_size

        output = np.zeros(
            (
                num_filters,
                output_dim,
                output_dim
            )
        )

        for f in range(num_filters):

            for i in range(output_dim):
                for j in range(output_dim):

                    region = input[
                        f,
                        i*self.pool_size:(i+1)*self.pool_size,
                        j*self.pool_size:(j+1)*self.pool_size
                    ]

                    output[f, i, j] = np.max(region)

        return output

   
    # BACKWARD PASS
    def backward(self, dL_dout):

        dL_dinput = np.zeros(self.input.shape)

        num_filters, h, w = self.input.shape

        output_dim = h // self.pool_size

        for f in range(num_filters):

            for i in range(output_dim):
                for j in range(output_dim):

                    region = self.input[
                        f,
                        i*self.pool_size:(i+1)*self.pool_size,
                        j*self.pool_size:(j+1)*self.pool_size
                    ]

                    max_value = np.max(region)

                    for x in range(self.pool_size):
                        for y in range(self.pool_size):

                            if region[x, y] == max_value:

                                dL_dinput[
                                    f,
                                    i*self.pool_size + x,
                                    j*self.pool_size + y
                                ] = dL_dout[f, i, j]

        return dL_dinput



# FLATTEN LAYER
class FlattenLayer:

    def forward(self, input):

        self.input_shape = input.shape

        return input.flatten()

    def backward(self, dL_dout):

        return dL_dout.reshape(self.input_shape)



# FULLY CONNECTED LAYER
class FullyConnected:

    def __init__(self, input_len, output_len):

        self.weights = np.random.randn(
            input_len,
            output_len
        ) / input_len

        self.biases = np.zeros(output_len)

   
    # FORWARD PASS
    def forward(self, input):

        self.input_shape = input.shape

        self.input = input.flatten()

        output = np.dot(
            self.input,
            self.weights
        ) + self.biases

        return output

    
    # BACKWARD PASS
    def backward(self, dL_dout, learning_rate):

        dL_dw = np.outer(
            self.input,
            dL_dout
        )

        dL_db = dL_dout

        dL_dinput = np.dot(
            self.weights,
            dL_dout
        )

        # Update weights
        self.weights -= learning_rate * dL_dw

        # Update biases
        self.biases -= learning_rate * dL_db

        return dL_dinput.reshape(self.input_shape)