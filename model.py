import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, ReLU, Linear, Layer

import math


def conv_out(In):
    return (In - 3 + 2 * 1) // 2 + 1
    # (input−kernel_size+2*padding)//stride+1


class MARIO(Layer):
    def __init__(self, input_num, actions):
        super(MARIO, self).__init__()
        self.num_input = input_num
        self.channels = 32
        self.kernel = 3
        self.stride = 2
        self.padding = 1
        # self.fc = self.channels*math.pow(conv_out(conv_out(conv_out(conv_out(obs_dim[-1])))),2)
        self.fc = 32 * 6 * 6
        nn.initializer.set_global_initializer(nn.initializer.KaimingUniform(), nn.initializer.Constant(value=0.))
        self.conv0 = Conv2D(out_channels=self.channels,
                            kernel_size=self.kernel,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=[1, 1],
                            groups=1,
                            in_channels=input_num)
        self.relu0 = ReLU()
        self.conv1 = Conv2D(out_channels=self.channels,
                            kernel_size=self.kernel,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=[1, 1],
                            groups=1,
                            in_channels=self.channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(out_channels=self.channels,
                            kernel_size=self.kernel,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=[1, 1],
                            groups=1,
                            in_channels=self.channels)
        self.relu2 = ReLU()
        self.conv3 = Conv2D(out_channels=self.channels,
                            kernel_size=self.kernel,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=[1, 1],
                            groups=1,
                            in_channels=self.channels)
        self.relu3 = ReLU()
        self.linear0 = Linear(in_features=int(self.fc), out_features=512)
        self.linear1 = Linear(in_features=512, out_features=actions)
        self.linear2 = Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.linear0(x)
        logits = self.linear1(x)
        value = self.linear2(x)
        return logits, value