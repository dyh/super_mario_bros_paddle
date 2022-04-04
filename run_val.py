import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, ReLU, Linear, Layer
import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
import math
import numpy as np
import subprocess as sp


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError as ex:
            print('ffmpeg error')
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


def create_train_env(world, stage, actions, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    return env


def conv_out(In):
    return (In - 3 + 2 * 1) // 2 + 1
    # (input−kernel_size+2*padding)//stride+1


class MARIO(Layer):
    def __init__(self, actions, obs_dim):
        super(MARIO, self).__init__()
        self.channels = 32
        self.kernel = 3
        self.stride = 2
        self.padding = 1
        self.fc = self.channels * math.pow(conv_out(conv_out(conv_out(conv_out(obs_dim[-1])))), 2)
        self.conv0 = Conv2D(out_channels=self.channels,
                            kernel_size=self.kernel,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=[1, 1],
                            groups=1,
                            in_channels=obs_dim[1])
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
        x = paddle.to_tensor(data=x)
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = paddle.reshape(x, shape=[1, -1])
        x = self.linear0(x)
        logits = self.linear1(x)
        value = self.linear2(x)
        return logits, value


def main(world, stage):
    actions = SIMPLE_MOVEMENT
    obs_dim = [1, 4, 84, 84]
    env = create_train_env(world, stage, actions, "./video/mario_{}_{}.avi".format(world, stage))
    paddle.disable_static()
    params = paddle.load('./models/mario_{}_{}.pdparams'.format(world, stage))
    model = MARIO(len(actions), obs_dim)
    model.set_dict(params)
    model.eval()
    state = env.reset()
    while True:
        logits, value = model(state)
        print(logits)
        policy = F.softmax(logits).numpy()
        action = np.argmax(policy)
        state, reward, done, info = env.step(action)
        state = np.array(state).astype('float32')
        # env.render()
        if done:
            break


world = 1
stage = 1
if __name__ == '__main__':
    main(world, stage)
