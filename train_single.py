"""
项目作者 王子瑞
文章地址 https://blog.csdn.net/wzduang/article/details/113093206
项目代码 https://github.com/Wongziseoi/PaddleMario
"""

import os

from game_env import Environment
from game_env import create_train_env
from model import MARIO

import paddle
from paddle.distribution import Categorical
import paddle.nn.functional as F
import numpy as np
import shutil
from visualdl import LogWriter

from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from tqdm import trange


def eval(local_model, log_writer, eval_epch):
    # 选择操作模式
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    env = create_train_env(world, stage, actions)

    state = paddle.to_tensor(env.reset(), dtype="float32")

    curr_step = 0
    total_reward = 0
    actions = deque(maxlen=max_actions)

    while True:
        curr_step += 1
        logits, value = local_model(state)
        policy = F.softmax(logits, axis=-1).numpy()
        action = np.argmax(policy)
        state, reward, done, info = env.step(int(action))

        total_reward += reward

        # 通关时保存模型
        if info["flag_get"]:
            print("Finished")
            paddle.save(local_model.state_dict(), "{}/mario_{}_{}.pdparams".format(saved_path, world, stage))
        pass

        actions.append(action)
        if curr_step > num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            eval_epch += 1
            actions.clear()
            log_writer.add_scalar("Eval reward", value=paddle.to_tensor(total_reward), step=eval_epch)
            total_reward = 0
            break
        state = paddle.to_tensor(state, dtype="float32")
    return eval_epch


def train():
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    pass

    os.makedirs(log_path)

    if not os.path.isdir(saved_path):
        os.makedirs(saved_path)
    pass

    envs = Environment(world, stage, action_type)

    model = MARIO(envs.num_states, envs.num_actions)

    clip_grad = paddle.nn.ClipGradByNorm(clip_norm=0.5)
    optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters(), grad_clip=clip_grad)
    log_writer = LogWriter(logdir=log_path, comment="Super Mario Bros")

    curr_states = [envs.env.reset()]

    curr_states = paddle.to_tensor(np.concatenate(curr_states, 0), dtype="float32")

    curr_episode = 0
    eval_epch = 0
    while True:
        # 定期保存模型
        if curr_episode % save_interval == 0 and curr_episode > 0:
            paddle.save(model.state_dict(),
                        "{}/mario_{}_{}_{}.pdparams".format(saved_path, world, stage, curr_episode))
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        # 预热部分
        train_reward = 0
        for _ in range(num_local_steps):
            states.append(curr_states)

            logits, value = model(curr_states)

            if value.shape == [1, 1]:
                values.append(value[0])
            else:
                values.append(value.squeeze())
            pass

            policy = F.softmax(logits, axis=-1)

            old_m = Categorical(policy)

            action = old_m.sample([1])

            if action.shape == [1, 1]:
                action = action[0]
            else:
                action = action.squeeze()
            pass

            actions.append(action)

            origin_old_log_policy = old_m.log_prob(action)

            old_log_policy = paddle.tensor.tril(origin_old_log_policy)
            old_log_policy = paddle.tensor.triu(old_log_policy)
            old_log_policy = paddle.sum(old_log_policy, axis=1)

            old_log_policies.append(old_log_policy)

            state, reward, done, info = envs.env.step(int(action))

            train_reward += np.mean(reward)

            state = [state]

            state = paddle.to_tensor(np.concatenate(state, 0), dtype="float32")
            reward = paddle.to_tensor(reward, dtype="float32")
            done = paddle.to_tensor(done, dtype="float32")

            rewards.append(reward)
            dones.append(done)
            curr_states = state

        log_writer.add_scalar("Training Reward", value=paddle.to_tensor(train_reward, dtype="float32"),
                              step=curr_episode)

        _, next_value, = model(curr_states)

        if next_value.shape == [1, 1]:
            next_value = next_value[0]
        else:
            next_value = next_value.squeeze()
        pass

        old_log_policies = paddle.concat(old_log_policies, axis=-1).detach()
        actions = paddle.concat(actions).squeeze()
        values = paddle.concat(values).squeeze().detach()
        states = paddle.concat(states).squeeze()
        gae = paddle.to_tensor([0.])
        R = []

        # PG 优势函数计算过程
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * gamma * tau
            gae = gae + reward + gamma * next_value.detach() * (1.0 - done) - value.detach()
            next_value = value
            R.append(gae + value.detach())

        R = R[::-1]  # 倒序
        R = paddle.concat(R).detach()
        advantages = R - values

        for i in trange(num_epochs):
            model.train()
            indice = paddle.randperm(num_local_steps)  # 返回一个 0 到 n-1 的数组
            for j in range(batch_size):
                batch_indices = indice[
                                int(j * (num_local_steps / batch_size)): int((j + 1) * (
                                        num_local_steps / batch_size))]

                batch_advantages = paddle.gather(advantages, batch_indices, axis=0)
                batch_R = paddle.gather(R, batch_indices, axis=0)
                batch_old_log_policies = paddle.gather(old_log_policies, batch_indices, axis=0)
                batch_states = paddle.gather(states, batch_indices, axis=0)
                batch_actions = paddle.gather(actions, batch_indices, axis=0)

                logits, value = model(batch_states)
                new_policy = F.softmax(logits, axis=-1)

                new_m = Categorical(new_policy)
                origin_new_log_policy = new_m.log_prob(batch_actions)
                # eye = paddle.eye(new_policy.shape[0])
                # new_log_policy = paddle.sum(paddle.multiply(origin_new_log_policy, eye), axis=1).squeeze()
                new_log_policy = paddle.tensor.tril(origin_new_log_policy)
                new_log_policy = paddle.tensor.triu(new_log_policy)
                new_log_policy = paddle.sum(new_log_policy, axis=1).squeeze()

                ratio = paddle.exp(new_log_policy - batch_old_log_policies)

                actor_loss = paddle.concat([paddle.unsqueeze(ratio * batch_advantages, axis=0), paddle.unsqueeze(
                    paddle.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * batch_advantages, axis=0)])

                actor_loss = -paddle.mean(paddle.min(actor_loss, axis=0))
                # critic_loss = paddle.mean((batch_R - value.squeeze()).pow(2)) / 2
                critic_loss = F.smooth_l1_loss(batch_R, value.squeeze())
                entropy_loss = paddle.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - beta * entropy_loss

                if not str(total_loss.numpy().item()) == "nan":
                    pass
                else:
                    continue
                optimizer.clear_grad()
                total_loss.backward()
                optimizer.step()
                pass
            pass
        pass

        print("Episode: {}. Total loss: {}".format(curr_episode, total_loss.numpy().item()))
        model.eval()
        eval_epch = eval(model, log_writer, eval_epch)
        if not str(total_loss.numpy().item()) == "nan":
            log_writer.add_scalar("Total loss", value=total_loss, step=curr_episode)
        else:
            continue
        pass
    pass


if __name__ == "__main__":
    # 不需要调整的全局变量
    gamma = 0.9  # 奖励的折算因子
    tau = 1.0  # GAE(Generalized Advantage Estimation), 即优势函数的参数
    beta = 0.01  # 交叉熵的系数
    epsilon = 0.2  # 裁剪后的替代目标函数(PPO 提出)的参数
    batch_size = 16
    num_epochs = 10
    num_local_steps = 512
    num_global_steps = int(5e6)
    save_interval = 50  # 定期保存间隔
    max_actions = 512
    log_path = "./log"  # 日志保存路径
    saved_path = "./models"

    # 可以调整的全局变量
    world = 1  # 世界
    stage = 1  # 关卡
    action_type = "simple"  # 操作模式
    lr = float(1e-4)  # 学习率

    paddle.seed(314)
    print("Proximal Policy Optimization Algorithms (PPO) playing Super Mario Bros")
    train()
