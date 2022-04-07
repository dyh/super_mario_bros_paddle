# 超级玛丽 PPO训练54分钟 打通第1关

#### 注意：本项目多进程训练代码 train_mp.py ，只能在 linux 下运行

### 项目信息

- 项目作者 王子瑞
- 文章地址 https://blog.csdn.net/wzduang/article/details/113093206
- 项目代码 https://github.com/Wongziseoi/PaddleMario

### 视频





## 运行环境

- ubuntu 20.04
- python 3.8
- CUDA 11.2
- CUDNN 8.3
- paddlepaddle-gpu==2.2.2.post112

## 如何运行

1. 下载代码

    ```
    $ git clone https://github.com/dyh/super_mario_bros_paddle.git
    ```

2. 进入目录

    ```
    $ cd super_mario_bros_paddle
    ```

3. 创建 python 虚拟环境

    ```
    $ python3 -m venv venv
    ```

4. 激活虚拟环境

    ```
    $ source venv/bin/activate
    ```
   
5. 升级pip和setuptools

    ```
    $ python -m pip install --upgrade pip

    $ pip install --upgrade setuptools
    ```

6. 安装paddlepaddle

    > 根据你的操作系统、安装工具以及CUDA版本，在 https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html 找到对应的安装命令。我的环境是 ubuntu 20.04、pip、CUDA 11.2。

    ```
    $ python -m pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
    ```

7. 安装其他包

    ```
    $ pip install -r requirements.txt
    ```

8. 运行多进程训练程序

    > 设置 8 进程，总耗时 54.31 分钟，日志请见 train_log_1_1.txt 文件，大约 Episode: 352 完成第一关的训练

    ```
    python train_mp.py
    ```

9. 运行预测程序

    > 第一关训练完成的权重文件保存在 ./models/mario_1_1.pdparams

    ```
    python run_eval.py
    ```
