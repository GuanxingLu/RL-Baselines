# RL-Baselines

## Requirements

```
conda create -n py38 python==3.8

conda activate py38

conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

pip install wheel==0.38.0   # [Optional]
pip install gym==0.21.0 tqdm matplotlib numpy readchar -i https://pypi.tuna.tsinghua.edu.cn/simple
```

To run Montezuma's Revenge, please install:

```
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install gym[atari] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install autorom[accept-rom-license] -i https://pypi.tuna.tsinghua.edu.cn/simple
```

To run RL5, please install:
```
pip install gym==0.26.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Usage

To run Montezuma's Revenge:

```
python dqn_for_montezuma.py
```

## Note

- ``dqn_for_montezuma.py`` is implemented upon gym==0.26.2. Please consider revising the code if you are using older versions of gym.
- ``dqn_for_montezuma.py`` only acts as a very simple baseline!

## Error Catch

- libGL error:
```
conda install -c conda-forge gcc
```

If you have any question, please feel free to pose an issue or contact: ``lgx23 at mails dot tsinghua dot edu dot cn``

We welcome contributions to this repository. Please submit a pull request if you want to contribute!

## References

https://hrl.boyuai.com/

https://github.com/boyu-ai/Hands-on-RL

https://github.com/AssistiveRoboticsUNH/bc_tutorial
