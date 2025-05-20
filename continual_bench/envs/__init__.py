import numpy as np

from continual_bench.envs.mujoco.sawyer_bench import SawyerBenchEnv


def initialize(env, seed=None, render_mode=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
    super(type(env), env).__init__()
    if seed is not None:
        env.seed(seed)
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.render_mode = render_mode
    env.reset()
    env._freeze_rand_vec = True
    if seed is not None:
        np.random.set_state(st0)


d = {}
d["__init__"] = initialize
ContinualBenchEnv = type("ContinualBench", (SawyerBenchEnv,), d)
