import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.95'  # 降低内存分配比例
# 设置XLA内存分配策略
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # 禁用预分配
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # 使用平台分配器
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'  # 减少编译并行度

import jax
import wandb
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import optax
from flax.linen.initializers import constant, orthogonal
import functools
from typing import Sequence, NamedTuple, Tuple, Optional, Union, Any, Dict
from flax.training.train_state import TrainState
import distrax
import tensorboardX
import jax.experimental
from jax.tree_util import tree_map  # 导入正确的tree_map函数
from envs.wrappers import LogWrapper
from envs.aeroplanax_formation import AeroPlanaxFormationEnv, FormationTaskParams
import orbax.checkpoint as ocp
from gymnax.environments import spaces

# 启用JAX垃圾收集
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', False)  # 使用float32以减少内存使用


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    # print('batchify', x.shape)
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    env_params = FormationTaskParams()
    env = AeroPlanaxFormationEnv(env_params)
    env = LogWrapper(env)
    config["NUM_ACTORS"] = env.num_agents
    if "NUM_UPDATES" not in config:
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    if "LOADDIR" in config:
        action_space = env.action_spaces[env.agents[0]]
        # 提取action_space的形状
        if isinstance(action_space, spaces.Box):
            action_dim = action_space.shape[0]
        elif hasattr(action_space, '__dict__') and hasattr(action_space, 'spaces'):
            # 对于gymnax的Dict类型空间，直接访问其spaces属性
            spaces_dict = action_space.spaces
            total_dim = 0
            for space in spaces_dict.values():
                if hasattr(space, 'n'):  # Discrete空间
                    total_dim += 1
                elif hasattr(space, 'shape'):  # Box空间
                    total_dim += np.prod(space.shape)
                else:
                    raise ValueError(f"不支持的子空间类型: {type(space)}")
            action_dim = total_dim
        else:
            raise ValueError(f"不支持的动作空间类型: {type(action_space)}")
        
        network = ActorCriticRNN(action_dim, config=config)
        rng = jax.random.PRNGKey(42)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"] * config["NUM_ACTORS"], *env.observation_space(env.agents[0], env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        state = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": jnp.array(0)}
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = ckptr.restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))
    else:
        checkpoint = None

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        action_space = env.action_spaces[env.agents[0]]
        # 提取action_space的形状
        if isinstance(action_space, spaces.Box):
            action_dim = action_space.shape[0]
        elif hasattr(action_space, '__dict__') and hasattr(action_space, 'spaces'):
            # 对于gymnax的Dict类型空间，直接访问其spaces属性
            spaces_dict = action_space.spaces
            total_dim = 0
            for space in spaces_dict.values():
                if hasattr(space, 'n'):  # Discrete空间
                    total_dim += 1
                elif hasattr(space, 'shape'):  # Box空间
                    total_dim += np.prod(space.shape)
                else:
                    raise ValueError(f"不支持的子空间类型: {type(space)}")
            action_dim = total_dim
        else:
            raise ValueError(f"不支持的动作空间类型: {type(action_space)}")
        
        network = ActorCriticRNN(action_dim, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"] * config["NUM_ACTORS"], *env.observation_space(env.agents[0], env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        if checkpoint is not None:
            params = checkpoint["params"]
            opt_state = checkpoint["opt_state"]
            train_state = train_state.replace(params=params, opt_state=opt_state)
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = 0
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        # INIT Tensorboard
        if config.get("DEBUG"):
            writer = tensorboardX.SummaryWriter(config["LOGDIR"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            # 使用重物化(rematerialization)来节省内存
            # 在反向传播时重新计算前向传播的中间结果，而不是存储
            @functools.partial(jax.remat, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                ac_in = (
                    last_obs[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, 
                  unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]))
                reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
                done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (
                last_obs[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            # 优化批处理过程中的算法流程，减少冗余计算和内存使用
            def _calculate_gae(traj_batch, last_val):
                # 使用更高效的scan实现，提高内存复用率
                @functools.partial(jax.remat, policy=jax.checkpoint_policies.everything_saveable)
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16
                )
                return advantages, advantages + traj_batch.value
            
            # 使用jax.lax.dynamic_slice切片处理更大的批量
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch = (
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                )
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                shuffled_batch = tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch
                )
                
                # 修改更新过程，采用梯度累积方式减少内存占用
                def _update_minbatch(carry, batch_info):
                    train_state, accumulated_grads, batch_count = carry
                    init_hstate, traj_batch, advantages, targets = batch_info
                    
                    # 提取计算梯度功能作为单独函数，便于重物化
                    @functools.partial(jax.remat, policy=jax.checkpoint_policies.checkpoint_dots)
                    def _compute_gradients(params):
                        def _loss_fn(params):
                            # RERUN NETWORK
                            _, pi, value = network.apply(
                                params,
                                init_hstate.squeeze(0),
                                (traj_batch.obs, traj_batch.done),
                            )
                            log_prob = pi.log_prob(traj_batch.action)

                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = 0.5 * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()

                            # CALCULATE ACTOR LOSS
                            logratio = log_prob - traj_batch.log_prob
                            ratio = jnp.exp(logratio)
                            gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            # debug
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)
                        
                        return jax.grad(_loss_fn, has_aux=True)(params)
                    
                    # 计算当前批次梯度
                    grads, loss_info = _compute_gradients(train_state.params)
                    
                    # 确保累积梯度初始化为与grads结构一致的值
                    if accumulated_grads is None:
                        # 对于scan的第一次迭代，确保初始化为相同结构
                        accumulated_grads = tree_map(lambda x: jnp.zeros_like(x), grads)
                    
                    # 累积梯度
                    accumulated_grads = tree_map(
                        lambda g1, g2: g1 + g2, accumulated_grads, grads
                    )
                    
                    batch_count += 1
                    
                    # 每累积gradient_accumulation_steps个批次应用一次梯度
                    def apply_grads():
                        # 对梯度进行归一化
                        normalized_grads = tree_map(
                            lambda g: g / batch_count, accumulated_grads
                        )
                        # 应用梯度
                        new_train_state = train_state.apply_gradients(grads=normalized_grads)
                        # 返回空梯度，而不是None，确保与accumulated_grads结构一致
                        empty_grads = tree_map(lambda x: jnp.zeros_like(x), accumulated_grads)
                        return new_train_state, empty_grads, 0, loss_info
                    
                    # 继续累积梯度
                    def continue_accumulate():
                        return train_state, accumulated_grads, batch_count, loss_info
                    
                    # 是否应用梯度取决于批次计数或是否为最后一个批次
                    accumulation_steps = config.get("GRADIENT_ACCUMULATION_STEPS", 1)
                    should_apply = (batch_count >= accumulation_steps)
                    
                    train_state, accumulated_grads, batch_count, loss_info = jax.lax.cond(
                        should_apply,
                        apply_grads,
                        continue_accumulate
                    )
                    
                    return (train_state, accumulated_grads, batch_count), loss_info
                
                # 初始化梯度累积状态
                # 不要使用None作为初始梯度，而是创建一个空的梯度结构
                # 这样可以保证scan的输入和输出结构一致
                dummy_params = train_state.params
                
                # 创建一个零初始化的梯度结构，与模型参数结构匹配
                def get_zero_grads(params):
                    # 直接创建与参数相同结构的零梯度
                    return tree_map(lambda x: jnp.zeros_like(x), params)
                
                dummy_grads = get_zero_grads(dummy_params)
                init_carry = (train_state, dummy_grads, 0)
                (train_state, _, _), losses = jax.lax.scan(
                    _update_minbatch, init_carry, minibatches
                )
                
                # 返回更新后的状态和损失
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            # adding an additional "fake" dimensionality to perform minibatching correctly
            initial_hstate = initial_hstate[None, :]
            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            
            # 处理损失信息
            value_loss = loss_info[0].mean()
            actor_loss = loss_info[1].mean()
            entropy = loss_info[2].mean()
            ratio = loss_info[3].mean()
            approx_kl = loss_info[4].mean()
            clip_frac = loss_info[5].mean()
            total_loss = actor_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
            
            metric["loss"] = {
                "total_loss": total_loss,
                "value_loss": value_loss,
                "actor_loss": actor_loss,
                "entropy": entropy,
                "ratio": ratio,
                "approx_kl": approx_kl,
                "clip_frac": clip_frac,
            }

            rng = update_state[-1]
            metric["update_steps"] = update_steps
            if config.get("DEBUG"):
                def callback(metric):
                    env_steps = metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
                    for k, v in metric["loss"].items():
                        writer.add_scalar('loss/{}'.format(k), v, env_steps)
                    writer.add_scalar('eval/episodic_return', metric["returned_episode_returns"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/episodic_length', metric["returned_episode_lengths"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/success_rate', metric["success"][metric["returned_episode"]].mean(), env_steps)
                    print("EnvStep={:<10} EpisodeLength={:<4.2f} Return={:<4.2f} SuccessRate={:.3f}".format(
                        metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                        metric["returned_episode_lengths"][metric["returned_episode"]].mean(),
                        metric["returned_episode_returns"][metric["returned_episode"]].mean(),
                        metric["success"][metric["returned_episode"]].mean(),
                    ))
                jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1    
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
            jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, start_epoch), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    "GROUP": "formation",
    "SEED": 42,
    "LR": 3e-4,
    "NUM_ENVS": 1500,  # 
    "NUM_ACTORS": 2,
    "NUM_STEPS": 1000,
    "TOTAL_TIMESTEPS": 1e9, # 4070tis最多1e8
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "UPDATE_EPOCHS": 8,   # 从16减少到8
    "NUM_MINIBATCHES": 4, # 从5减少到4
    "GRADIENT_ACCUMULATION_STEPS": 16,  # 添加梯度累积步数
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 1e-3,
    "VF_COEF": 1,
    "MAX_GRAD_NORM": 2,
    "ACTIVATION": "relu",
    "ANNEAL_LR": False,
    "DEBUG": True,
    "OUTPUTDIR": "results/" + str_date_time,
    "LOGDIR": "results/" + str_date_time + "/logs",
    "SAVEDIR": "results/" + str_date_time + "/checkpoints",
    # "LOADDIR": "/home/lczh/Git Project/results/2025-04-26-15-31/checkpoints/checkpoint_epoch_62" 
}

seed = config['SEED']
wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
wandb.init(
    # set the wandb project where this run will be logged
    project="AeroPlanax",
    # track hyperparameters and run metadata
    config=config,
    name=f'formation_{str_date_time}',
    group=config['GROUP'],
    notes='2 agents',
    # dir=config['LOGDIR'],
    reinit=True,
)

output_dir = config["OUTPUTDIR"]
Path(output_dir).mkdir(parents=True, exist_ok=True)
save_dir = config["SAVEDIR"]
Path(save_dir).mkdir(parents=True, exist_ok=True)

# 获取训练函数，但不使用jit
rng = jax.random.PRNGKey(seed)
train_fn = make_train(config)

# 改为使用Python的训练循环而不是JAX的scan
# 计算总更新次数
total_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
# 每次运行的更新次数，控制在50次以内
updates_per_run = min(50, total_updates)

# 如果有加载点，从加载点继续训练
if "LOADDIR" in config and config["LOADDIR"]:
    print(f"Loading checkpoint from {config['LOADDIR']}")
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpoint = ckptr.restore(config['LOADDIR'])
    start_epoch = int(checkpoint["epoch"])
    print(f"Starting from epoch {start_epoch}")
else:
    start_epoch = 0
    checkpoint = None

# 存储训练结果
all_metrics = []

# 循环训练
for start_idx in range(start_epoch, total_updates, updates_per_run):
    # 限制本次运行的更新次数
    end_idx = min(start_idx + updates_per_run, total_updates)
    current_updates = end_idx - start_idx
    
    print(f"Running updates {start_idx} to {end_idx-1} (total: {current_updates})")
    
    # 修改config以适应当前运行
    current_config = config.copy()
    current_config["NUM_UPDATES"] = current_updates
    
    # 创建当前运行的训练函数并编译
    current_train_fn = make_train(current_config)
    current_train_jit = jax.jit(current_train_fn)
    
    # 运行训练
    out = current_train_jit(rng)
    
    # 保存结果
    all_metrics.append(out["metric"])
    
    # 更新随机数种子
    rng = jax.random.fold_in(rng, end_idx)
    
    # 保存检查点
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpoint = {
        "params": out['runner_state'][0][0].params,
        "opt_state": out['runner_state'][0][0].opt_state,
        "epoch": jnp.array(end_idx)
    }
    checkpoint_path = os.path.abspath(os.path.join(config["SAVEDIR"], f"checkpoint_epoch_{end_idx}"))
    ckptr.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint))
    ckptr.wait_until_finished()
    print(f"Checkpoint saved at epoch {end_idx}")

wandb.finish()

# 合并指标并绘图
# 这部分需要根据实际的metrics结构进行调整
# 简单示例:
if all_metrics:
    returns = jnp.concatenate([m["returned_episode_returns"].mean(-1).reshape(-1) for m in all_metrics])
    plt.plot(returns)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(output_dir + '/returned_episode_returns.png')
    plt.cla()
    
    lengths = jnp.concatenate([m["returned_episode_lengths"].mean(-1).reshape(-1) for m in all_metrics])
    plt.plot(lengths)
    plt.xlabel("Update Step")
    plt.ylabel("Length")
    plt.savefig(output_dir + '/returned_episode_lengths.png')
else:
    print("No metrics collected")
