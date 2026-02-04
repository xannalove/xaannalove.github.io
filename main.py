import copy
import os
import random
from datetime import datetime
import warnings
import numpy as np
from memory import ReplayBuffer
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from torch.utils.tensorboard import SummaryWriter
import env
from MATD3 import MATD3Agent
from Game import GameAgent
from CooperativeGame import CooperativeGameAgent

class Args:
    """Configuration for the agent."""

    def __init__(self):
        # 智能体参数
        self.seed = 3
        self.n_agents: int = 5
        self.local_state_dim = 2
        self.global_state_dim = self.n_agents
        self.action_dim = 1
        self.action_bounds = (7, 9)
        self.noise_params = {
            'policy': 0.2,
            'clip': 0.5,
            'action_std': 0.5,
        }
        # 算法参数
        self.target_net_iteration: int = 2
        self.gamma: float = 0.99
        self.tau: float = 0.005
        # num = 3,    4         4     num = 5,    16        5     num = 10,   512       170    num = 20,   524288    1000
        self.shapley_sample_times: int = 10
        self.learning_rates = {
            'actor': 3e-4,
            'critic': 3e-4,
            'alliance': 3e-4
        }
        # 训练循环参数
        self.episode: int = 3000
        self.step: int = 10
        self.frequency: int = 100

        self.memory_capacity: int = 10000
        self.batch_size: int = 256

        # 算法选择
        # self.algorithm: str = "MATD3"
        # self.algorithm: str = "Game"
        self.algorithm: str = "CooperativeGame"

def get_timestamp(args: Args) -> str:
    """生成带共享标识的时间戳"""
    return f"{args.n_agents}-agent-{datetime.now():%m-%d_%H-%M-%S}-SHARED-{args.algorithm}"

def get_agent_class(algorithm: str):
    """获取智能体类并验证算法有效性"""
    agent_map = {
        "MATD3": MATD3Agent,
        "Game": GameAgent,
        "CooperativeGame": CooperativeGameAgent
    }
    if algorithm not in agent_map:
        raise ValueError(f"无效算法，请从 {list(agent_map.keys())} 中选择")
    return agent_map[algorithm]


class NetworkManager:
    def __init__(self, num_agents, local_state_dim, global_state_dim, action_dim, action_bounds, with_mask, device):
        self.num_agents = num_agents
        self.device = device
        self.episode_counter = 0
        self.share_interval = 10

        self.max_loss_history = 1
        # 记录每个智能体的actor_loss
        self.agent_losses = {i: [] for i in range(num_agents)}

        # 初始化每个智能体的独立网络
        self.networks = {}
        for i in range(num_agents):
            self.networks[i] = self._create_network(
                local_state_dim, global_state_dim, action_dim,
                action_bounds, with_mask, device
            )

        self.performance_metrics = {
            'best_edge_loss': float('inf'),
            'best_middle_loss': float('inf'),
            'last_share_time': None
        }

    def update_metrics(self, best_edge_loss, best_middle_loss):
        """只在有有效loss时更新指标"""
        if best_edge_loss != float('inf') or best_middle_loss != float('inf'):
            if best_edge_loss != float('inf'):
                self.performance_metrics['best_edge_loss'] = min(self.performance_metrics['best_edge_loss'],
                                                                 best_edge_loss)
            if best_middle_loss != float('inf'):
                self.performance_metrics['best_middle_loss'] = min(self.performance_metrics['best_middle_loss'],
                                                                   best_middle_loss)
            self.performance_metrics['last_share_time'] = datetime.now()

    def update_loss(self, agent_id, actor_loss):
        """更新智能体的actor_loss记录并维护固定长度"""
        if actor_loss is not None:
            self.agent_losses[agent_id].append(actor_loss)
            # 限制历史记录长度
            if len(self.agent_losses[agent_id]) > self.max_loss_history:
                self.agent_losses[agent_id] = self.agent_losses[agent_id][-self.max_loss_history:]

    def _get_best_agent(self, agent_ids):
        """从指定的智能体中选择actor_loss最close 0的"""
        losses = {}
        for agent_id in agent_ids:
            # 确保有足够的记录
            if not self.agent_losses[agent_id]:
                continue

            # 计算最近1回合的actor_loss
            recent_losses = self.agent_losses[agent_id][-1]
            losses[agent_id] = abs(recent_losses)

        if not losses:  # 如果没有有效的loss记录
            return None

        # 返回actor_loss最接近0的智能体ID
        return min(losses.items(), key=lambda x: abs(x[1]))[0]

    def _create_network(self, local_state_dim, global_state_dim, action_dim, action_bounds, with_mask, device):
        """为每个智能体创建独立的网络"""
        from model import Actor, Critic, Alliance

        actor = Actor(local_state_dim, action_bounds[0], action_bounds[1]).to(device)
        critic = Critic(global_state_dim, action_dim, self.num_agents, with_mask).to(device)

        # 创建优化器
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

        networks = {
            'actor': actor,
            'actor_target': copy.deepcopy(actor),
            'critic': critic,
            'critic_target': copy.deepcopy(critic),
            'actor_optimizer': actor_optimizer,
            'critic_optimizer': critic_optimizer,
        }

        # 只有非MATD3算法才需要Alliance网络
        if with_mask:
            alliance = Alliance(global_state_dim, self.num_agents).to(device)
            alliance_optimizer = torch.optim.Adam(alliance.parameters(), lr=3e-4)
            networks.update({
                'alliance': alliance,
                'alliance_target': copy.deepcopy(alliance),
                'alliance_optimizer': alliance_optimizer
            })

        return networks

    def get_networks(self, agent_id):
        """获取指定智能体的网络"""
        return self.networks[agent_id]

    def _share_parameters(self, from_agent_id, to_agent_id, tau=0.005):
        """使用软更新方式共享actor网络的参数"""
        try:
            # 使用软更新方式更新actor网络参数
            for target_param, source_param in zip(self.networks[to_agent_id]['actor'].parameters(),
                                                  self.networks[from_agent_id]['actor'].parameters()):
                target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

            # 使用软更新方式更新actor_target网络参数
            for target_param, source_param in zip(self.networks[to_agent_id]['actor_target'].parameters(),
                                                  self.networks[from_agent_id]['actor_target'].parameters()):
                target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

            # 重新创建actor优化器并复制状态
            new_optimizer = type(self.networks[from_agent_id]['actor_optimizer'])(
                self.networks[to_agent_id]['actor'].parameters(),
                **self.networks[from_agent_id]['actor_optimizer'].defaults
            )

            # 如果源优化器有状态，复制状态
            if self.networks[from_agent_id]['actor_optimizer'].state_dict()['state']:
                new_optimizer.load_state_dict(
                    self.networks[from_agent_id]['actor_optimizer'].state_dict()
                )

            self.networks[to_agent_id]['actor_optimizer'] = new_optimizer

            print(f"Successfully soft-updated actor parameters from agent {from_agent_id} to agent {to_agent_id} with tau={tau}")
        except Exception as e:
            print(f"Error soft-updating actor parameters from agent {from_agent_id} to agent {to_agent_id}: {str(e)}")

    def share_if_needed(self):
        """检查是否需要共享参数并执行共享"""
        self.episode_counter += 1

        if self.episode_counter % self.share_interval == 0:
            # 处理首尾智能体的参数共享
            edge_agents = [0, self.num_agents - 1]
            best_edge_agent = self._get_best_agent(edge_agents)
            if best_edge_agent is not None:  # 只在有有效loss记录时进行共享
                other_edge_agent = edge_agents[1] if best_edge_agent == 0 else edge_agents[0]
                self._share_parameters(best_edge_agent, other_edge_agent)
                edge_loss = self.agent_losses[best_edge_agent][-1]
                print(
                    f"Edge agents shared: Best agent {best_edge_agent} (avg loss: {edge_loss:.4f}) shared to agent {other_edge_agent}")
            else:
                print("Skipping edge agents sharing due to insufficient loss records")

            # 处理中间智能体的参数共享
            middle_agents = list(range(1, self.num_agents - 1))
            if middle_agents:  # 确保有中间智能体
                best_middle_agent = self._get_best_agent(middle_agents)
                if best_middle_agent is not None:  # 只在有有效loss记录时进行共享
                    middle_loss = self.agent_losses[best_middle_agent][-1]
                    for agent_id in middle_agents:
                        if agent_id != best_middle_agent:
                            self._share_parameters(best_middle_agent, agent_id)
                    print(
                        f"Middle agents shared: Best agent {best_middle_agent} (avg loss: {middle_loss:.4f}) shared to others")
                    # 更新性能指标
                    self.update_metrics(edge_loss if 'edge_loss' in locals() else float('inf'), middle_loss)
                else:
                    print("Skipping middle agents sharing due to insufficient loss records")

def main():
    args = Args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestamp = get_timestamp(args)
    paths = {
        'log': f"./reward/{timestamp}/",
        'model': f"./weight/{timestamp}/"
    }
    os.makedirs(paths['model'], exist_ok=True)
    writer = SummaryWriter(log_dir=paths['log'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_instance = env.MultiAgentEnv(agent_num=args.n_agents, render=False)
    agent_class = get_agent_class(args.algorithm)

    if args.algorithm == "MATD3":
        with_mask = False
    else:
        with_mask = True

    network_manager = NetworkManager(
        num_agents=args.n_agents,
        local_state_dim=args.local_state_dim,
        global_state_dim=args.global_state_dim,
        action_dim = args.action_dim,
        action_bounds=args.action_bounds,
        with_mask=with_mask,
        device=device
    )

    def init_agents():
        """初始化智能体"""
        multi_agent = []

        for i in range(args.n_agents):
            agent = agent_class(
                args.local_state_dim,
                args.global_state_dim,
                args.action_dim,
                args,
                device,
                i,
                network_manager,
            )
            multi_agent.append(agent)

        # 设置所有智能体的引用
        for agent in multi_agent:
            agent.all_agents = multi_agent

        return multi_agent, network_manager

    # 初始化智能体
    multi_agent, agent_network_manager = init_agents()

    # 加载模型和经验
    for i in range(args.n_agents):
        model_files = {
            'actor': paths['model'] + f'actor_{i}',
            'critic': paths['model'] + f'critic_{i}',
            'alliance': paths['model'] + f'alliance_{i}'
        }

        # 加载模型
        if os.path.exists(model_files['actor'] + '.pt'):
            multi_agent[i].load_model(model_files['actor'], model_files['critic'], model_files['alliance'])

    print(f"Using algorithm: {args.algorithm}")

    # 训练循环
    i_episode = 0
    start_recording_episode = 0  # Start recording from this episode
    recording_offset = start_recording_episode

    while i_episode < args.episode:
        env_instance.reset()
        obs, state = env_instance.get_obs()
        # print("obs", obs)
        # print("state", state)
        total_reward = [0] * args.n_agents

        for current_step in range(args.step):
            joint_action = []
            for i, agent in enumerate(multi_agent):
                local_obs = obs[i * args.local_state_dim:(i + 1) * args.local_state_dim]
                # print("local_obs", local_obs)
                action = agent.select_action(local_obs, is_training=True).item()
                joint_action.append(action)

            next_local_obs, next_global_state, rewards, linearity, done, safety = env_instance.step(joint_action)
            # print("next_global_state", next_global_state)

            # 存储经验和训练
            for i, agent in enumerate(multi_agent):
                agent.memory.put(obs, state, joint_action, rewards, next_local_obs, next_global_state, done)
                if args.algorithm == "MATD3":
                    actor_loss, critic_loss = agent.learn(i, i_episode)
                    if i_episode >= start_recording_episode:
                        if actor_loss is not None:
                            network_manager.update_loss(i, actor_loss)
                            writer.add_scalar(f'agent_{i}/actor_loss', actor_loss, i_episode - recording_offset)
                        if critic_loss is not None:
                            writer.add_scalar(f'agent_{i}/critic_loss', critic_loss, i_episode - recording_offset)
                else:
                    actor_loss, critic_loss, Alliance_loss = agent.learn(i, i_episode)
                    # 记录损失（仅当i_episode >= start_recording_episode时）
                    if i_episode >= start_recording_episode:
                        if actor_loss is not None:
                            network_manager.update_loss(i, actor_loss)
                            writer.add_scalar(f'agent_{i}/actor_loss', actor_loss, i_episode - recording_offset)
                            writer.add_scalar(f'agent_{i}/critic_loss', critic_loss, i_episode - recording_offset)
                            if args.algorithm == "CooperativeGame" and Alliance_loss is not None:
                                writer.add_scalar(f'agent_{i}/Alliance_loss', Alliance_loss, i_episode - recording_offset)

                total_reward[i] += rewards[i]

            if done:
                break

            obs = next_local_obs
            state = next_global_state

        # 记录奖励（仅当i_episode >= start_recording_episode时）
        if i_episode >= start_recording_episode:
            for i in range(args.n_agents):
                writer.add_scalar(f'agent_{i}/reward', total_reward[i], i_episode - recording_offset)
            writer.add_scalar('real_linearity', linearity, i_episode - recording_offset)
            writer.add_scalar('safety_reward', safety, i_episode - recording_offset)

        network_manager.share_if_needed()

        i_episode += 1
        print(f"Episode {i_episode}/{args.episode} completed. ")

        # 保存模型和经验
        if (i_episode + 1) % args.frequency == 0:
            for i in range(args.n_agents):
                if args.algorithm == "CooperativeGame":
                    multi_agent[i].save_model(
                        paths['model'] + f'actor_{i}',
                        paths['model'] + f'critic_{i}',
                        paths['model'] + f'alliance_{i}'
                    )
                else:
                    multi_agent[i].save_model(
                        paths['model'] + f'actor_{i}',
                        paths['model'] + f'critic_{i}',
                    )
            print("Models and experiences saved")

    writer.close()

if __name__ == "__main__":
    main()

