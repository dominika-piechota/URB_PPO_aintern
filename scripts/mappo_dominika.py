import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import json
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from routerl         import TrafficEnvironment
from tqdm            import tqdm
from collections     import deque

from baseline_models import BaseLearningModel
from iql             import Network
from utils           import clear_SUMO_files
from utils           import print_agent_counts
from utils           import run_metrics_analysis
from utils           import save_loss_records
from utils           import script_path_for_config
from clustered_routes import ClusteredRoutesLoader, resolve_route_set


class MAPPO(BaseLearningModel):
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: int = 20,
        # --- policy settings ---
        shared_policy: bool = False,
        policy_nets: list[nn.Module] | None = None,
        policy_arch_kwargs: dict | None = None,
        # --- critic settings ---
        share_critic: bool = True,
        critic_nets: list[nn.Module] | None = None,
        critic_arch_kwargs: dict | None = None,
        # --- default architecture parameters ---
        default_widths: list[int] = [32, 64, 32],
        # --- hyperparameters ---
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.0003,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        batch_size: int = 64,
        memory_size: int = 5000,
        device: torch.device | None = None,
        **kwargs
    ):
        super().__init__()
        # device setup 
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.num_agents = num_agents

        # training phase flag
        self.training = True
        
        # hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.last_states = {}
        self.last_actions = {}
        self.last_log_probs = {}

        # architecture args
        ws = default_widths if policy_arch_kwargs is None else policy_arch_kwargs.get('widths', default_widths)

        # --- Policy networks ---
        if policy_nets is not None:
            assert len(policy_nets) == num_agents or shared_policy, \
                "If policy_nets is provided, it must match the number of agents or be shared."
            self.policies = [net.to(self.device) for net in (policy_nets if not shared_policy else [policy_nets[0]] * num_agents)]
        else:
            self.policies = []
            for _ in range(num_agents):
                net = Network(state_size, action_space_size, len(ws) - 1, ws).to(self.device)
                self.policies.append(net)
            if shared_policy:
                self.policies = [self.policies[0]] * num_agents

        # create actor optimizers
        if shared_policy:
            self.actor_optimizer = optim.Adam(self.policies[0].parameters(), lr=lr_actor)
        else:
            self.actor_optimizer = [optim.Adam(policy.parameters(), lr=lr_actor) for policy in self.policies]
        
        # --- Critic networks ---
        if critic_nets is not None:
            assert len(critic_nets) == num_agents or share_critic, \
                "If critic_nets is provided, it must match the number of agents or be shared."
            self.critics = [net.to(self.device) for net in (critic_nets if not share_critic else [critic_nets[0]] * num_agents)]
        else:
            # build critics using generic Network class
            ch_ws = critic_arch_kwargs.get('widths', default_widths) if critic_arch_kwargs else default_widths
            self.critics = []
            for _ in range(num_agents):
                net = Network(state_size, 1, len(ch_ws) - 1, ch_ws).to(self.device)
                self.critics.append(net)
            if share_critic:
                shared_critic = self.critics[0]
                self.critics = [shared_critic] * num_agents

        # create critic optimizers
        if share_critic:
            self.critic_optim = optim.Adam(self.critics[0].parameters(), lr=lr_critic)
        else:
            self.critic_optim = [optim.Adam(net.parameters(), lr=lr_critic) for net in self.critics]

        # loss tracking
        self.loss_actor = []
        self.loss_critic = []

    def train(self):
        """Set all models to training mode"""
        for policy in self.policies:
            policy.train()
        for critic in self.critics:
            critic.train()
        self.training = True
        return self
    
    def eval(self):
        """Set all models to evaluation mode"""
        for policy in self.policies:
            policy.eval()
        for critic in self.critics:
            critic.eval()
        self.training = False
        return self

    def push(self, agent_id: int | float, reward: float | None = None, next_state=None, done: bool = True):
        if reward is None:
            reward = float(agent_id)
            agent_id = 0
        if agent_id not in self.last_states:
            return
        if next_state is None:
            next_state = self.last_states[agent_id]
        self.memory.append(
            (
                self.last_states.pop(agent_id),
                self.last_actions.pop(agent_id),
                float(reward),
                self.last_log_probs.pop(agent_id),
                next_state,
                bool(done),
                agent_id,
            )
        )

    def act(self, state: any, agent_id: int):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policies[agent_id](state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item() if self.training else torch.argmax(logits).item()
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        self.last_states[agent_id] = state
        self.last_actions[agent_id] = action
        self.last_log_probs[agent_id] = log_prob
        return action

    def learn(
        self,
        states: list | None = None,
        actions: list | None = None,
        rewards: list | None = None,
        old_log_probs: list | None = None,
        next_states: list | None = None,
        dones: list | None = None,
        agent_ids: list | None = None
    ):
        if states is not None:
            for s, a, r, lp, ns, d, aid in zip(states, actions, rewards, old_log_probs, next_states, dones, agent_ids):
                self.memory.append((s, a, r, lp, ns, d, aid))

        if len(self.memory) < self.batch_size: return
        
        batch = random.sample(self.memory, self.batch_size)
        s_batch, a_batch, r_batch, lp_batch, ns_batch, d_batch, id_batch = zip(*batch)

        states_tensor = torch.FloatTensor(np.array(s_batch)).to(self.device)
        actions_tensor = torch.LongTensor(a_batch).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(r_batch).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(ns_batch)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(lp_batch).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(d_batch).unsqueeze(1).to(self.device)

        id_tensor = torch.LongTensor(id_batch)
        unique_ids = id_tensor.unique()

        total_policy_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_count = 0

        for aid in unique_ids.tolist():
            mask = (id_tensor == aid)
            states_tensor_a = states_tensor[mask]
            actions_tensor_a = actions_tensor[mask]
            rewards_tensor_a = rewards_tensor[mask]
            next_states_tensor_a = next_states_tensor[mask]
            old_log_probs_tensor_a = old_log_probs_tensor[mask]
            dones_tensor_a = dones_tensor[mask]

            # critic forward
            values = self.critics[aid](states_tensor_a)
            next_values = self.critics[aid](next_states_tensor_a)
            with torch.no_grad():
                targets = rewards_tensor_a + self.gamma * next_values * (1 - dones_tensor_a)
            critic_loss = nn.MSELoss()(values, targets)

            # policy forward
            logits = self.policies[aid](states_tensor_a)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_tensor_a.squeeze(1)).unsqueeze(1)
            ratios = torch.exp(new_log_probs - old_log_probs_tensor_a)
            clipped_ratios = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -torch.min(ratios * (targets - values.detach()), clipped_ratios * (targets - values.detach())).mean()
            entropy = dist.entropy().mean()

            batch_size_a = mask.sum().item()
            total_critic_loss += critic_loss * batch_size_a
            total_policy_loss += policy_loss * batch_size_a
            total_entropy += entropy * batch_size_a
            total_count += batch_size_a

        if total_count == 0: return

        # average losses
        avg_critic_loss = total_critic_loss / total_count
        avg_policy_loss = total_policy_loss / total_count
        avg_entropy = total_entropy / total_count

        # update critic
        if isinstance(self.critic_optim, list):
            for aid in unique_ids.tolist():
                self.critic_optim[aid].zero_grad()
            avg_critic_loss.backward()
            for aid in unique_ids.tolist():
                self.critic_optim[aid].step()
        else:
            self.critic_optim.zero_grad()
            avg_critic_loss.backward()
            self.critic_optim.step()
        self.loss_critic.append(avg_critic_loss.item())

        # update actor
        total_loss = avg_policy_loss - self.entropy_coef * avg_entropy
        if isinstance(self.actor_optimizer, list):
            for aid in unique_ids.tolist():
                self.actor_optimizer[aid].zero_grad()
            total_loss.backward()
            for aid in unique_ids.tolist():
                self.actor_optimizer[aid].step()
        else:
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
        self.loss_actor.append(avg_policy_loss.item())

    def get_last_observation(self, agent_id: int):
        return self.last_states.get(agent_id, None)
    
    def get_last_action(self, agent_id: int):
        return self.last_actions.get(agent_id, None)
    
    def get_last_log_prob(self, agent_id: int):
        return self.last_log_probs.get(agent_id, None)
    
    def get_policy(self, agent_id: int):
        return self.policies[agent_id] if agent_id < len(self.policies) else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="clusters")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument(
        '--route-set',
        type=str,
        default="default_pre_integration",
        help="Named route-set subdirectory. Uses the network default when omitted.",
    )
    parser.add_argument("--shuffle", action="store_true", default=False)
    args = parser.parse_args()

    ALGORITHM = "mappo_dominika"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    requested_route_set = args.route_set
    shuffle = args.shuffle

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"Requested route set: {requested_route_set or 'network default'}")
    print(f"Shuffle: {shuffle}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    print("Device is: ", device)
        
    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], env_params, task_params

    observation_type = params.get(
        "observation_type",
        params.get("observations", "previous_agents_plus_start_time"),
    )
    path_gen_workers_value = params.get("path_gen_workers", 4)

    use_clustered_routes = params.get("use_clustered_routes", False)
    route_set = (
        resolve_route_set(network, requested_route_set)
        if use_clustered_routes
        else None
    )
    print(f"Route set: {route_set or 'none (unclustered)'}")

    # Set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    # Copy agents.csv from custom_network_folder to records_folder
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
        max_start_time = pd.read_csv(new_agents_csv_path)['start_time'].max()
    else:
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}. Please check the network folder.")
            
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()

    # Load pre-generated clustered routes and their per-OD action masks.
    configured_number_of_paths = number_of_paths
    create_paths_flag = True
    action_masks = None

    if use_clustered_routes:
        try:
            route_set_dir = os.path.join(custom_network_folder, "clustered_routes", route_set)
            clustered_loader = ClusteredRoutesLoader(
                network,
                custom_network_folder,
                shuffle,
                env_seed,
                route_set_dir=route_set_dir,
            )
            number_of_paths = clustered_loader.get_number_of_paths()
            clustered_loader.export_paths_routes(records_folder, origins, destinations)
            action_masks = clustered_loader.create_masks(origins, destinations)
            if not action_masks:
                raise ValueError("The clustered route set contains no action masks.")

            for od_pair, mask in action_masks.items():
                mask_array = np.asarray(mask)
                if mask_array.shape != (number_of_paths,):
                    raise ValueError(
                        f"Action mask for OD pair {od_pair} has shape "
                        f"{mask_array.shape}; expected ({number_of_paths},)."
                    )
                if not np.isin(mask_array, (0, 1)).all():
                    raise ValueError(
                        f"Action mask for OD pair {od_pair} must be binary."
                    )
                if not mask_array.any():
                    raise ValueError(
                        f"Action mask for OD pair {od_pair} has no valid actions."
                    )

            agent_ods = {
                (int(row.origin), int(row.destination))
                for row in pd.read_csv(
                    agents_csv_path,
                    usecols=["origin", "destination"],
                ).itertuples()
            }
            missing_ods = sorted(agent_ods.difference(action_masks))
            if missing_ods:
                raise ValueError(
                    "Missing action masks for agent OD pairs: "
                    + ", ".join(map(str, missing_ods))
                )

            create_paths_flag = False
            dump_config["number_of_paths"] = number_of_paths
        except FileNotFoundError as e:
            use_clustered_routes = False
            number_of_paths = configured_number_of_paths
            print(f"[CLUSTERED ROUTES] Warning: {e}")
            print("[CLUSTERED ROUTES] Falling back to JanuX generation\n")

    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["route_set"] = route_set
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["script"] = script_path_for_config(__file__)
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["use_clustered_routes"] = use_clustered_routes
    dump_config["use_action_masks"] = action_masks is not None
    dump_config["shuffle"] = shuffle
    dump_config["observation_type"] = observation_type
    dump_config["path_gen_workers"] = path_gen_workers_value
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    wandb.init(
        entity="aintern26coexistence",
        project="PPO Enhancement",
        name=exp_id,
        config=dump_config
    )

    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = create_paths_flag,
        action_masks = action_masks,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters": {
                "model": human_model,
                "alpha": human_alpha,
                "beta": human_beta,
                "beta_randomness": human_beta_randomness,
                "deterministic": human_deterministic,
            },
            "machine_parameters" : {
                "behavior" : av_behavior,
                "observation_type" : observation_type
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
            "simulation_timesteps" : max_start_time
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "path_gen_workers" : path_gen_workers_value,
            "visualize_paths" : False
        } 
    )

    env.start()
    env.reset()
    print_agent_counts(env)

    ### Human learning phase ###
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()

    # Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    print_agent_counts(env)
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    
    # Set policies for machine agents (Identycznie jak w IPPO!)
    for idx in range(len(env.machine_agents)):
        agent = env.machine_agents[idx]
        mask = None
        if action_masks is not None:
            key = (int(agent.origin), int(agent.destination))
            if key not in action_masks:
                key = (agent.origin, agent.destination)
            if key not in action_masks:
                raise ValueError(
                    f"Missing action mask for agent {agent.id} "
                    f"({agent.origin} -> {agent.destination})."
                )
            mask = action_masks[key]

        # Bezpieczne stworzenie parametrów z JSON dla MAPPO
        model_params = params.copy()
        model_params.update({
            "state_size": obs_size,
            "action_space_size": agent.action_space_size,
            "num_agents": 1,
            "shared_policy": True,
            "share_critic": True,
            "device": device,
            "action_mask": mask
        })
        agent.model = MAPPO(**model_params)

    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}
    
    ### Learning phase ###
    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(training_eps):
        env.reset()
        
        episode_rewards = []
        episode_travel_times = []
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                if agent_id in agent_lookup:
                    agent_lookup[agent_id].model.push(agent_id=0, reward=reward)
                    if episode % update_every == 0:
                        agent_lookup[agent_id].model.learn()
                
                action = None
                episode_rewards.append(reward)
                if "travel_time" in info:
                    episode_travel_times.append(info["travel_time"])
            else:
                if agent_id in agent_lookup:
                    action = agent_lookup[agent_id].model.act(observation, agent_id=0)
                else:
                    action = None
                
            env.step(action)
        
        episode_losses = [agent.model.loss_actor[-1] for agent in env.machine_agents if len(agent.model.loss_actor) > 0]
        
        metrics = {"episode": episode + human_learning_episodes}
        if episode_losses:
            metrics["train/avg_loss"] = sum(episode_losses) / len(episode_losses)
            
        metrics["train/reward_sum"] = float(np.sum(episode_rewards))
        metrics["train/reward_mean"] = float(np.mean(episode_rewards))
        metrics["train/travel_time_mean"] = float(np.mean(episode_travel_times)) if episode_travel_times else 0.0
            
        wandb.log(metrics)
        
        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
    
    ### Testing phase ###
    for agent in env.machine_agents:
        agent.model.eval()
        
    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        
        episode_rewards = []
        episode_travel_times = []
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
                episode_rewards.append(reward)
                if "travel_time" in info:
                    episode_travel_times.append(info["travel_time"])
            else:
                if agent_id in agent_lookup:
                    action = agent_lookup[agent_id].model.act(observation, agent_id=0)
                else:
                    action = None
                
            env.step(action)
            
        wandb.log(
            {
                "episode": human_learning_episodes + training_eps + episode,
                "testing/reward_sum": float(np.sum(episode_rewards)),
                "testing/reward_mean": float(np.mean(episode_rewards)),
                "testing/travel_time_mean": float(np.mean(episode_travel_times)) if episode_travel_times else 0.0,
                "testing/travel_time_sum": float(np.sum(episode_travel_times)) if episode_travel_times else 0.0,
            },
            step=human_learning_episodes + training_eps + episode,
        )
        
        pbar.update()
    
    # Finalize the experiment
    pbar.close()
    env.plot_results()
    
    loss_records = []
    for agent in env.machine_agents:
        for iteration, loss_value in enumerate(agent.model.loss_actor, start=1):
            loss_records.append(
                {
                    "iteration": iteration,
                    "agent_id": agent.id,
                    "loss": loss_value,
                }
            )
    save_loss_records(
        records_folder,
        loss_records,
        columns=["iteration", "agent_id", "loss"],
    )

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    run_metrics_analysis(exp_id, results_folder="../results")
    
    wandb.finish()