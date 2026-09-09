"""
This script implements the Feudal MAPPO algorithm with the CTDE paradigm, 
where agents within a single cluster share the weights of the Manager, Controller, and Critic.
During training, the centralized Critic uses the averaged global state of the cluster to precisely evaluate
and optimize local decisions made by the Manager (macro-goal selection) and the Controller (micro-action selection).
In the testing phase, the Critic is discarded, and the vehicles operate in a fully decentralized manner,
relying solely on their local observations and learned policies.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from baseline_models import BaseLearningModel
from torch.distributions import Categorical
from routerl import TrafficEnvironment
from utils import (  # type: ignore
    clear_SUMO_files,
    print_agent_counts,
    run_metrics_analysis,
    save_loss_records,
    script_path_for_config,
)

def load_cluster_lookup(cluster_csv_path, key_columns):
    df = pd.read_csv(cluster_csv_path)
    if "cluster" not in df.columns:
        raise ValueError(f"No 'cluster' column in {cluster_csv_path}")
    unique_clusters = sorted(df["cluster"].unique())
    cluster_to_idx = {c: i + 1 for i, c in enumerate(unique_clusters)}
    lookup = {}
    for _, row in df.iterrows():
        key = tuple(row[col] for col in key_columns)
        lookup[key] = cluster_to_idx[row["cluster"]]
    num_clusters = len(unique_clusters) + 1
    return lookup, num_clusters

def build_agent_cluster_map(agents_csv_path, cluster_lookup, key_columns):
    agents_df = pd.read_csv(agents_csv_path)
    cluster_map = {}
    missing = []
    for idx, row in agents_df.iterrows():
        key = tuple(row[col] for col in key_columns)
        if key in cluster_lookup:
            cluster_map[idx] = int(cluster_lookup[key])
        else:
            cluster_map[idx] = 0
            missing.append(idx)
    return cluster_map, missing

def build_mlp_optimizer(module: nn.Module, lr: float) -> optim.Optimizer:
    return optim.Adam(module.parameters(), lr=lr)

@dataclass
class Transition:
    state: np.ndarray          # Lokalna obserwacja (dla Aktorów)
    global_state: np.ndarray   # Stan klastra (dla Krytyka)
    action: int
    log_prob: float
    reward: float
    
class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Categorical:
        logits = self.net(x)
        return Categorical(logits=logits)
    
class ClusterMAPPOAgent(BaseLearningModel):
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        config: dict,
        device: torch.device,
        cluster_id: int = 0,
    ):
        super().__init__()
        self.device = device
        self.cluster_id = int(cluster_id)
        self.action_space_size = int(action_space_size)
 
        # Hyperparametry PPO i Feudal
        self.batch_size = int(config["batch_size"])
        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 0.95))
        
        self.deterministic = False
        self.memory: List[Transition] = []
        self.loss: List[Dict[str, float]] = []
        
        # Zamiast self.manager i self.controller dajemy:
        self.actor = Actor(
            obs_dim=state_size,
            action_dim=self.action_space_size,
            hidden_dims=config.get("actor_hidden_dims", [64, 64]), # upewnij się, że masz to w configu
        ).to(self.device)

        self.critic = CentralizedCritic(
            obs_dim=state_size, 
            hidden_dims=config.get("critic_hidden_dims", [64, 64])
        ).to(self.device)

        self.clip_eps = float(config.get("clip_eps", 0.2)) # standardowy PPO clip eps

        self.optimizer = build_mlp_optimizer(
            nn.ModuleList([self.actor, self.critic]), 
            float(config["lr"]) # zaktualizuj nazwę w configu z manager_lr na lr
        )

    def _to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def push(self, transition: Transition):
        self.memory.append(transition)
    def act(self, local_state: np.ndarray) -> tuple:
        state_tensor = self._to_tensor(local_state)
        
        with torch.no_grad():
            dist = self.actor(state_tensor)
            if self.deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()
                
        return int(action.item()), float(dist.log_prob(action).item())
    
    def compute_gae(self, rewards, values, next_value):
        """Oblicza GAE na podstawie ocen Krytyka."""
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return torch.tensor(advantages, dtype=torch.float32, device=self.device)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = self.memory[:]
        self.memory.clear()
        
        local_states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        global_states = torch.as_tensor(np.stack([b.global_state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor([b.log_prob for b in batch], dtype=torch.float32, device=self.device)
        rewards = [b.reward for b in batch]
        
        values = self.critic(global_states).squeeze()
        advantages = self.compute_gae(rewards, values.tolist(), next_value=0.0)
        returns = advantages + values.detach()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Aktualizacja Aktora
        dist = self.actor(local_states)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Aktualizacja Krytyka
        critic_loss = F.mse_loss(self.critic(global_states).squeeze(), returns)

        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy # 0.01 to przykładowy entropy coef
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.loss.append({
            "actor_loss": float(actor_loss.item()), 
            "critic_loss": float(critic_loss.item())
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--env-conf", type=str, default="clusters")
    parser.add_argument("--task-conf", type=str, required=True)
    parser.add_argument("--alg-conf", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    args = parser.parse_args()

    ALGORITHM = "mappo_dominika"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Torch seed: {torch_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    print("Device is:", device)

    params = {}
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"]

    for key, value in params.items():
        globals()[key] = value

    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, "r", encoding="utf-8") as f:
        data = ast.literal_eval(f.read())
    origins = data["origins"]
    destinations = data["destinations"]

    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, "r", encoding="utf-8") as src, open(new_agents_csv_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        max_start_time = pd.read_csv(new_agents_csv_path)["start_time"].max()
    else:
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}.")

    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps

    cluster_csv_path = None
    if alg_params.get("use_cluster_embedding", False) and alg_params.get("cluster_csv_path"):
        cluster_csv_path = os.path.join(repo_root, alg_params["cluster_csv_path"])
    
    key_columns = alg_params.get("cluster_key_columns", ["start_time", "origin", "destination"])
    agent_cluster_map = {}

    if alg_params.get("use_cluster_embedding", False):
        if cluster_csv_path and os.path.exists(cluster_csv_path):
            cluster_lookup, num_clusters = load_cluster_lookup(cluster_csv_path, key_columns)
            agent_cluster_map, missing_indices = build_agent_cluster_map(agents_csv_path, cluster_lookup, key_columns)
            params["num_clusters"] = num_clusters
        else:
            raise FileNotFoundError(
                f"\n[BŁĄD HRL] 'use_cluster_embedding'jest na true, ale nie znaleziono pliku: '{cluster_csv_path}'. "
                f"Sprawdź ścieżkę w config.json!"
            )
    else:
        params["num_clusters"] = 1

    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config.update(
        {
            "network": network,
            "env_seed": env_seed,
            "torch_seed": torch_seed,
            "env_config": env_config,
            "task_config": task_config,
            "alg_config": alg_config,
            "script": script_path_for_config(__file__),
            "algorithm": ALGORITHM,
            "num_agents": num_agents,
            "num_machines": num_machines,
        }
    )
    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(dump_config, f, indent=4)


    net_abbrs = {
        "saint_arnoult": "sa", 
        "provins": "prov", 
        "ingolstadt_custom": "ingolc", 
        "ingolstadt_custom2": "ingolc2"
    }
    net_abbr = net_abbrs.get(network, network[:5])
    
    group_name = f"{ALGORITHM}_{net_abbr}_{alg_config}"

    wandb.init(
        entity="aintern26coexistence",
        project="PPO Enhancement",
        name=exp_id,
        group=group_name,
        config=dump_config,
    )

    if cluster_csv_path and os.path.exists(cluster_csv_path):
        artifact = wandb.Artifact(name=f"cluster_{alg_config}", type="dataset")
        artifact.add_file(cluster_csv_path)
        wandb.run.log_artifact(artifact)
   

    env = TrafficEnvironment(
        seed=env_seed,
        create_agents=False,
        create_paths=True,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": {
                "model": human_model,
                "alpha": human_alpha,
                "beta": human_beta,
                "beta_randomness": human_beta_randomness,
                "deterministic": human_deterministic,
            },
            "machine_parameters": {
                "behavior": av_behavior,
                "observation_type": "previous_agents_plus_start_time",
            },
        },
        environment_parameters={
            "save_every": save_every,
        },
        simulator_parameters={
            "network_name": network,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo",
            "simulation_timesteps": max_start_time,
        },
        plotter_parameters={
            "phases": phases,
            "phase_names": phase_names,
            "smooth_by": smooth_by,
            "plot_choices": plot_choices,
            "records_folder": records_folder,
            "plots_folder": plots_folder,
        },
        path_generation_parameters={
            "origins": origins,
            "destinations": destinations,
            "number_of_paths": number_of_paths,
            "beta": path_gen_beta,
            "num_samples": num_samples,
            "path_gen_workers": path_gen_workers,
            "visualize_paths": False,
        },
    )

    env.start()
    env.reset()
    print_agent_counts(env)

    pbar = tqdm(total=total_episodes, desc="Human learning")
    for _ in range(human_learning_episodes):
        env.step()
        pbar.update()

    env.mutation(
        disable_human_learning=not should_humans_adapt,
        mutation_start_percentile=-1,
    )
    print_agent_counts(env)

    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    
    unique_clusters = set(agent_cluster_map.values())
    if not unique_clusters:
        unique_clusters = {0}
        
    cluster_models = {}
    for c_id in unique_clusters:
        cluster_models[c_id] = ClusterMAPPOAgent(
            state_size=obs_size,
            action_space_size=env.action_space(env.possible_agents[0]).n,
            config=params,
            device=device,
            cluster_id=c_id
        )

    agent_to_cluster = {}
    for idx in range(len(env.machine_agents)):
        agent_obj = env.machine_agents[idx]
        try:
            agent_int_id = int(str(agent_obj.id).split('_')[-1])
        except:
            agent_int_id = idx
        c_id = agent_cluster_map.get(agent_int_id, 0)
        agent_to_cluster[str(agent_obj.id)] = c_id

    os.makedirs(plots_folder, exist_ok=True)
    pbar.set_description("AV learning MAPPO")
    
    for episode in range(training_eps):
        env.reset()
        episode_rewards = []
        episode_travel_times = []
        
        agent_context = {} 

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            c_id = agent_to_cluster.get(agent_id, 0)
            model = cluster_models[c_id]

            # Obliczanie Global State (Mean Field dla danego klastra)
            cluster_active_agents = [a for a in env.agents if agent_to_cluster.get(a, 0) == c_id]
            if len(cluster_active_agents) > 0:
                cluster_obs = [env.observe(a) for a in cluster_active_agents]
                global_state = np.mean(cluster_obs, axis=0)
            else:
                global_state = observation # Fallback

            # Zrzut danych z poprzedniego kroku (jeśli istnieje)
            if agent_id in agent_context:
                prev = agent_context[agent_id]
                
                transition = Transition(
                    state=prev["state"],
                    global_state=prev["global_state"],
                    action=prev["action"],
                    log_prob=prev["log_prob"],
                    reward=float(reward)
                )
                model.push(transition)

            if termination or truncation:
                reward = float(reward)
                episode_rewards.append(reward)
                if isinstance(info, dict) and "travel_time" in info:
                    episode_travel_times.append(float(info["travel_time"]))
                else:
                    episode_travel_times.append(-reward)
                
                model.learn()
                action = None
                
                if agent_id in agent_context:
                    del agent_context[agent_id]
            else:
                # Wybór nowej akcji - ZWYKŁE MAPPO
                action, log_prob = model.act(observation)
                
                agent_context[agent_id] = {
                    "state": observation.copy(),
                    "global_state": global_state.copy(),
                    "action": action,
                    "log_prob": log_prob,
                }
            
            env.step(action)

        # Logowanie W&B i wykresy po epizodzie
        log_data = {
            "episode": human_learning_episodes + episode,
            "training/reward_sum": float(np.sum(episode_rewards)),
            "training/reward_mean": float(np.mean(episode_rewards)),
            "training/travel_time_mean": float(np.mean(episode_travel_times)),
            "training/travel_time_sum": float(np.sum(episode_travel_times)),
        }
        
        ep_a_loss = [loss["actor_loss"] for m in cluster_models.values() for loss in m.loss if m.loss]
        ep_crit_loss = [loss["critic_loss"] for m in cluster_models.values() for loss in m.loss if m.loss]
        
        if ep_a_loss:
            log_data.update({
                "training/actor_loss": float(np.mean(ep_a_loss)),
                "training/critic_loss": float(np.mean(ep_crit_loss)),
            })
            
        wandb.log(log_data, step=human_learning_episodes + episode)

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()

    for model in cluster_models.values():
        model.deterministic = True
        model.actor.eval()
        model.critic.eval()

    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        episode_rewards = []
        episode_travel_times = []
        agent_context = {}

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                reward = float(reward)
                episode_rewards.append(reward)
                if isinstance(info, dict) and "travel_time" in info:
                    episode_travel_times.append(float(info["travel_time"]))
                else:
                    episode_travel_times.append(-reward)
                action = None
            else:
                c_id = agent_to_cluster.get(agent_id, 0)
                model = cluster_models[c_id]
                
                action, _ = model.act(observation)
                
            env.step(action)

        wandb.log(
            {
                "episode": human_learning_episodes + training_eps + episode,
                "testing/reward_sum": float(np.sum(episode_rewards)),
                "testing/reward_mean": float(np.mean(episode_rewards)),
                "testing/travel_time_mean": float(np.mean(episode_travel_times)),
                "testing/travel_time_sum": float(np.sum(episode_travel_times)),
            },
            step=human_learning_episodes + training_eps + episode,
        )
        pbar.update()

    pbar.close()
    env.plot_results()

    loss_records = []
    for c_id, model in cluster_models.items():
        for iteration, loss_value in enumerate(model.loss, start=1):
            loss_records.append(
                {
                    "iteration": iteration,
                    "cluster_id": c_id,
                    "actor_loss": loss_value["actor_loss"],
                    "critic_loss": loss_value["critic_loss"],
                }
            )
    save_loss_records(
        records_folder,
        loss_records,
        columns=["iteration", "cluster_id", "actor_loss", "critic_loss"],
    )

    env.stop_simulation()
    clear_SUMO_files(
        os.path.join(records_folder, "SUMO_output"),
        os.path.join(records_folder, "episodes"),
        remove_additional_files=True,
    )
    run_metrics_analysis(exp_id, results_folder="../results")
    
    rewards_path = os.path.join(plots_folder, "rewards.png")
    travel_times_path = os.path.join(plots_folder, "travel_times.png")
    plots_to_log = {}
    if os.path.exists(rewards_path):
        plots_to_log["Plots/Rewards"] = wandb.Image(rewards_path)
    if os.path.exists(travel_times_path):
        plots_to_log["Plots/Travel_Times"] = wandb.Image(travel_times_path)
    
    if plots_to_log:
        wandb.log(plots_to_log)
        
    wandb.finish()