import torch
import torch.optim as optim
import numpy as np
from gym import Env
from typing import List, Dict

from utils import rollout, relabel_action, BCPolicy
from policy import MakeDeterministic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def simulate_policy_dagger(
    env: Env,
    policy: BCPolicy,
    expert_paths: List[Dict[str, np.ndarray]],
    expert_policy: MakeDeterministic,
    num_epochs: int = 500,
    episode_length: int = 50,
    batch_size: int = 32,
    num_dagger_iters: int = 10,
    num_trajs_per_dagger: int = 10,
):
    # TODO: Fill in your dagger implementation here.

    # Hint: Loop through num_dagger_iters iterations, at each iteration train a policy on the current dataset.
    # Then rollout the policy, use relabel_action to relabel the actions along the trajectory with "expert_policy" and then add this to current dataset
    # Repeat this so the dataset grows with states drawn from the policy, and relabeled actions using the expert.

    # Optimizer code
    optimizer = optim.Adam(list(policy.parameters()))
    losses = []
    returns = []

    trajs = expert_paths
    # Dagger iterations
    for dagger_itr in range(num_dagger_iters):
        idxs = np.array(range(len(trajs)))
        num_batches = len(idxs) * episode_length // batch_size
        losses = []
        # Train the model with Adam
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i in range(num_batches):
                optimizer.zero_grad()
                # TODO start: Fill in your behavior cloning implementation here

                # TODO end
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            # print('[%d, %5d] loss: %.8f' %(epoch + 1, i + 1, running_loss))
            losses.append(loss.item())

        # Collecting more data for dagger
        trajs_recent = []
        for k in range(num_trajs_per_dagger):
            env.reset()
            # TODO start: Rollout the policy on the environment to collect more data, relabel them, add them into trajs_recent

            # TODO end

        trajs += trajs_recent
        mean_return = np.mean(
            np.array([traj["rewards"].sum() for traj in trajs_recent])
        )
        print("Average DAgger return is " + str(mean_return))
        returns.append(mean_return)
