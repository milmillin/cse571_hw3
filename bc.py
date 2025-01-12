import torch
import torch.optim as optim
import numpy as np
from utils import rollout, BCPolicy
from typing import List, Dict
from gym import Env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def simulate_policy_bc(
    env: Env,
    policy: BCPolicy,
    expert_data: List[Dict[str, np.ndarray]],
    num_epochs: int = 500,
    episode_length: int = 50,
    batch_size: int = 32,
):
    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy.
    states = torch.tensor(
        np.array([d["observations"] for d in expert_data]),
        device=device,
        dtype=torch.float32,
    ).flatten(
        0, 1
    )  # (n-run * episode_length, n-dim)
    actions = torch.tensor(
        np.array([d["actions"] for d in expert_data]),
        device=device,
        dtype=torch.float32,
    ).flatten(
        0, 1
    )  # (n-run * episode_length, n-dim)

    optimizer = optim.Adam(list(policy.parameters()))
    idxs = np.array(range(len(expert_data)))
    num_batches = len(idxs) * episode_length // batch_size
    losses = []
    for epoch in range(num_epochs):
        ## TODO Students
        np.random.shuffle(idxs)
        running_loss = 0.0

        idx = torch.randperm(states.shape[0])[:num_batches * batch_size]
        states_ = states[idx].reshape(num_batches, batch_size, -1)
        actions_ = actions[idx].reshape(num_batches, batch_size, -1)

        for i in range(num_batches):
            optimizer.zero_grad()
            # TODO start: Fill in your behavior cloning implementation here
            s_batch = states_[i]
            a_batch = actions_[i]

            a_hat = policy(s_batch)
            loss = ((a_hat - a_batch) ** 2).mean()

            # TODO end
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 50 == 0:
            print("[%d] loss: %.8f" % (epoch, running_loss / 10.0))
        losses.append(loss.item())
