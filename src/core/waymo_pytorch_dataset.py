import os
import numpy as np
import torch
from torch.utils.data import Dataset


class WaymoMotionDataset(Dataset):
    """
    v4: adds the agent type (agent['type']) to the return of
    __getitem__, to allow reporting the loss separately per category
    (Vehicle / Pedestrian / Cyclist) instead of just a general number that
    mixes the three.
    """

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        file_list = [f for f in os.listdir(cache_dir) if f.endswith('.npy')]

        if len(file_list) == 0:
            print(f"WARNING: No file found in {cache_dir}")

        self.samples = []
        for fname in file_list:
            path = os.path.join(cache_dir, fname)
            data = np.load(path, allow_pickle=True).item()
            for agent in data['agents']:
                if agent.get('is_target', False):
                    self.samples.append((path, agent['id']))

        if len(self.samples) == 0 and len(file_list) > 0:
            print(f"WARNING: no agent with is_target=True found in {cache_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, target_id = self.samples[idx]
        data = np.load(file_path, allow_pickle=True).item()

        agent = next((a for a in data['agents'] if a['id'] == target_id), None)
        if agent is None:
            raise RuntimeError(
                f"Agent id={target_id} not found in {file_path}."
            )

        traj = agent['trajectory'].copy()
        mask = agent['mask']

        traj[~mask] = 0.0

        traj_tensor = torch.tensor(traj, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        x_past = traj_tensor[:11, :]
        y_future = traj_tensor[11:, :]
        future_mask = mask_tensor[11:]
        agent_type = torch.tensor(int(agent['type']), dtype=torch.long)

        return x_past, y_future, future_mask, agent_type


if __name__ == "__main__":
    print("OK: WaymoMotionDataset class (v4, with agent type) ready.")