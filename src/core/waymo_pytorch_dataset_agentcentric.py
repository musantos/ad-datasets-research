import os
import numpy as np
import torch
from torch.utils.data import Dataset


# --- full_state column layout (verbatim from the .npy contract) ---------------
# full_state has shape [91, 7] and stores, per frame:
#   0:x  1:y  2:length  3:width  4:heading  5:vx  6:vy
# x, y are already SDC-relative (rotated by the SDC heading at frame 10), and
# `heading` is heading_agent - heading_SDC (i.e. already relative to the SDC).
# These indices are the single source of truth for this file; if the
# preprocessor ever changes the layout, the shape assert below fires loudly.
FS_X, FS_Y = 0, 1
FS_LENGTH, FS_WIDTH = 2, 3
FS_HEADING = 4
FS_VX, FS_VY = 5, 6

ANCHOR_FRAME = 10          # last observed frame; split point past|future
N_PAST = 11                # frames [0:11]
N_FUTURE = 80              # frames [11:91]

# Canonical per-frame feature vocabulary. "heading" expands to (sin, cos) by
# default (see heading_as_sincos) to avoid the +/-pi wrap discontinuity; every
# other name is a single channel. The model's input_dim must equal
# `dataset.n_features` -- read it off the instance, never hard-code it.
_SCALAR_FEATURES = {"x", "y", "length", "width", "vx", "vy"}
_ANGLE_FEATURES = {"heading"}


class WaymoMotionDatasetAgentCentric(Dataset):
    """
    Item 4 dataset: agent-centric normalization + rich (full_state) input.

    This is a SIBLING of WaymoMotionDataset (the SDC-centric x,y-only baseline),
    kept as a separate file so the baseline stays byte-for-byte untouched. It is
    parameterized so a single class covers BOTH arms of the item-4 grid
    {SDC-centric, agent-centric} x {rich input}:

      * agent_centric=False, features=("x", "y")
            -> reproduces the baseline __getitem__ EXACTLY (control arm).
      * agent_centric=True,  features=("x","y","heading","vx","vy")
            -> the item-4 treatment: re-center + rotate onto the target agent's
               frame-10 pose and feed heading/velocity, not just position.

    IMPORTANT -- what "agent-centric" does to the TARGET, and why inference must
    invert it:
      The transform is applied to all 91 frames together and only then sliced
      into past/future. That means the regression target y_future is ALSO in the
      agent frame -- this is deliberate; normalizing only the input while leaving
      the target in the SDC frame mixes coordinate frames and defeats the point.
      Consequence: a model trained on this dataset predicts in the agent frame,
      so run_inference_* MUST map predictions back to the SDC frame before
      writing the prediction .npy that validate_motion_official reads. The
      inverse is self-contained: rotate by +theta0 then add p0, where (p0,
      theta0) is the target's frame-10 pose read from full_state[10]. This
      dataset does NOT need to hand those params out (the return stays a 4-tuple,
      so train_* only changes by passing n_features to the model); inference
      recomputes the pose from the same cache file.

    Return of __getitem__ (unchanged arity vs baseline):
        x_past      [11, n_features]  rich, agent-centric (or SDC per the flag)
        y_future    [80, 2]           (x, y) target, same frame as the input
        future_mask [80]              1.0 where the future frame is valid
        agent_type  scalar long       Vehicle / Pedestrian / Cyclist / ...
    """

    def __init__(
        self,
        cache_dir,
        agent_centric=True,
        features=("x", "y", "heading", "vx", "vy"),
        heading_as_sincos=True,
    ):
        self.cache_dir = cache_dir
        self.agent_centric = agent_centric
        self.features = tuple(features)
        self.heading_as_sincos = heading_as_sincos

        # Validate the requested features once, up front (fail loud, fail early).
        for f in self.features:
            if f not in _SCALAR_FEATURES and f not in _ANGLE_FEATURES:
                raise ValueError(
                    f"Unknown feature '{f}'. Valid: "
                    f"{sorted(_SCALAR_FEATURES | _ANGLE_FEATURES)}"
                )

        # n_features: heading contributes 2 channels (sin, cos) unless disabled.
        self.n_features = sum(
            2 if (f in _ANGLE_FEATURES and self.heading_as_sincos) else 1
            for f in self.features
        )

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

    # -- geometry ------------------------------------------------------------
    @staticmethod
    def _rotation_neg(theta):
        """
        R(-theta): maps SDC-frame coords into the agent frame (agent faces +x at
        frame 10). Applied to positions (after translation) and to velocities
        (no translation -- velocity is a free vector).
            x' =  cos*theta * x + sin*theta * y
            y' = -sin*theta * x + cos*theta * y
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, s], [-s, c]], dtype=np.float64)

    def __getitem__(self, idx):
        file_path, target_id = self.samples[idx]
        data = np.load(file_path, allow_pickle=True).item()

        agent = next((a for a in data['agents'] if a['id'] == target_id), None)
        if agent is None:
            raise RuntimeError(f"Agent id={target_id} not found in {file_path}.")

        full_state = np.asarray(agent['full_state'], dtype=np.float64).copy()
        mask = np.asarray(agent['mask'])
        assert full_state.shape == (91, 7), (
            f"full_state shape {full_state.shape} != (91, 7) in {file_path}; "
            f"the .npy layout changed -- re-check the column indices."
        )
        assert mask.shape[0] == 91, f"mask len {mask.shape[0]} != 91."

        xy = full_state[:, [FS_X, FS_Y]].copy()          # [91, 2]
        heading = full_state[:, FS_HEADING].copy()        # [91]
        vel = full_state[:, [FS_VX, FS_VY]].copy()        # [91, 2]

        if self.agent_centric:
            # The frame-10 pose is the anchor; it MUST be a valid observation,
            # otherwise the transform is undefined. Targets are valid at the
            # current time by WOMD construction, so this should never trip --
            # but assert rather than silently normalize onto garbage.
            assert bool(mask[ANCHOR_FRAME]), (
                f"anchor frame {ANCHOR_FRAME} is invalid for target "
                f"{target_id} in {file_path}; cannot build agent-centric frame."
            )
            p0 = xy[ANCHOR_FRAME].copy()                  # [2] origin
            theta0 = float(heading[ANCHOR_FRAME])         # scalar rotation
            R = self._rotation_neg(theta0)

            xy = (xy - p0) @ R.T                           # translate then rotate
            vel = vel @ R.T                                # rotate only
            heading = heading - theta0                     # relative heading

        # --- assemble the rich per-frame feature matrix [91, n_features] ---
        # Column order follows `self.features`; heading expands to (sin, cos).
        cols = []
        for f in self.features:
            if f == "x":
                cols.append(xy[:, 0:1])
            elif f == "y":
                cols.append(xy[:, 1:2])
            elif f == "vx":
                cols.append(vel[:, 0:1])
            elif f == "vy":
                cols.append(vel[:, 1:2])
            elif f == "length":
                cols.append(full_state[:, FS_LENGTH:FS_LENGTH + 1])
            elif f == "width":
                cols.append(full_state[:, FS_WIDTH:FS_WIDTH + 1])
            elif f == "heading":
                if self.heading_as_sincos:
                    cols.append(np.sin(heading)[:, None])
                    cols.append(np.cos(heading)[:, None])
                else:
                    cols.append(heading[:, None])
        feats = np.concatenate(cols, axis=1)              # [91, n_features]

        # Invalid frames: transform FIRST (done above), zero AFTER. This mirrors
        # the baseline's `traj[~mask] = 0` but applied post-transform so a
        # re-centered invalid frame can't leak a spurious position/velocity.
        invalid = ~mask.astype(bool)
        feats[invalid] = 0.0
        xy[invalid] = 0.0

        # --- split past | future ---
        x_past = feats[:N_PAST, :]                         # [11, n_features]
        y_future = xy[N_PAST:, :]                          # [80, 2] target
        future_mask = mask[N_PAST:].astype(np.float32)     # [80]

        return (
            torch.from_numpy(x_past).float(),
            torch.from_numpy(y_future).float(),
            torch.from_numpy(future_mask).float(),
            torch.tensor(int(agent['type']), dtype=torch.long),
        )


if __name__ == "__main__":
    # Minimal self-check: shapes, n_features, and the control-arm equivalence
    # claim. Point CACHE at a real cache_* dir to exercise it end-to-end.
    print("OK: WaymoMotionDatasetAgentCentric class ready.")
    print("Feature-count examples:")
    for feats, sc in [
        (("x", "y"), True),
        (("x", "y", "heading", "vx", "vy"), True),
        (("x", "y", "heading", "vx", "vy"), False),
        (("x", "y", "length", "width", "heading", "vx", "vy"), True),
    ]:
        ds = WaymoMotionDatasetAgentCentric.__new__(WaymoMotionDatasetAgentCentric)
        ds.features = feats
        ds.heading_as_sincos = sc
        n = sum(2 if (f in _ANGLE_FEATURES and sc) else 1 for f in feats)
        print(f"  features={feats} sincos={sc} -> n_features={n}")

    CACHE = os.environ.get("WAYMO_CACHE", "")
    if CACHE and os.path.isdir(CACHE):
        ds = WaymoMotionDatasetAgentCentric(
            CACHE, agent_centric=True,
            features=("x", "y", "heading", "vx", "vy"),
        )
        xp, yf, fm, at = ds[0]
        print(f"[smoke] n_features={ds.n_features} | x_past={tuple(xp.shape)} "
              f"y_future={tuple(yf.shape)} mask={tuple(fm.shape)} type={int(at)}")
        assert xp.shape == (11, ds.n_features)
        assert yf.shape == (80, 2)
        # Control-arm equivalence: agent_centric=False + (x,y) must equal the
        # baseline's first 11 x,y frames (invalid-zeroed).
        base = WaymoMotionDatasetAgentCentric(
            CACHE, agent_centric=False, features=("x", "y"),
        )
        bxp, byf, bfm, bat = base[0]
        assert bxp.shape == (11, 2), bxp.shape
        print(f"[smoke] control-arm x_past={tuple(bxp.shape)} OK")
