"""
Variant 1: Fine-tune a PPO agent with STOP signs rendered into the existing BEV channel.

Modules:
- env_wrapper: TopDownMetaDriveWithStopSigns — environment with sign support
- reward_wrapper: CustomRewardWrapper — wrapper for custom rewards
"""
from metadrive_core.ppo_w_stop_sign_6ch.env_wrapper import TopDownMetaDriveWithStopSigns

__all__ = [
    'TopDownMetaDriveWithStopSigns',
]
