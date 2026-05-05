"""
Вариант 1: Дообучение PPO агента с отрисовкой знаков STOP в существующем BEV канале

Модули:
- env_wrapper: TopDownMetaDriveWithStopSigns - окружение с поддержкой знаков
- reward_wrapper: CustomRewardWrapper - wrapper для кастомных reward
"""
from metadrive_core.ppo_w_stop_sign_6ch.env_wrapper import TopDownMetaDriveWithStopSigns

__all__ = [
    'TopDownMetaDriveWithStopSigns',
]
