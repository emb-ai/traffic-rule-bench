#!/usr/bin/env python3
"""
Fine-tune BEV PPO agent with stop signs (pdd_bench).
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback


SCRIPT_DIR = Path(__file__).resolve().parent
PDD_BENCH_DIR = SCRIPT_DIR.parents[3]
SDC_ROOT = PDD_BENCH_DIR.parent
METADRIVE_DIR = SDC_ROOT / "metadrive"
for path in (PDD_BENCH_DIR, METADRIVE_DIR, SDC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from metadrive_core.pdd_bench_bev_ppo.bev_ft_train_signs import (  # noqa: E402
    create_env,
)
from metadrive_core.bev_cnn import CustomBEVCNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BEV PPO agent with stop signs (pdd_bench)."
    )
    parser.add_argument(
        "--pretrained",
        default=str(
            PDD_BENCH_DIR
            / "checkpoints"
            / "pdd_bench"
            / "bev_ppo_training_eval"
            / "signs_5000k"
            / "ppo_v1_draw_signs.zip"
        ),
        help="Path to pretrained model (.zip).",
    )
    parser.add_argument(
        "--finetune-dir",
        default=str(PDD_BENCH_DIR / "outputs" / "bev_ppo_finetune"),
        help="Directory to save fine-tuned checkpoints/logs.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5_000_000,
        help="Total timesteps for fine-tuning.",
    )
    parser.add_argument(
        "--traffic-density",
        type=float,
        default=0.25,
        help="Traffic density.",
    )
    parser.add_argument(
        "--stop-sign-prob",
        type=float,
        default=0.3,
        help="Probability of adding stop signs.",
    )
    parser.add_argument(
        "--stop-sign-penalty",
        type=float,
        default=-10.0,
        help="Penalty for stop sign violations.",
    )
    parser.add_argument(
        "--custom-reward-weight",
        type=float,
        default=1.0,
        help="Weight for custom reward wrapper.",
    )
    parser.add_argument(
        "--speed-reward-coef",
        type=float,
        default=3.0,
        help="Override speed reward coefficient (None to skip).",
    )
    parser.add_argument(
        "--driving-reward-coef",
        type=float,
        default=4.0,
        help="Override driving reward coefficient (None to skip).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=32,
        help="Number of parallel training envs.",
    )
    parser.add_argument(
        "--eval-envs",
        type=int,
        default=64,
        help="Number of parallel eval envs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Device for training ("cuda" or "cpu"). Auto if omitted.',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--no-custom-reward",
        action="store_true",
        help="Disable custom reward wrapper.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    finetune_path = Path(args.finetune_dir)
    finetune_path.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA device count: {torch.cuda.device_count()}")
    else:
        print("CUDA not available, using CPU")

    use_custom_reward = not args.no_custom_reward

    print("=" * 70)
    print(f"Loading pretrained model: {args.pretrained}")
    print(f"Finetuning for: {args.timesteps:,} timesteps")
    print(f"Stop sign penalty: {args.stop_sign_penalty}")
    print(f"Output path: {finetune_path}")
    print(f"Device: {device}")
    print("=" * 70)

    train_env = SubprocVecEnv(
        [
            lambda: create_env(
                need_monitor=True,
                debug=False,
                traffic_density=args.traffic_density,
                use_custom_reward=use_custom_reward,
                custom_reward_weight=args.custom_reward_weight,
                add_stop_signs=True,
                stop_sign_probability=args.stop_sign_prob,
                stop_sign_penalty=args.stop_sign_penalty,
                speed_reward_coef=args.speed_reward_coef,
                driving_reward_coef=args.driving_reward_coef,
            )
            for _ in range(args.num_envs)
        ]
    )

    try:
        model = PPO.load(
            args.pretrained,
            env=train_env,
            device=device,
            custom_objects={"policy_kwargs": dict(features_extractor_class=CustomBEVCNN)},
        )
        model.batch_size = 512
        model.learning_rate = model.learning_rate / 2
        model.tensorboard_log = str(finetune_path)
        print("Pretrained model loaded successfully!")
        print(f"n_steps: {model.n_steps}")
        print(f"batch_size: {model.batch_size}")
        print(f"n_epochs: {model.n_epochs}")
    except Exception as exc:
        print(f"Error loading model: {exc}")
        print("Training new model from scratch...")
        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            n_steps=4096,
            batch_size=512,
            n_epochs=10,
            verbose=1,
            device=device,
            learning_rate=1e-3,
            policy_kwargs=dict(features_extractor_class=CustomBEVCNN),
            tensorboard_log=str(finetune_path),
        )

    eval_env = SubprocVecEnv(
        [
            lambda: create_env(
                need_monitor=True,
                debug=False,
                traffic_density=args.traffic_density,
                use_custom_reward=use_custom_reward,
                custom_reward_weight=args.custom_reward_weight,
                add_stop_signs=True,
                stop_sign_probability=args.stop_sign_prob,
                stop_sign_penalty=args.stop_sign_penalty,
                speed_reward_coef=args.speed_reward_coef,
                driving_reward_coef=args.driving_reward_coef,
            )
            for _ in range(args.eval_envs)
        ]
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(finetune_path, "best_model"),
        log_path=os.path.join(finetune_path, "eval_logs"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1,
    )

    callback_list = eval_callback

    print(f"Fine-tuning for: {args.timesteps:,} steps")
    model.learn(
        total_timesteps=args.timesteps,
        log_interval=100,
        callback=callback_list,
        reset_num_timesteps=False,
    )

    final_model_path = finetune_path / "ppo_v1_draw_signs"
    model.save(str(final_model_path))
    print(f"Model saved to: {final_model_path}")

    train_env.close()
    eval_env.close()

    print("\n" + "=" * 70)
    print("All done!")
    print(f"TensorBoard: {finetune_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
