#!/usr/bin/env python3
"""
Семплирование параметров для pg maps / MetaDrive на основе статистик nuPlan.
Поддерживает:
- spawn_velocity (начальная скорость)
- accident_prob (вероятность статического препятствия)
- traffic_density (плотность трафика, количество машин на кадр)
- horizon (длительность эпизода, секунды)
- тип машины (s/m/l/xl) на основе геометрии
- NORMAL_SPEED, MAX_SPEED, CREEP_SPEED
- ACC_FACTOR, DEACC_FACTOR
- DISTANCE_WANTED, TIME_WANTED
- LANE_CHANGE_FREQ
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

_DEFAULT_STATS_DIR = Path(__file__).resolve().parent / "nuplan_statistics"


class NuPlanSampler:
    def __init__(self, stats_dir=None):
        """
        stats_dir: папка, содержащая CSV-файлы:
            speeds.csv, acc_pos.csv, acc_neg.csv, following.csv,
            routes.csv, densities.csv, lane_changes.csv (опционально)

        Default: ``core/profiles/nuplan_statistics`` next to this module.
        """
        self.stats_dir = str(stats_dir if stats_dir is not None else _DEFAULT_STATS_DIR)
        self._load_data()
        self._prepare_distributions()

    def _load_data(self):
        """Загрузка CSV-файлов"""
        self.speeds = pd.read_csv(os.path.join(self.stats_dir, 'speeds.csv'))['speed'].dropna().values
        self.acc_pos = pd.read_csv(os.path.join(self.stats_dir, 'acc_pos.csv'))['acceleration'].dropna().values
        self.acc_neg = pd.read_csv(os.path.join(self.stats_dir, 'acc_neg.csv'))['deceleration'].dropna().values
        self.following = pd.read_csv(os.path.join(self.stats_dir, 'following.csv'))['following_distance'].dropna().values
        self.routes = pd.read_csv(os.path.join(self.stats_dir, 'routes.csv'))
        self.densities = pd.read_csv(os.path.join(self.stats_dir, 'densities.csv'))['count'].dropna().values
        # lane_changes – опционально, для частоты смены полос
        lc_path = os.path.join(self.stats_dir, 'lane_changes.csv')
        if os.path.exists(lc_path):
            self.lane_changes = pd.read_csv(lc_path)
        else:
            self.lane_changes = pd.DataFrame()

    def _prepare_distributions(self):
        """Подготовка эмпирических распределений (KDE для непрерывных величин)"""
        # Для скорости и ускорений используем KDE для более гладкого семплирования
        if len(self.speeds) > 10:
            self.speed_kde = gaussian_kde(self.speeds, bw_method='scott')
        else:
            self.speed_kde = None
        if len(self.acc_pos) > 10:
            self.acc_pos_kde = gaussian_kde(self.acc_pos, bw_method='scott')
        else:
            self.acc_pos_kde = None
        if len(self.acc_neg) > 10:
            self.acc_neg_kde = gaussian_kde(self.acc_neg, bw_method='scott')
        else:
            self.acc_neg_kde = None
        if len(self.following) > 10:
            self.following_kde = gaussian_kde(self.following, bw_method='scott')
        else:
            self.following_kde = None

        # Распределение типов машин (s/m/l/xl) на основе размеров из routes (если есть)
        # Если в routes есть колонки length/width – используем их, иначе эмулируем по категории
        if 'length' in self.routes.columns and 'width' in self.routes.columns:
            # Классификация по длине (типичные пороги)
            def classify_size(length, width):
                if length < 4.0:   # очень маленький (мотоцикл, смарт)
                    return 's'
                elif length < 4.8:  # компактный
                    return 's' if width < 1.8 else 'm'
                elif length < 5.2:  # средний (седан)
                    return 'm'
                elif length < 6.0:  # большой (внедорожник, минивэн)
                    return 'l'
                else:               # очень большой (грузовик, автобус)
                    return 'xl'
            sizes = self.routes.apply(lambda row: classify_size(row.get('length', 4.5), row.get('width', 1.8)), axis=1)
            self.size_dist = sizes.value_counts(normalize=True).to_dict()
        else:
            summary_path = os.path.join(self.stats_dir, 'metadrive_config.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    report = json.load(f)
                print("NOT CONST")
                self.size_dist = report.get('size_prob', None)
            else:
                print("CONST")
                self.size_dist = {'m': 0.7, 's': 0.1, 'l': 0.15, 'xl': 0.05}

        # Распределение начальной скорости (spawn_velocity) – из первого кадра каждого трека
        if 'initial_speed' in self.routes.columns:
            self.spawn_speeds = self.routes['initial_speed'].dropna().values
        else:
            # Если нет, используем общее распределение скоростей (но начальные обычно ниже)
            self.spawn_speeds = self.speeds  # fallback

        # Распределение длительности эпизода (horizon) – из routes
        self.horizons = self.routes['duration'].dropna().values

        # Частота смены полос: оценим как число lane_changes на километр
        if len(self.lane_changes) > 0 and 'distance' in self.routes.columns:
            total_distance_km = self.routes['distance'].sum() / 1000.0
            total_lane_changes = len(self.lane_changes)
            self.lane_change_rate_per_km = total_lane_changes / total_distance_km if total_distance_km > 0 else 0.5
        else:
            self.lane_change_rate_per_km = 0.5  # заглушка

        # Вероятность статического препятствия (accident_prob) – доля кадров со static
        summary_path = os.path.join(self.stats_dir, 'metadrive_config.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                report = json.load(f)
            self.accident_prob = report.get('accident_prob', 0.05)
        else:
            self.accident_prob = 0.05

        # Плотность трафика (среднее количество машин на кадр)
        self.traffic_density_mean = np.mean(self.densities) if len(self.densities) > 0 else 5.0

    def sample_continuous(self, kde_obj, default_val=10.0, clip_min=0.0, clip_max=None):
        """Семплирование из KDE (или из эмпирических данных, если KDE не удалось)"""
        if kde_obj is not None:
            sample = kde_obj.resample(1)[0][0]
        else:
            # fallback: используем равномерное распределение
            sample = np.random.uniform(clip_min, clip_max if clip_max else default_val*2)
        # Ограничиваем разумными пределами
        sample = max(clip_min, sample)
        if clip_max is not None:
            sample = min(clip_max, sample)
        return sample

    def spawn_velocity(self):
        """Начальная скорость (м/с)"""
        return np.random.choice(self.spawn_speeds)

    def normal_speed(self):
        """Целевая крейсерская скорость (м/с) – семплируем из распределения скоростей"""
        return self.sample_continuous(self.speed_kde, default_val=12.0, clip_min=0.5, clip_max=30.0)

    def max_speed(self):
        """Максимальная скорость – например, 95-й процентиль из распределения"""
        return np.percentile(self.speeds, 95)

    def creep_speed(self):
        """Минимальная скорость движения – 5-й процентиль"""
        return np.percentile(self.speeds, 5)

    def acc_factor(self):
        """Ускорение (м/с²)"""
        return self.sample_continuous(self.acc_pos_kde, default_val=2.0, clip_min=0.1, clip_max=6.0)

    def deacc_factor(self):
        """Замедление (м/с²) – семплируем из распределения торможений"""
        return self.sample_continuous(self.acc_neg_kde, default_val=3.0, clip_min=0.1, clip_max=8.0)

    def distance_wanted(self):
        """Желаемая дистанция до лидера (м)"""
        return self.sample_continuous(self.following_kde, default_val=15.0, clip_min=1.0, clip_max=60.0)

    def time_wanted(self, speed=None):
        """Временной зазор (секунды) = distance_wanted / speed. Если speed не задана, используем normal_speed"""
        if speed is None:
            speed = self.normal_speed()
        dist = self.distance_wanted()
        return dist / max(speed, 0.5)   # минимум 0.5 м/с

    def lane_change_freq(self):
        """Частота смены полос (событий на километр) – фиксированная из данных"""
        return self.lane_change_rate_per_km

    def horizon(self):
        """Длительность эпизода (секунды)"""
        return np.random.choice(self.horizons)

    def traffic_density(self):
        """Плотность трафика (количество машин на кадр) – семплируем из распределения плотностей"""
        return np.random.choice(self.densities)

    def get_accident_prob(self):
        """Вероятность статического препятствия (бинарная)"""
        return self.accident_prob

    def vehicle_type(self):
        """Тип автомобиля s/m/l/xl"""
        types = list(self.size_dist.keys())
        probs = list(self.size_dist.values())
        return np.random.choice(types, p=probs)

    def sample_all(self):
        """Сгенерировать полный набор параметров для MetaDrive"""
        params = {
            'spawn_velocity': self.spawn_velocity(),
            'accident_prob': self.get_accident_prob(),
            'traffic_density': self.traffic_density(),
            'horizon_seconds': self.horizon(),
            'vehicle_type': self.vehicle_type(),
            'NORMAL_SPEED': self.normal_speed(),
            'MAX_SPEED': self.max_speed(),
            'CREEP_SPEED': self.creep_speed(),
            'ACC_FACTOR': self.acc_factor(),
            'DEACC_FACTOR': self.deacc_factor(),
            'DISTANCE_WANTED': self.distance_wanted(),
            'TIME_WANTED': self.time_wanted(),
            'LANE_CHANGE_FREQ': self.lane_change_freq(),
        }
        return params

# Пример использования
if __name__ == "__main__":
    sampler = NuPlanSampler()

    # Сгенерировать 5 случайных наборов параметров
    print("Генерация 5 наборов параметров для MetaDrive:\n")
    for i in range(5):
        params = sampler.sample_all()
        print(f"--- Набор {i+1} ---")
        for k, v in params.items():
            if isinstance(v, float):
                print(f"{k:20s}: {v:6.2f}")
            else:
                print(f"{k:20s}: {v}")
        print()

    # Сохранить 100 наборов в JSON для последующего использования
    configs = [sampler.sample_all() for _ in range(100)]

    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    configs_serializable = []
    for cfg in configs:
        cfg_serializable = {k: convert_to_serializable(v) for k, v in cfg.items()}
        configs_serializable.append(cfg_serializable)

    with open("metadrive_sampled_configs.json", "w") as f:
        json.dump(configs_serializable, f, indent=2)

    print("\n100 конфигураций сохранены в 'metadrive_sampled_configs.json'")