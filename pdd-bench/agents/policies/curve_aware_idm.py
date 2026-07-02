"""Curve-aware base IDM — ego-бейзлайн с оборонительным слоем «как у sumoidm».

Голый MetaDrive IDMPolicy рулит с превью 1 м и не знает о кривизне дороги —
на скоростях 40+ км/ч он не вписывается в повороты OSM-карт (OOR у
idm_default 43%). При этом NPC (SumoTrajectoryIDMPolicy) и rule-эксперт
(ComprehensiveRuleExpertPolicy) давно возят с кривизна-капом скорости,
расширенным lookahead руления и торможением перед пересекающим трафиком.

Этот класс даёт БАЗОВОМУ idm ровно тот же оборонительный слой, что у
эксперта — методы и константы переиспользуются из
ComprehensiveRuleExpertPolicy напрямую (не копией), чтобы не разъезжались.
Знаний о знаках здесь НЕТ: пара idm/idm_rule теперь отличается только
sign-compliance, а не управляемостью — зазор пары становится чистым.

Откат: EGO_CURVE_AWARE=0 в окружении возвращает сырой IDMPolicy
(см. bench/policy_factory.py).
"""
from __future__ import annotations

from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.idm_policy import IDMPolicy

from agents.policies.comprehensive_rule_expert import ComprehensiveRuleExpertPolicy as _EXPERT


class CurveAwareIDMPolicy(IDMPolicy):
    # Константы оборонительного слоя — те же, что у эксперта (единый источник).
    STEERING_LOOKAHEAD_PG = _EXPERT.STEERING_LOOKAHEAD_PG
    STEERING_LOOKAHEAD_SUMO = _EXPERT.STEERING_LOOKAHEAD_SUMO
    MAX_LONG_DIST = _EXPERT.MAX_LONG_DIST
    SAFE_LANE_CHANGE_DISTANCE = _EXPERT.SAFE_LANE_CHANGE_DISTANCE
    INTERSECTION_SCAN_RADIUS = _EXPERT.INTERSECTION_SCAN_RADIUS
    INTERSECTION_HALF_ANGLE = _EXPERT.INTERSECTION_HALF_ANGLE
    CURVATURE_LOOK_AHEAD = _EXPERT.CURVATURE_LOOK_AHEAD
    CURVATURE_MU_PG = _EXPERT.CURVATURE_MU_PG
    CURVATURE_MU_SUMO = _EXPERT.CURVATURE_MU_SUMO
    CURVATURE_MIN_SPEED = _EXPERT.CURVATURE_MIN_SPEED

    # Переиспользуем реализацию эксперта (обычные функции/дескрипторы —
    # работают на инстансах этого класса, sign-логика не затягивается).
    _detect_sumo = _EXPERT._detect_sumo
    STEERING_LOOKAHEAD = _EXPERT.STEERING_LOOKAHEAD          # property (PG/SUMO)
    CURVATURE_MU = _EXPERT.CURVATURE_MU                      # property (PG/SUMO)
    steering_control = _EXPERT.steering_control              # lookahead 2–3 м
    _curvature_target_speed = _EXPERT._curvature_target_speed
    _find_crossing_obstacle = _EXPERT._find_crossing_obstacle

    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed)
        # Те же PID-гейны, что у эксперта/NPC (совпадают с базовыми, но
        # фиксируем явно — единая настройка пары).
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, 0.002, 0.05)
        self._is_sumo = self._detect_sumo()

    def acceleration(self, front_obj, dist_to_front) -> float:
        # (1) Кап скорости по кривизне впереди. Врезка именно здесь: базовый
        # act() перезаписывает target_speed в lane-change ветках ДО вызова
        # acceleration, так что клампим последним словом.
        try:
            cap = self._curvature_target_speed()
            if cap < self.target_speed:
                self.target_speed = cap
        except Exception:
            pass
        # (2) Оборонительное торможение перед пересекающим трафиком на
        # перекрёстке (зеркало эксперта/SumoTrajectoryIDMPolicy).
        try:
            if dist_to_front is None or dist_to_front > self.INTERSECTION_SCAN_RADIUS:
                obj, dist = self._find_crossing_obstacle()
                if obj is not None and (dist_to_front is None or dist < dist_to_front):
                    front_obj, dist_to_front = obj, dist
        except Exception:
            pass
        return super().acceleration(front_obj, dist_to_front)
