"""
Модуль для проверки пересечения автомобилем сплошной линии в MetaDrive симуляторе.

Этот модуль отслеживает траекторию движения автомобиля и проверяет,
пересекает ли он сплошные линии разметки после проезда.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
from shapely.geometry import LineString, Point
from metadrive.constants import PGLineType, MetaDriveType
from metadrive.component.lane.abs_lane import AbstractLane


class SolidLineDetector:
    """
    Класс для обнаружения пересечения сплошных линий разметки.
    
    Отслеживает траекторию автомобиля и проверяет пересечения
    со сплошными линиями на границах полос.
    """
    
    def __init__(self, trajectory_history_size: int = 1000):
        """
        Инициализация детектора.
        
        Args:
            trajectory_history_size: Максимальный размер истории траектории
        """
        self.trajectory_history = deque(maxlen=trajectory_history_size)
        self.intersections = []
        self.current_lane = None
        self.previous_lane = None
        
    def add_position(self, position, lane: Optional[AbstractLane] = None):
        """
        Добавить позицию автомобиля в историю траектории.
        
        Args:
            position: Позиция автомобиля (x, y) - может быть np.ndarray, Vector, list или tuple
            lane: Текущая полоса, на которой находится автомобиль
        """
        # Конвертируем позицию в numpy array
        if isinstance(position, np.ndarray):
            pos_array = position.copy()
        elif hasattr(position, '__iter__'):
            # Для Vector, list, tuple и других итерируемых объектов
            pos_array = np.array([float(position[0]), float(position[1])])
            if len(position) > 2:
                pos_array = np.array([float(position[0]), float(position[1]), float(position[2])])
        else:
            raise ValueError(f"Неподдерживаемый тип позиции: {type(position)}")
        
        self.trajectory_history.append((pos_array, lane))
        self.previous_lane = self.current_lane
        self.current_lane = lane
    
    def get_lane_boundaries(self, lane: AbstractLane, sample_interval: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Получить границы полосы (левую и правую).
        
        Args:
            lane: Полоса для получения границ
            sample_interval: Интервал дискретизации вдоль полосы (в метрах)
            
        Returns:
            Tuple[left_boundary, right_boundary]: Списки точек левой и правой границ
        """
        if lane is None:
            return [], []
        
        left_boundary = []
        right_boundary = []
        
        # Получаем точки вдоль полосы
        for s in np.arange(0, lane.length, sample_interval):
            width = lane.width_at(s)
            # Левая граница (относительно направления движения)
            left_pos = lane.position(s, lateral=width / 2)
            left_boundary.append(left_pos[:2])
            # Правая граница
            right_pos = lane.position(s, lateral=-width / 2)
            right_boundary.append(right_pos[:2])
        
        # Добавляем последнюю точку
        if lane.length > 0:
            width = lane.width_at(lane.length)
            left_pos = lane.position(lane.length, lateral=width / 2)
            left_boundary.append(left_pos[:2])
            right_pos = lane.position(lane.length, lateral=-width / 2)
            right_boundary.append(right_pos[:2])
        
        return left_boundary, right_boundary
    
    def get_solid_line_segments(self, lane: AbstractLane, sample_interval: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Получить сегменты сплошных линий на границах полосы.
        
        Args:
            lane: Полоса для проверки
            sample_interval: Интервал дискретизации вдоль полосы (в метрах)
            
        Returns:
            List of (start_point, end_point, side): Список сегментов сплошных линий
            side может быть 'left' или 'right'
        """
        if lane is None:
            return []
        
        solid_segments = []
        
        # Проверяем типы линий на границах полосы
        if hasattr(lane, 'line_types'):
            line_types = lane.line_types
        else:
            # Если у полосы нет явных типов линий, считаем что они могут быть сплошными
            line_types = (PGLineType.CONTINUOUS, PGLineType.CONTINUOUS)
        
        # Получаем границы полосы
        left_boundary, right_boundary = self.get_lane_boundaries(lane, sample_interval)
        
        # Проверяем левую границу
        if len(left_boundary) > 1 and self.is_solid_line(line_types[0]):
            for i in range(len(left_boundary) - 1):
                solid_segments.append((
                    np.array(left_boundary[i]),
                    np.array(left_boundary[i + 1]),
                    'left'
                ))
        
        # Проверяем правую границу
        if len(right_boundary) > 1 and self.is_solid_line(line_types[1]):
            for i in range(len(right_boundary) - 1):
                solid_segments.append((
                    np.array(right_boundary[i]),
                    np.array(right_boundary[i + 1]),
                    'right'
                ))
        
        return solid_segments
    
    def is_solid_line(self, line_type) -> bool:
        """
        Проверить, является ли линия сплошной.
        
        Args:
            line_type: Тип линии
            
        Returns:
            True если линия сплошная
        """
        if line_type is None:
            return False
        
        # Проверяем различные представления сплошной линии
        solid_types = [
            PGLineType.CONTINUOUS,
            MetaDriveType.LINE_SOLID_SINGLE_WHITE,
            MetaDriveType.LINE_SOLID_DOUBLE_WHITE,
            MetaDriveType.LINE_SOLID_SINGLE_YELLOW,
            MetaDriveType.LINE_SOLID_DOUBLE_YELLOW,
            PGLineType.SIDE,  # Боковая граница дороги тоже считается сплошной
        ]
        
        return line_type in solid_types or MetaDriveType.is_solid_line(line_type)
    
    def check_intersection(self, 
                          trajectory_segment: Tuple[np.ndarray, np.ndarray],
                          solid_line_segment: Tuple[np.ndarray, np.ndarray]) -> bool:
        """
        Проверить пересечение сегмента траектории со сплошной линией.
        
        Args:
            trajectory_segment: (start_point, end_point) сегмент траектории
            solid_line_segment: (start_point, end_point, side) сегмент сплошной линии
            
        Returns:
            True если есть пересечение
        """
        try:
            traj_start, traj_end = trajectory_segment
            line_start, line_end, _ = solid_line_segment
            
            # Создаем LineString объекты для проверки пересечения
            traj_line = LineString([(traj_start[0], traj_start[1]), 
                                    (traj_end[0], traj_end[1])])
            line_line = LineString([(line_start[0], line_start[1]), 
                                   (line_end[0], line_end[1])])
            
            # Проверяем пересечение
            return traj_line.intersects(line_line)
        except Exception as e:
            # В случае ошибки возвращаем False
            return False
    
    def detect_crossings(self, 
                        current_map=None,
                        all_lanes: Optional[List[AbstractLane]] = None) -> List[Dict]:
        """
        Обнаружить пересечения сплошных линий на основе истории траектории.
        
        Args:
            current_map: Текущая карта (опционально, для получения всех полос)
            all_lanes: Список всех полос для проверки (опционально)
            
        Returns:
            List[Dict]: Список обнаруженных пересечений с информацией о них
        """
        if len(self.trajectory_history) < 2:
            return []
        
        intersections = []
        
        # Получаем все полосы для проверки
        lanes_to_check = []
        if all_lanes is not None:
            lanes_to_check = all_lanes
        elif current_map is not None and hasattr(current_map, 'road_network'):
            # Получаем все полосы из дорожной сети
            try:
                graph = current_map.road_network.graph
                for start_node, end_node, lanes in graph.edges(data='lanes'):
                    if lanes:
                        lanes_to_check.extend(lanes)
            except:
                pass
        
        # Добавляем текущую и предыдущую полосы
        if self.current_lane is not None:
            # Проверяем, что полоса еще не добавлена (сравниваем по id или объекту)
            if not any(lane is self.current_lane or 
                      (hasattr(lane, 'index') and hasattr(self.current_lane, 'index') and 
                       lane.index == self.current_lane.index) 
                      for lane in lanes_to_check):
                lanes_to_check.append(self.current_lane)
        if self.previous_lane is not None:
            if not any(lane is self.previous_lane or 
                      (hasattr(lane, 'index') and hasattr(self.previous_lane, 'index') and 
                       lane.index == self.previous_lane.index) 
                      for lane in lanes_to_check):
                lanes_to_check.append(self.previous_lane)
        
        # Проверяем пересечения для каждой полосы
        for lane in lanes_to_check:
            if lane is None:
                continue
            
            solid_segments = self.get_solid_line_segments(lane)
            
            # Проверяем каждый сегмент траектории с каждым сегментом сплошной линии
            for i in range(len(self.trajectory_history) - 1):
                pos1, _ = self.trajectory_history[i]
                pos2, _ = self.trajectory_history[i + 1]
                
                trajectory_segment = (pos1[:2], pos2[:2])
                
                for solid_segment in solid_segments:
                    if self.check_intersection(trajectory_segment, solid_segment):
                        # Проверяем, не было ли уже зарегистрировано это пересечение
                        # Убеждаемся, что pos2 - это numpy array
                        pos2_array = np.array(pos2[:2]) if not isinstance(pos2, np.ndarray) else pos2[:2].copy()
                        intersection_info = {
                            'position': pos2_array,
                            'lane_id': lane.index if hasattr(lane, 'index') else None,
                            'side': solid_segment[2],
                            'step': i + 1,
                            'timestamp': len(self.trajectory_history) - (len(self.trajectory_history) - i - 1)
                        }
                        
                        # Проверяем, не дублируется ли пересечение
                        is_duplicate = False
                        for existing in intersections:
                            if (np.linalg.norm(existing['position'] - intersection_info['position']) < 2.0 and
                                existing['lane_id'] == intersection_info['lane_id'] and
                                existing['side'] == intersection_info['side']):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            intersections.append(intersection_info)
        
        # Сохраняем новые пересечения
        self.intersections.extend(intersections)
        
        return intersections
    
    def get_all_intersections(self) -> List[Dict]:
        """
        Получить все обнаруженные пересечения.
        
        Returns:
            List[Dict]: Список всех пересечений
        """
        return self.intersections.copy()
    
    def reset(self):
        """Сбросить историю траектории и пересечений."""
        self.trajectory_history.clear()
        self.intersections = []
        self.current_lane = None
        self.previous_lane = None
    
    def get_trajectory(self) -> np.ndarray:
        """
        Получить полную траекторию движения.
        
        Returns:
            np.ndarray: Массив позиций (N, 2)
        """
        if len(self.trajectory_history) == 0:
            return np.array([]).reshape(0, 2)
        
        positions = [pos[:2] for pos, _ in self.trajectory_history]
        return np.array(positions)

