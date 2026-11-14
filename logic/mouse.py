import time
import math
import threading
import numpy as np
import supervision as sv

from logic.config_watcher import cfg
from logic.visual import visuals
from logic.shooting import shooting
from logic.logger import logger
from logic.platform import input_backend

if cfg.arduino_move or cfg.arduino_shoot:
    from logic.arduino import arduino

if cfg.ch9329_move or cfg.ch9329_shoot:
    from logic.ch9329 import ch9329

# 首帧直锁：目标偏移超过该像素且属于"刚捕获"时，跳过平滑与限速，一帧内全量移动
SNAP_DISTANCE = 40.0
# 两次瞄准之间超过该秒数，视为目标丢失后重新捕获
SNAP_GAP_SECONDS = 0.12

# 卡尔曼滤波参数：状态 [x, y, vx, vy]，恒速模型
KF_PROCESS_NOISE = 180.0    # 目标加速度噪声标准差 (px/s^2)，越大越信任观测
KF_MEASUREMENT_NOISE = 3.0  # 检测框中心观测噪声标准差 (px)，越大越平滑
KF_MAX_DT = 0.15            # 单步时间上限，防止长时间停滞后一步外推过远


class MouseThread:
    def __init__(self):
        self.initialize_parameters()
        self.setup_hardware()

    def initialize_parameters(self):
        self._lock = threading.Lock()
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.disable_prediction = cfg.disable_prediction
        self.prediction_interval = cfg.prediction_interval
        self.bScope_multiplier = cfg.bScope_multiplier
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.prev_x = 0
        self.prev_y = 0
        self.prev_time = None
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2
        self.min_speed_multiplier = cfg.mouse_min_speed_multiplier
        self.max_speed_multiplier = cfg.mouse_max_speed_multiplier
        self.prev_distance = None
        self.speed_correction_factor = 0.1
        self.bScope = False
        self.arch = self.get_arch()
        self.section_size_x = self.screen_width / 100
        self.section_size_y = self.screen_height / 100
        self._warned_input_unavailable = False
        self.subpixel_x = 0.0
        self.subpixel_y = 0.0
        # 后坐力补偿专用子像素累加器（不受 reset_aim_state 影响）
        self.recoil_subpixel_x = 0.0
        self.recoil_subpixel_y = 0.0
        # 后坐力补偿状态
        self.recoil_bullet_count = 0
        self.recoil_time_accumulator = 0.0
        self.recoil_last_fire_time = None
        self.recoil_exhausted = False
        # 预测用速度状态（供 calc_movement 速度感知 EMA 使用）
        self.prev_velocity_x = 0.0
        self.prev_velocity_y = 0.0
        self.smoothed_velocity_mag = 0.0  # 平滑后的速度幅值，避免噪声尖峰导致过冲
        self._last_target_time = 0.0  # 上一次处理目标的时刻（首帧直锁判定用）
        # 卡尔曼滤波器状态（None 表示尚未初始化/已重置）
        self._kf_state = None
        self._kf_cov = None

    def get_arch(self):
        if cfg.AI_enable_AMD:
            return f'hip:{cfg.AI_device}'
        ai_device = str(cfg.AI_device).lower()
        if 'cpu' in ai_device:
            return 'cpu'
        if ai_device == 'cuda':
            return 'cuda'
        if ai_device.startswith('cuda:'):
            return cfg.AI_device
        return f'cuda:{cfg.AI_device}'

    def setup_hardware(self):
        pass

    def process_data(self, data):
        with self._lock:
            self._process_data_inner(data)

    def _process_data_inner(self, data):
        if isinstance(data, sv.Detections):
            xyxy = data.xyxy[0]
            target_x = (xyxy[0] + xyxy[2]) / 2
            target_y = (xyxy[1] + xyxy[3]) / 2
            target_w = xyxy[2] - xyxy[0]
            target_h = xyxy[3] - xyxy[1]
            target_cls = data.class_id[0] if data.class_id is not None and data.class_id.size > 0 else None
        else:
            target_x, target_y, target_w, target_h, target_cls = data

        shooting_state = self.get_shooting_key_state()
        self.visualize_target(target_x, target_y, target_cls)
        self.bScope = self.check_target_in_scope(target_x, target_y, target_w, target_h, self.bScope_multiplier) if cfg.auto_shoot or cfg.triggerbot else False
        self.bScope = cfg.force_click or self.bScope

        if not self.disable_prediction:
            current_time = time.time()
            if not isinstance(data, sv.Detections):
                target_x, target_y = self.predict_target_position(target_x, target_y, current_time)
            self.visualize_prediction(target_x, target_y, target_cls)

        move_x, move_y, normalized_dist = self.calc_movement(target_x, target_y, target_cls)

        # 后坐力补偿：开火时根据武器曲线叠加下压/左右修正
        move_x, move_y = self.apply_recoil_compensation(move_x, move_y, shooting_state)

        # 瞄准调试日志
        if cfg.log_debug_aim and shooting_state:
            offset_x = target_x - self.center_x
            offset_y = target_y - self.center_y
            distance = math.sqrt(offset_x**2 + offset_y**2)
            cls_name = {1: "head", 2: "player"}.get(target_cls, str(target_cls))
            rounded_x = round(move_x)
            rounded_y = round(move_y)
            will_move = not (abs(rounded_x) <= cfg.mouse_dead_zone and abs(rounded_y) <= cfg.mouse_dead_zone)
            print(
                f"[瞄准] 目标: ({target_x:.0f}, {target_y:.0f}) "
                f"类别: {cls_name} "
                f"大小: {target_w:.0f}x{target_h:.0f} | "
                f"偏移: ({offset_x:.0f}, {offset_y:.0f}) "
                f"距离: {distance:.0f} | "
                f"鼠标移动: ({move_x:.1f}, {move_y:.1f}) → 取整: ({rounded_x}, {rounded_y}) "
                f"| {'发送' if will_move else '死区内,跳过'}"
            )

        self.visualize_history(target_x, target_y)
        shooting.submit(self.bScope, shooting_state)
        # 后坐力补偿正在开火时，即使未按瞄准热键也强制发送鼠标移动
        recoil_firing = cfg.recoil_enable and (shooting.button_pressed or input_backend.is_pressed("left"))
        self.move_mouse(move_x, move_y, shooting_state, normalized_dist, force=recoil_firing)

    def _kf_reset_filter(self, x, y):
        """用当前观测重新初始化卡尔曼滤波器（首次锁定或目标跳变时调用）"""
        self._kf_state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        # 初始位置不确定度取观测噪声，速度不确定度取较大值（未知初速度）
        self._kf_cov = np.diag([
            KF_MEASUREMENT_NOISE ** 2,
            KF_MEASUREMENT_NOISE ** 2,
            400.0,
            400.0,
        ])

    def _kf_update(self, target_x, target_y, delta_time):
        """
        卡尔曼滤波：恒速模型的状态预测 + 观测修正。
        相比两帧差分，多帧递归估计速度，能抑制检测框抖动带来的速度尖峰。
        """
        dt = delta_time
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q2 = KF_PROCESS_NOISE ** 2

        # 状态转移矩阵 F（恒速模型）
        transition = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # 过程噪声 Q（连续加速度模型的离散化形式）
        process_cov = q2 * np.array([
            [dt4 / 4, 0.0, dt3 / 2, 0.0],
            [0.0, dt4 / 4, 0.0, dt3 / 2],
            [dt3 / 2, 0.0, dt2, 0.0],
            [0.0, dt3 / 2, 0.0, dt2],
        ])

        self._kf_state = transition @ self._kf_state
        self._kf_cov = transition @ self._kf_cov @ transition.T + process_cov

        # 观测矩阵 H（只观测位置）与观测噪声 R
        observation = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        measurement_cov = np.eye(2) * (KF_MEASUREMENT_NOISE ** 2)

        innovation = np.array([target_x, target_y]) - observation @ self._kf_state
        innovation_cov = observation @ self._kf_cov @ observation.T + measurement_cov
        gain = self._kf_cov @ observation.T @ np.linalg.inv(innovation_cov)

        self._kf_state = self._kf_state + gain @ innovation
        identity = np.eye(4)
        self._kf_cov = (identity - gain @ observation) @ self._kf_cov

    def predict_target_position(self, target_x, target_y, current_time):
        if math.isnan(target_x) or math.isnan(target_y):
            return target_x, target_y

        # First target
        if self._kf_state is None or self.prev_time is None:
            self._kf_reset_filter(target_x, target_y)
            self.prev_time = current_time
            self.prev_x, self.prev_y = target_x, target_y
            self.prev_velocity_x = 0.0
            self.prev_velocity_y = 0.0
            return target_x, target_y

        delta_time = current_time - self.prev_time
        if delta_time <= 0:
            delta_time = 1e-3
        delta_time = min(delta_time, KF_MAX_DT)

        # 目标跳变（换人/误检）：相对滤波估计位置的跳变超过画面 30%
        max_jump = max(self.screen_width, self.screen_height) * 0.3
        if (abs(target_x - self._kf_state[0]) > max_jump
                or abs(target_y - self._kf_state[1]) > max_jump):
            self._kf_reset_filter(target_x, target_y)
            self.prev_time = current_time
            self.prev_x, self.prev_y = target_x, target_y
            self.prev_velocity_x = 0.0
            self.prev_velocity_y = 0.0
            return target_x, target_y

        self._kf_update(target_x, target_y, delta_time)

        # 预测时刻 = 当前时刻 + prediction_interval 个循环周期
        horizon = delta_time * self.prediction_interval
        predicted_x = self._kf_state[0] + self._kf_state[2] * horizon
        predicted_y = self._kf_state[1] + self._kf_state[3] * horizon

        # 预测偏移钳位: 单帧预测不超过配置上限，防止异常速度导致大幅跳跃
        max_pred_offset = cfg.prediction_max_offset
        pred_dx = predicted_x - target_x
        pred_dy = predicted_y - target_y
        pred_mag = math.sqrt(pred_dx ** 2 + pred_dy ** 2)
        if pred_mag > max_pred_offset:
            scale = max_pred_offset / pred_mag
            predicted_x = target_x + pred_dx * scale
            predicted_y = target_y + pred_dy * scale

        frame_displacement = math.sqrt((target_x - self.prev_x) ** 2 + (target_y - self.prev_y) ** 2)
        self.prev_x, self.prev_y = target_x, target_y
        self.prev_velocity_x = float(self._kf_state[2])
        self.prev_velocity_y = float(self._kf_state[3])
        self.prev_time = current_time
        self.prev_distance = frame_displacement

        return predicted_x, predicted_y

    def calculate_speed_multiplier(self, target_x, target_y, distance):
        if any(map(math.isnan, (target_x, target_y))):
            return self.min_speed_multiplier

        normalized_distance = min(distance / self.max_distance, 1)
        # 远→快（快速 acquisition），近→慢（精确 tracking）
        base_speed = self.min_speed_multiplier + (self.max_speed_multiplier - self.min_speed_multiplier) * normalized_distance

        if self.prev_distance is not None:
            speed_adjustment = 1 + (abs(distance - self.prev_distance) / self.max_distance) * self.speed_correction_factor
            base_speed *= speed_adjustment

        # 移动目标速度驱动加速（使用平滑后的速度，避免噪声尖峰）
        velocity_mag = math.sqrt(self.prev_velocity_x ** 2 + self.prev_velocity_y ** 2)
        # EMA 平滑：α=0.3 表示 30% 新值 + 70% 历史值，约 3 帧收敛
        self.smoothed_velocity_mag = 0.3 * velocity_mag + 0.7 * self.smoothed_velocity_mag
        velocity_correction = min(2.0, self.smoothed_velocity_mag / 250.0)  # 500px/s 时达到最大 ×3.0
        base_speed *= (1.0 + velocity_correction)

        return base_speed

    def calc_movement(self, target_x, target_y, target_cls):
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y
        distance = math.sqrt(offset_x**2 + offset_y**2)
        speed_multiplier = self.calculate_speed_multiplier(target_x, target_y, distance)

        # 首帧直锁：目标刚被捕获（或丢失后重新出现）且偏移较大时，
        # 跳过 EMA 平滑与速度倍率，用原始几何量一帧内移动到位
        now = time.time()
        acquiring = (now - self._last_target_time) > SNAP_GAP_SECONDS and distance >= SNAP_DISTANCE
        self._last_target_time = now

        degrees_per_pixel_x = self.fov_x / self.screen_width
        degrees_per_pixel_y = self.fov_y / self.screen_height

        mouse_move_x = offset_x * degrees_per_pixel_x
        mouse_move_y = offset_y * degrees_per_pixel_y

        # 动态平滑: 距离 + 速度联合自适应
        # 目标移动越快 alpha 越高（减少旧值拖拽），近距离 alpha 高（快速追踪）
        normalized_dist = min(distance / self.max_distance, 1.0)
        velocity_mag = math.sqrt(self.prev_velocity_x ** 2 + self.prev_velocity_y ** 2)
        velocity_factor = min(1.0, velocity_mag / 800.0)  # 800px/s 时达到最大
        # 距离分量贡献 [0, 0.04]，速度分量贡献 [0, 0.02]，总范围 [0.92, 0.98]
        alpha = min(0.98, 0.92 + normalized_dist * 0.04 + velocity_factor * 0.02)  # → [0.92, 0.98]

        if acquiring:
            alpha = 1.0
            speed_multiplier = 1.0

        if not hasattr(self, 'last_move_x'):
            self.last_move_x, self.last_move_y = 0, 0

        move_x = alpha * mouse_move_x + (1 - alpha) * self.last_move_x
        move_y = alpha * mouse_move_y + (1 - alpha) * self.last_move_y

        self.last_move_x, self.last_move_y = move_x, move_y

        sensitivity = max(self.mouse_sensitivity, 1e-6)
        move_x = (move_x / 360) * (self.dpi * (1 / sensitivity)) * speed_multiplier
        move_y = (move_y / 360) * (self.dpi * (1 / sensitivity)) * speed_multiplier

        return move_x, move_y, normalized_dist

    def apply_recoil_compensation(self, move_x, move_y, shooting_state):
        """
        根据武器后坐力曲线叠加补偿量（帧率无关）。
        曲线值含义: 每发子弹的**总补偿量**（鼠标计数），运行时按 DPI/灵敏度换算。
        每帧按实际经过时间分配一个比例，保证不同帧率下总补偿一致。
        """
        if not cfg.recoil_enable:
            if self.recoil_bullet_count > 0:
                self.recoil_bullet_count = 0
                self.recoil_time_accumulator = 0.0
                self.recoil_last_fire_time = None
                self.recoil_subpixel_x = 0.0
                self.recoil_subpixel_y = 0.0
            return move_x, move_y

        pattern = cfg.recoil_pattern
        if not pattern:
            return move_x, move_y

        # 检测是否正在开火：自动射击系统已按下 或 用户物理按下鼠标左键
        is_firing = shooting.button_pressed or input_backend.is_pressed("left")

        if not is_firing:
            self.recoil_bullet_count = 0
            self.recoil_time_accumulator = 0.0
            self.recoil_last_fire_time = None
            self.recoil_subpixel_x = 0.0
            self.recoil_subpixel_y = 0.0
            self.recoil_exhausted = False
            return move_x, move_y

        # 曲线已执行完毕，等待松键重置
        if getattr(self, 'recoil_exhausted', False):
            return move_x, move_y

        # 计时
        current_time = time.time()
        if self.recoil_last_fire_time is None:
            self.recoil_last_fire_time = current_time
            return move_x, move_y

        delta = min(current_time - self.recoil_last_fire_time, 0.05)  # 上限 50ms，防止掉帧后跳跃
        self.recoil_last_fire_time = current_time
        self.recoil_time_accumulator += delta

        # 子弹间隔: 优先用 recoil_duration 均匀分配，否则用 fire_rate
        if cfg.recoil_duration > 0 and len(pattern) > 0:
            bullet_interval = cfg.recoil_duration / len(pattern)
        else:
            bullet_interval = 1.0 / max(cfg.recoil_fire_rate, 1.0)

        while self.recoil_time_accumulator >= bullet_interval and self.recoil_bullet_count < len(pattern):
            self.recoil_time_accumulator -= bullet_interval
            self.recoil_bullet_count += 1

        # 曲线执行完毕，标记停止
        if self.recoil_bullet_count >= len(pattern):
            self.recoil_exhausted = True
            return move_x, move_y

        # 查表获取当前子弹的补偿值
        idx = self.recoil_bullet_count - 1
        if idx < 0:
            return move_x, move_y

        base_dx, base_dy = pattern[idx]

        # 按实际 DPI/灵敏度换算（基准: DPI=800, sensitivity=1.0）
        scale = (self.dpi / 800.0) * (1.0 / max(self.mouse_sensitivity, 1e-6)) * cfg.recoil_scale

        # 帧率无关: 曲线值 = 每发总量，按经过时间占比分配
        progress = delta / bullet_interval
        recoil_x = base_dx * scale * progress
        recoil_y = base_dy * scale * progress

        move_x += recoil_x
        move_y += recoil_y

        if cfg.log_debug_aim:
            print(f"[后坐力] 第{self.recoil_bullet_count}发 补偿: ({recoil_x:.1f}, {recoil_y:.1f}) progress={progress:.3f}")

        return move_x, move_y

    def process_recoil_only(self):
        """无目标时独立执行后坐力补偿（对墙扫射、准星偏离目标等场景）
        使用专用子像素累加器，不受 reset_aim_state 和瞄准死区影响。
        """
        with self._lock:
            self._process_recoil_only_inner()

    def _process_recoil_only_inner(self):
        if not cfg.recoil_enable:
            return

        is_firing = shooting.button_pressed or input_backend.is_pressed("left")
        if not is_firing:
            if self.recoil_bullet_count > 0:
                self.recoil_bullet_count = 0
                self.recoil_time_accumulator = 0.0
                self.recoil_last_fire_time = None
                self.recoil_subpixel_x = 0.0
                self.recoil_subpixel_y = 0.0
                self.recoil_exhausted = False
            return

        move_x, move_y = self.apply_recoil_compensation(0, 0, True)

        # 使用专用子像素累加器（不被 reset_aim_state 清零）
        self.recoil_subpixel_x += move_x
        self.recoil_subpixel_y += move_y

        send_x = round(self.recoil_subpixel_x)
        send_y = round(self.recoil_subpixel_y)

        # 后坐力补偿不使用瞄准死区 — 这些是刻意的补偿量，不是检测噪声
        if send_x == 0 and send_y == 0:
            return

        self.recoil_subpixel_x -= send_x
        self.recoil_subpixel_y -= send_y

        self._dispatch_mouse(send_x, send_y)

    def move_mouse(self, x, y, shooting_state, normalized_dist=0.0, force=False):
        # 子像素累积：保留小数部分，避免远距离微小移动被取整丢弃
        self.subpixel_x += x
        self.subpixel_y += y

        move_x = round(self.subpixel_x)
        move_y = round(self.subpixel_y)

        # 距离自适应死区：远距离目标框小，需要更精细的修正，死区缩小
        # 近距离目标框大，死区保持正常防止抖动
        # normalized_dist: 0=中心(近) → 1=边缘(远)
        effective_dead_zone = max(1, round(cfg.mouse_dead_zone * (1.0 - normalized_dist * 0.75)))

        if abs(move_x) <= effective_dead_zone and abs(move_y) <= effective_dead_zone:
            return

        # 发送成功后扣除已发送的整数部分，保留残余给下一帧
        self.subpixel_x -= move_x
        self.subpixel_y -= move_y

        if force or shooting_state or cfg.mouse_auto_aim:
            self._dispatch_mouse(move_x, move_y)

    def _dispatch_mouse(self, move_x, move_y):
        """底层鼠标移动发送（纯硬件分发，不做子像素/死区处理）"""
        if cfg.ch9329_move:
            from logic.ch9329 import ch9329
            if not ch9329.is_connected():
                if not getattr(self, '_warned_ch9329', False):
                    logger.error("[鼠标] CH9329 未连接，鼠标移动指令不会被执行！请检查 USB 连接和串口配置。")
                    self._warned_ch9329 = True
                return
            ch9329.move(move_x, move_y)
        elif cfg.arduino_move:
            from logic.arduino import arduino
            arduino.move(move_x, move_y)
        else:
            if not input_backend.move_mouse(move_x, move_y):
                if not self._warned_input_unavailable:
                    logger.warning("[鼠标] 原生鼠标移动后端在当前平台不可用。")
                    self._warned_input_unavailable = True

    def get_shooting_key_state(self):
        for key_name in cfg.hotkey_targeting_list:
            if input_backend.is_pressed(key_name.strip()):
                return True
        return False

    def check_target_in_scope(self, target_x, target_y, target_w, target_h, reduction_factor):
        reduced_w, reduced_h = target_w * reduction_factor / 2, target_h * reduction_factor / 2
        x1, x2, y1, y2 = target_x - reduced_w, target_x + reduced_w, target_y - reduced_h, target_y + reduced_h
        bScope = self.center_x > x1 and self.center_x < x2 and self.center_y > y1 and self.center_y < y2

        if cfg.show_window and cfg.show_bScope_box:
            visuals.draw_bScope(x1, x2, y1, y2, bScope)

        return bScope

    def reset_aim_state(self):
        """重置瞄准状态（目标丢失或切换时调用，防止旧数据影响新目标）"""
        with self._lock:
            self.last_move_x = 0
            self.last_move_y = 0
            self.prev_time = None
            self.prev_distance = None
            self.subpixel_x = 0.0
            self.subpixel_y = 0.0
            # 重置预测速度，防止目标丢失后旧速度影响新目标
            self.prev_velocity_x = 0.0
            self.prev_velocity_y = 0.0
            self.smoothed_velocity_mag = 0.0  # 重置平滑速度，避免旧目标速度影响新目标
            self._last_target_time = 0.0  # 下次检测到目标时重新触发首帧直锁
            self._kf_state = None  # 丢弃旧目标的卡尔曼状态
            self._kf_cov = None
            # 注意: 不重置后坐力状态 — 后坐力由 apply_recoil_compensation 自行管理
            # （停止开火时自动归零，持续开火时不应因目标丢失而中断曲线进度）

    def update_settings(self):
        with self._lock:
            self.dpi = cfg.mouse_dpi
            self.mouse_sensitivity = cfg.mouse_sensitivity
            self.fov_x = cfg.mouse_fov_width
            self.fov_y = cfg.mouse_fov_height
            self.disable_prediction = cfg.disable_prediction
            self.prediction_interval = cfg.prediction_interval
            self.bScope_multiplier = cfg.bScope_multiplier
            self.screen_width = cfg.detection_window_width
            self.screen_height = cfg.detection_window_height
            self.center_x = self.screen_width / 2
            self.center_y = self.screen_height / 2
            self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2
            self.min_speed_multiplier = cfg.mouse_min_speed_multiplier
            self.max_speed_multiplier = cfg.mouse_max_speed_multiplier
            self.section_size_x = self.screen_width / 100
            self.section_size_y = self.screen_height / 100
            self.arch = self.get_arch()
            self.prev_time = None
            self.prev_distance = None
            self.prev_velocity_x = 0.0
            self.prev_velocity_y = 0.0
            self._last_target_time = 0.0
            self._kf_state = None
            self._kf_cov = None
            self.setup_hardware()

    def visualize_target(self, target_x, target_y, target_cls):
        if (cfg.show_window and cfg.show_target_line) or (cfg.show_overlay and cfg.overlay_show_target_line):
            visuals.draw_target_line(target_x, target_y, target_cls)

    def visualize_prediction(self, target_x, target_y, target_cls):
        if (cfg.show_window and cfg.show_target_prediction_line) or (cfg.show_overlay and cfg.overlay_show_target_prediction_line):
            visuals.draw_predicted_position(target_x, target_y, target_cls)

    def visualize_history(self, target_x, target_y):
        if (cfg.show_window and cfg.show_history_points) or (cfg.show_overlay and cfg.show_history_points):
            visuals.draw_history_point_add_point(target_x, target_y)

mouse = MouseThread()
