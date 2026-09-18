"""
Physics and Degradation Engine Module
Simulates realistic industrial physical behaviors, inter-sensor cross-correlations,
state transitions (NORMAL -> WARNING -> ABNORMAL), and non-random failure generation.
"""

import math
import random
from typing import Dict
from .config import FactoryProfile, OperatingCondition


class IndustrialPhysicsEngine:
    """
    Simulates physics-grounded industrial machinery dynamics with:
    1. First-order autoregressive temporal inertia (smooth physical transitions)
    2. Physical cross-coupling (load -> current -> heat -> vibration -> RPM drop)
    3. Multi-state degradation cycles (NORMAL -> WARNING -> ABNORMAL -> MAINTENANCE OVERHAUL)
    4. Physics-informed failure hazard probability function
    """

    def __init__(self, profile: FactoryProfile, seed: int = 42):
        self.profile = profile
        self.rng = random.Random(seed)

        # Operational state
        self.current_state = OperatingCondition.NORMAL
        self.state_counter = 0
        self.cycle_length = self.rng.randint(850, 1450)  # Time steps in current operational cycle
        self.wear_accumulator = 0.0  # Physical tool/bearing wear (0.0 to 1.0)

        # Failure & maintenance state tracking
        self.active_failure_ticks = 0
        self.maintenance_hold = 0

        # Smoothed sensor state variables (temporal inertia)
        self.current_temp = profile.temperature.mean
        self.current_humidity = profile.humidity.mean
        self.current_vibration = profile.vibration.mean
        self.current_current = profile.current.mean
        self.current_rpm = profile.rpm.mean
        self.current_pressure = profile.pressure.mean

    def _update_operational_state(self):
        """Transitions between NORMAL, WARNING, and ABNORMAL operational conditions."""
        # 1. If currently in maintenance overhaul
        if self.maintenance_hold > 0:
            self.maintenance_hold -= 1
            if self.maintenance_hold == 0:
                # Maintenance complete: machine overhauled, lubricated, re-calibrated
                self.current_state = OperatingCondition.NORMAL
                self.wear_accumulator = self.rng.uniform(0.01, 0.03)
                self.state_counter = 0
                self.cycle_length = self.rng.randint(850, 1450)

                # Reset sensors to pristine nominal baselines
                p = self.profile
                self.current_temp = p.temperature.mean + self.rng.gauss(0, p.temperature.std * 0.25)
                self.current_vibration = p.vibration.mean + self.rng.gauss(0, p.vibration.std * 0.2)
                self.current_current = p.current.mean + self.rng.gauss(0, p.current.std * 0.25)
                self.current_rpm = p.rpm.mean + self.rng.gauss(0, p.rpm.std * 0.25)
                self.current_pressure = p.pressure.mean + self.rng.gauss(0, p.pressure.std * 0.25)
            return

        # 2. If machine is currently experiencing an active failure event
        if self.active_failure_ticks > 0:
            self.active_failure_ticks -= 1
            if self.active_failure_ticks == 0:
                # Failure event concludes -> machine immediately enters maintenance shutdown
                self.maintenance_hold = self.rng.randint(25, 45)
            return

        # 3. Standard operational progression based on duty cycle and wear accumulation
        self.state_counter += 1
        progress = self.state_counter / max(1, self.cycle_length)

        if progress < 0.82:
            self.current_state = OperatingCondition.NORMAL
            self.wear_accumulator += self.profile.base_degradation_rate * self.rng.uniform(0.8, 1.2)
        elif progress < 0.94:
            self.current_state = OperatingCondition.WARNING
            self.wear_accumulator += self.profile.base_degradation_rate * self.rng.uniform(2.2, 3.8)
        else:
            self.current_state = OperatingCondition.ABNORMAL
            self.wear_accumulator += self.profile.base_degradation_rate * self.rng.uniform(6.0, 9.5)

        self.wear_accumulator = min(1.0, self.wear_accumulator)

    def step(self, step_idx: int) -> Dict[str, float]:
        """
        Advance physics simulation by one time-step and calculate next sensor readings.
        Returns sensor values and binary failure label.
        """
        p = self.profile

        # Case 1: Machine is offline / undergoing maintenance overhaul
        if self.maintenance_hold > 0:
            self._update_operational_state()
            # Idle/offline readings
            idle_temp = p.temperature.mean * 0.72 + self.rng.gauss(0, 0.4)
            idle_hum = p.humidity.mean + self.rng.gauss(0, 0.6)
            idle_vib = p.vibration.mean * 0.15 + self.rng.gauss(0, 0.03)
            idle_curr = p.current.mean * 0.12 + self.rng.gauss(0, 0.15)
            idle_rpm = 0.0
            idle_press = p.pressure.mean * 0.50 + self.rng.gauss(0, 0.2)
            return {
                "temperature": round(max(p.temperature.min_val, idle_temp), 2),
                "humidity": round(max(p.humidity.min_val, idle_hum), 2),
                "vibration": round(max(p.vibration.min_val, idle_vib), 3),
                "current": round(max(p.current.min_val, idle_curr), 2),
                "rpm": round(idle_rpm, 1),
                "pressure": round(max(p.pressure.min_val, idle_press), 2),
                "failure": 0,
                "operating_condition": "MAINTENANCE",
                "stress_index": 0.05,
            }

        # Case 2: Machine is actively in failure
        if self.active_failure_ticks > 0:
            self._update_operational_state()
            # In active failure, sensors reflect critical malfunction
            fail_temp = p.temperature.mean * p.temperature.abnormal_mult + self.rng.gauss(0, p.temperature.std * 1.5)
            fail_vib = p.vibration.mean * p.vibration.abnormal_mult + self.rng.gauss(0, p.vibration.std * 1.6)
            fail_curr = p.current.mean * p.current.abnormal_mult + self.rng.gauss(0, p.current.std * 1.4)
            fail_rpm = p.rpm.mean * p.rpm.abnormal_mult + self.rng.gauss(0, p.rpm.std * 2.0)
            fail_press = p.pressure.mean * p.pressure.abnormal_mult + self.rng.gauss(0, p.pressure.std * 1.5)

            # Smooth transitions even during failure
            self.current_temp = 0.85 * self.current_temp + 0.15 * fail_temp
            self.current_vibration = 0.70 * self.current_vibration + 0.30 * fail_vib
            self.current_current = 0.70 * self.current_current + 0.30 * fail_curr
            self.current_rpm = 0.75 * self.current_rpm + 0.25 * fail_rpm
            self.current_pressure = 0.75 * self.current_pressure + 0.25 * fail_press

            return {
                "temperature": round(min(p.temperature.max_val, max(p.temperature.min_val, self.current_temp)), 2),
                "humidity": round(min(p.humidity.max_val, max(p.humidity.min_val, self.current_humidity)), 2),
                "vibration": round(min(p.vibration.max_val, max(p.vibration.min_val, self.current_vibration)), 3),
                "current": round(min(p.current.max_val, max(p.current.min_val, self.current_current)), 2),
                "rpm": round(min(p.rpm.max_val, max(p.rpm.min_val, self.current_rpm)), 1),
                "pressure": round(min(p.pressure.max_val, max(p.pressure.min_val, self.current_pressure)), 2),
                "failure": 1,
                "operating_condition": OperatingCondition.ABNORMAL.value,
                "stress_index": 0.95,
            }

        # Case 3: Operational machine (NORMAL, WARNING, or ABNORMAL approaching failure)
        self._update_operational_state()
        state = self.current_state

        if state == OperatingCondition.NORMAL:
            target_temp = p.temperature.mean + self.rng.gauss(0, p.temperature.std * 0.4)
            target_vib = p.vibration.mean + self.rng.gauss(0, p.vibration.std * 0.3)
            target_curr = p.current.mean + self.rng.gauss(0, p.current.std * 0.35)
            target_rpm = p.rpm.mean + self.rng.gauss(0, p.rpm.std * 0.3)
            target_press = p.pressure.mean + self.rng.gauss(0, p.pressure.std * 0.3)
        elif state == OperatingCondition.WARNING:
            target_temp = p.temperature.mean * p.temperature.warning_mult + self.rng.gauss(0, p.temperature.std * 0.7)
            target_vib = p.vibration.mean * p.vibration.warning_mult + self.rng.gauss(0, p.vibration.std * 0.5)
            target_curr = p.current.mean * p.current.warning_mult + self.rng.gauss(0, p.current.std * 0.5)
            target_rpm = p.rpm.mean * p.rpm.warning_mult + self.rng.gauss(0, p.rpm.std * 0.6)
            target_press = p.pressure.mean * p.pressure.warning_mult + self.rng.gauss(0, p.pressure.std * 0.5)
        else:  # ABNORMAL
            target_temp = p.temperature.mean * (p.temperature.warning_mult + 0.15) + self.rng.gauss(0, p.temperature.std * 1.0)
            target_vib = p.vibration.mean * (p.vibration.warning_mult + 0.8) + self.rng.gauss(0, p.vibration.std * 1.1)
            target_curr = p.current.mean * (p.current.warning_mult + 0.25) + self.rng.gauss(0, p.current.std * 0.9)
            target_rpm = p.rpm.mean * (p.rpm.warning_mult - 0.10) + self.rng.gauss(0, p.rpm.std * 1.2)
            target_press = p.pressure.mean * (p.pressure.warning_mult + 0.10) + self.rng.gauss(0, p.pressure.std * 0.9)

        # Physics Cross-Coupling:
        # 1. High vibration causes friction & thermal buildup
        vib_excess = max(0.0, (self.current_vibration - p.vibration.mean) / max(0.1, p.vibration.std))
        target_temp += vib_excess * 1.5

        # 2. Increased mechanical resistance causes motor current surge
        curr_load_factor = (vib_excess * 0.7) + (self.wear_accumulator * 1.8)
        target_curr += curr_load_factor

        # 3. High load and resistance forces RPM drop
        target_rpm -= (vib_excess * 28.0) + (curr_load_factor * 14.0)

        # 4. Client-specific physical interactions
        if p.client_id == "client_2":
            # Hydraulic press: pressure surges cause shock vibration and fluid heating
            press_deviation = abs(self.current_pressure - p.pressure.mean) / p.pressure.std
            if press_deviation > 1.6:
                target_vib += press_deviation * 0.5
                target_temp += press_deviation * 0.9
        elif p.client_id == "client_3":
            # Compressor: pressure loss causes motor over-current
            if self.current_pressure < (p.pressure.mean * 0.88):
                target_curr += 4.0
                target_temp += 2.8

        # Diurnal ambient cycle
        diurnal_phase = (step_idx * p.sampling_rate_seconds) / 86400.0 * 2.0 * math.pi
        target_humidity = p.humidity.mean + 4.5 * math.sin(diurnal_phase) + self.rng.gauss(0, p.humidity.std * 0.3)
        target_temp += 1.4 * math.sin(diurnal_phase - math.pi / 4.0)

        # Autoregressive thermal & mechanical smoothing
        self.current_temp = 0.90 * self.current_temp + 0.10 * target_temp
        self.current_humidity = 0.90 * self.current_humidity + 0.10 * target_humidity
        self.current_vibration = 0.78 * self.current_vibration + 0.22 * target_vib
        self.current_current = 0.72 * self.current_current + 0.28 * target_curr
        self.current_rpm = 0.82 * self.current_rpm + 0.18 * target_rpm
        self.current_pressure = 0.80 * self.current_pressure + 0.20 * target_press

        # Clipping
        temp_val = max(p.temperature.min_val, min(p.temperature.max_val, self.current_temp))
        hum_val = max(p.humidity.min_val, min(p.humidity.max_val, self.current_humidity))
        vib_val = max(p.vibration.min_val, min(p.vibration.max_val, self.current_vibration))
        curr_val = max(p.current.min_val, min(p.current.max_val, self.current_current))
        rpm_val = max(p.rpm.min_val, min(p.rpm.max_val, self.current_rpm))
        press_val = max(p.pressure.min_val, min(p.pressure.max_val, self.current_pressure))

        # Normalized Machine Stress Index
        norm_temp = max(0.0, (temp_val - (p.temperature.mean + 1.2 * p.temperature.std)) / (p.temperature.std * 2.5))
        norm_vib = max(0.0, (vib_val - (p.vibration.mean + 1.2 * p.vibration.std)) / (p.vibration.std * 2.5))
        norm_curr = max(0.0, (curr_val - (p.current.mean + 1.2 * p.current.std)) / (p.current.std * 2.5))
        norm_rpm = max(0.0, ((p.rpm.mean - 1.2 * p.rpm.std) - rpm_val) / (p.rpm.std * 2.5))
        norm_press = max(0.0, abs(press_val - p.pressure.mean) - 1.2 * p.pressure.std) / (p.pressure.std * 2.5)

        msi = (
            p.weight_temp * min(2.0, norm_temp)
            + p.weight_vib * min(2.0, norm_vib)
            + p.weight_current * min(2.0, norm_curr)
            + p.weight_rpm * min(2.0, norm_rpm)
            + p.weight_pressure * min(2.0, norm_press)
            + 0.25 * self.wear_accumulator
        )

        # Non-random failure trigger
        failure = 0
        if state == OperatingCondition.ABNORMAL:
            # Physical threshold check
            critical_stress = (
                (norm_vib > 1.6 and norm_temp > 1.4)
                or (norm_vib > 1.9)
                or (norm_temp > 1.9 and norm_curr > 1.5)
                or (p.client_id == "client_2" and norm_press > 1.8 and norm_vib > 1.4)
                or (p.client_id == "client_3" and norm_curr > 1.8 and norm_press > 1.5)
            )
            prob_failure = 0.12 * min(1.0, max(0.0, msi - 0.4) * 2.5)

            if critical_stress or (self.rng.random() < prob_failure):
                failure = 1
                # Trigger active breakdown sequence (18-32 consecutive failure ticks)
                self.active_failure_ticks = self.rng.randint(18, 32)

        return {
            "temperature": round(temp_val, 2),
            "humidity": round(hum_val, 2),
            "vibration": round(vib_val, 3),
            "current": round(curr_val, 2),
            "rpm": round(rpm_val, 1),
            "pressure": round(press_val, 2),
            "failure": int(failure),
            "operating_condition": state.value,
            "stress_index": round(msi, 3),
        }
