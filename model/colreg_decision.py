import math
import numpy as np

class ColregDecision:

    def calc_dcpa_tcpa(self, ego_ship, target_ship):
        # 상대 위치 (dx, dy)를 미터 단위로 변환
        dx = (target_ship.x - ego_ship.x)
        dy = (target_ship.y - ego_ship.y)

        # 선박의 속도 벡터 계산 (속도는 이미 m/s 단위이므로 변환 불필요)
        ego_vx = ego_ship.speed * math.sin(math.radians(ego_ship.heading))
        ego_vy = ego_ship.speed * math.cos(math.radians(ego_ship.heading))
        target_vx = target_ship.speed * math.sin(math.radians(target_ship.heading))
        target_vy = target_ship.speed * math.cos(math.radians(target_ship.heading))

        # 상대 속도 벡터
        dvx = target_vx - ego_vx
        dvy = target_vy - ego_vy

        # 상대 속도 크기
        dv = dvx ** 2 + dvy ** 2

        if dv == 0:
            # 상대 속도가 0이면 충돌이 없고, DCPA는 현재 거리로 설정
            dcpa = math.sqrt(dx ** 2 + dy ** 2)
            tcpa = float('inf')
            return dcpa, tcpa

        # TCPA 계산 (시간)
        tcpa = -(dx * dvx + dy * dvy) / dv

        # TCPA에서의 두 선박 위치 계산 (픽셀 정보를 미터 단위로 변환)
        closest_x_ego = ego_ship.x + ego_vx * tcpa
        closest_y_ego = ego_ship.y + ego_vy * tcpa
        closest_x_target = target_ship.x + target_vx * tcpa
        closest_y_target = target_ship.y + target_vy * tcpa

        # DCPA 계산 (거리)
        dcpa = math.sqrt((closest_x_target - closest_x_ego) ** 2 + (closest_y_target - closest_y_ego) ** 2)

        return dcpa, tcpa

    def calc_rel_bearing(self, ego_x, ego_y, target_x, target_y, ego_heading):
        dx = target_x - ego_x
        dy = (target_y) - ego_y
        angle_to_target = math.degrees(math.atan2(dy, dx))

        # 상대적인 각도와 heading(방향)에 따른 조우 상황 판단
        relative_bearing = (90 - (angle_to_target)) - ego_heading

        if relative_bearing > 360:
            relative_bearing -= 360
        if relative_bearing < 0:
            relative_bearing += 360

        return relative_bearing

    def calc_abs_bearing(self, ego_x, ego_y, target_x, target_y):
        dx = target_x - ego_x
        dy = (target_y) - ego_y
        angle_to_target = math.degrees(math.atan2(-dy, dx))

        # 상대적인 각도와 heading(방향)에 따른 조우 상황 판단
        # abs_bearing = angle_to_target
        abs_bearing = 90 -(angle_to_target)

        if abs_bearing > 360:
            abs_bearing -= 360
        if abs_bearing < 0:
            abs_bearing += 360

        return abs_bearing

    def calc_encounter_role(self, ego_ship, target_ship):
        # 입력값을 2π 범위 내로 조정

        course_diff = target_ship.heading - ego_ship.heading
        rel_bearing = self.calc_rel_bearing(ego_ship.x, ego_ship.y,
                                            target_ship.x, target_ship.y, ego_ship.heading)

        if course_diff < 0:
            course_diff += 360

        psi_ot_deg = course_diff
        beta_ot_deg = rel_bearing

        psi_ot = np.radians(psi_ot_deg)
        beta_ot = np.radians(beta_ot_deg)

        psi_ot = psi_ot % (2 * np.pi)
        beta_ot = beta_ot % (2 * np.pi)

        # 절차 2: 안전(Safe) 조건 확인
        if target_ship.speed <= 0.1:
            return "Safe"

        if (psi_ot > np.pi / 2) and (beta_ot < 3 * np.pi / 2) and (abs(beta_ot - psi_ot) < np.pi / 2):
            return "Safe"

        # 절차 5-11: 정면, 스타보드 교차, 포트 교차 확인
        if (psi_ot >= 7 * np.pi / 8) and (psi_ot < 9 * np.pi / 8):
            return "Headon"  # okay
        elif (psi_ot >= 9 * np.pi / 8) and (psi_ot < 13 * np.pi / 8):
            return "StarboardCrossing"  # okay
        elif (psi_ot >= 3 * np.pi / 8) and (psi_ot < 7 * np.pi / 8):
            return "PortCrossing"  # okay

        # 절차 12-21: 추월, 추월 당함, 기타 상황 확인
        psi_to = (2 * np.pi - psi_ot) % (2 * np.pi)
        beta_to = (np.pi + beta_ot - psi_ot) % (2 * np.pi)

        # TODO: overtaking, overtaken 용어 정리
        # print("after")

        # if (beta_ot >= 5 * np.pi / 8) and (beta_ot < 11 * np.pi / 8) and (ego_ship.speed > target_ship.speed):
        #     return "Overtaking"
        # elif (beta_to >= 5 * np.pi / 8) and (beta_to < 11 * np.pi / 8) and (ego_ship.speed < target_ship.speed):
        #     return "Overtaken"
        #
        if (beta_ot >= 5 * np.pi / 8) and (beta_ot < 11 * np.pi / 8) and (ego_ship.speed < target_ship.speed):
            return "Overtaken"
        elif (beta_to >= 5 * np.pi / 8) and (beta_to < 11 * np.pi / 8) and (ego_ship.speed > target_ship.speed):
            return "Overtaking"
        elif beta_ot < np.pi:
            return "StarboardCrossing"
        else:
            return "PortCrossing"

        return "Safe"