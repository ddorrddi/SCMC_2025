#!/usr/bin/python3

import os
import re
import json
import rospy
from std_msgs.msg import String
from collections import deque

class Cits_traffic_light:
    def __init__(self):
        self.traffic_light_pub = rospy.Publisher("/traffic_light_status", String, queue_size=1)
        rospy.Subscriber("/todo", String, self.traffic_light_callback)

    def find_log_files_in_order(self, log_dir, base_log_name):
        """
        logback.log -> logback.2024-10-13.0.log -> logback.2024-10-13.1.log 순서대로 파일을 읽도록 정렬
        """
        log_files = os.listdir(log_dir)
        base_log = [f for f in log_files if f == f"{base_log_name}.log"]
        rotated_logs = [f for f in log_files if re.match(rf"{base_log_name}\.\d+-\d+-\d+\.\d+\.log", f)]
        rotated_logs.sort()
        sorted_log_files = base_log + rotated_logs

        return [os.path.join(log_dir, f) for f in sorted_log_files]
    
    def get_line_count(self, log_file_path):
        """파일의 총 줄 수를 확인하는 함수"""
        with open(log_file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)

    def tail_log_file(self, log_file_path, lines_to_read=10000):
        """
        로그 파일의 끝에서부터 일정한 줄 수만 읽는 함수 (tail과 유사한 방식)
        """
        with open(log_file_path, 'r', encoding='utf-8') as file:
            # deque를 사용해 끝에서 일정 줄만 읽기
            return deque(file, maxlen=lines_to_read)

    def get_latest_json_with_id(self, log_file_path, target_id, max_lines=30000, tail_lines=10000):
        json_data = ""
        latest_json = None 

        total_lines = self.get_line_count(log_file_path)  # 파일의 전체 줄 수 확인

        # 로그가 너무 길면 끝에서 일정 줄만 읽기 (tail 방식)
        if total_lines > max_lines:
            print(f"Total lines: {total_lines}, reading the last {tail_lines} lines.")
            lines = self.tail_log_file(log_file_path, lines_to_read=tail_lines)
        else:
            print(f"Total lines: {total_lines}, reading the full file.")
            with open(log_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

        for line in lines:
            if '"messageId":19' in line:
                json_data = "{" + line 
            elif json_data:
                if 'INFO' in line:
                    try:
                        json_obj = json.loads(json_data)
                        if json_obj["value"]["intersections"][0]["id"]["region"] == target_id:
                            latest_json = json_obj  
                    except Exception as e:
                        print(e)
                    finally:
                        json_data = ""  
                else:
                    json_data += line  

        return latest_json 
    
    # 200 -> 300 -> 400 -> 500 -> 610 -> 400 -> 300 -> 200
    def traffic_light_callback(self, msg):
        current_traffic_light = int(msg.data)
        state = None
        remain_time = None
        log_dir = "./log/"
        base_log_name = "logback"

        log_file_path = self.find_log_files_in_order(log_dir, base_log_name)[-1]

        if not log_file_path:
            print("No log file found.")
            return

        if current_traffic_light == 1:
            traffic_info = self.get_latest_json_with_id(log_file_path, 200)
            if traffic_info:
                traffic_info = traffic_info["value"]["intersections"][0]["states"][13]["state-time-speed"][0]  # 14 북서좌
                state = traffic_info["eventState"]
                remain_time = traffic_info["timing"]["minEndTime"]
        elif current_traffic_light == 2:
            traffic_info = self.get_latest_json_with_id(log_file_path, 300)
            if traffic_info:
                traffic_info = traffic_info["value"]["intersections"][0]["states"][14]["state-time-speed"][0]  # 15 남서직
                state = traffic_info["eventState"]
                remain_time = traffic_info["timing"]["minEndTime"]
        elif current_traffic_light == 3:
            traffic_info = self.get_latest_json_with_id(log_file_path, 400)
            if traffic_info:
                traffic_info = traffic_info["value"]["intersections"][0]["states"][0]["state-time-speed"][0]  # 1 남직
                state = traffic_info["eventState"]
                remain_time = traffic_info["timing"]["minEndTime"]
        elif current_traffic_light == 4:
            traffic_info = self.get_latest_json_with_id(log_file_path, 500)
            if traffic_info:
                traffic_info = traffic_info["value"]["intersections"][0]["states"][1]["state-time-speed"][0]  # 2 남좌
                state = traffic_info["eventState"]
                remain_time = traffic_info["timing"]["minEndTime"]
        elif current_traffic_light == 5:
            traffic_info = self.get_latest_json_with_id(log_file_path, 610)
            if traffic_info:
                traffic_info = traffic_info["value"]["intersections"][0]["states"][3]["state-time-speed"][0]  # 4 동좌
                state = traffic_info["eventState"]
                remain_time = traffic_info["timing"]["minEndTime"]
        elif current_traffic_light == 6:
            traffic_info = self.get_latest_json_with_id(log_file_path, 400)
            if traffic_info:
                traffic_info = traffic_info["value"]["intersections"][0]["states"][3]["state-time-speed"][0]  # 우회전
                state = "protected-Movement-Allowed"
                remain_time = "9999"
        elif current_traffic_light == 7:
            traffic_info = self.get_latest_json_with_id(log_file_path, 300)
            if traffic_info:
                traffic_info = traffic_info["value"]["intersections"][0]["states"][10]["state-time-speed"][0]  # 11 북동직
                state = traffic_info["eventState"]
                remain_time = traffic_info["timing"]["minEndTime"]
        elif current_traffic_light == 8:
            traffic_info = self.get_latest_json_with_id(log_file_path, 200)
            if traffic_info:
                traffic_info = traffic_info["value"]["intersections"][0]["states"][10]["state-time-speed"][0]  # 11 북동직
                state = traffic_info["eventState"]
                remain_time = traffic_info["timing"]["minEndTime"]

        # 상태와 남은 시간을 ROS 메시지로 출판
        if state and remain_time:
            self.traffic_light_pub.publish(f"State: {state}, Remaining Time: {remain_time}")

if __name__ == "__main__":
    rospy.init_node("cits_traffic", anonymous=True)
    lidar_camera_calibration = Cits_traffic_light()
    rospy.spin()
