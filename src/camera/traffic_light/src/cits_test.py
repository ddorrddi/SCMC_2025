import os
import json
import re
import time

class Cits_traffic_light:
    def __init__(self):
        pass

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

    def tail_log_file(self, log_file_path, lines_to_read=10000):
        """
        로그 파일의 끝에서부터 일정한 줄 수만 읽는 함수 (tail 방식)
        """
        with open(log_file_path, 'rb') as file:  # 바이너리 모드로 열기
            file.seek(0, os.SEEK_END)
            buffer_size = 1024
            buffer = b''  # 바이너리 문자열 초기화
            file_size = file.tell()
            position = file_size
            new_lines = []
            
            while position > 0 and len(new_lines) < lines_to_read:
                position = max(0, position - buffer_size)
                file.seek(position)
                buffer = file.read(min(buffer_size, position)) + buffer  # bytes 타입끼리 연결
                new_lines = buffer.decode('utf-8', errors='replace').splitlines()  # 디코딩하여 str로 변환

            return new_lines[-lines_to_read:]  # 필요한 줄만 반환

    def get_latest_json_with_id(self, log_file_path, target_id, tail_lines=10000):
        json_data = ""
        latest_json = None 

        t1 = time.time()

        # 끝에서 일정 줄만 읽기
        lines = self.tail_log_file(log_file_path, lines_to_read=tail_lines)
        # print(lines)

        t2 = time.time()
        print("Read last lines:", t2 - t1)

        t1 = time.time()

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
                        pass
                    finally:
                        json_data = ""  
                else:
                    json_data += line  

        t2 = time.time()
        print("parse", t2 - t1)
        print(latest_json)

        return latest_json 

    def traffic_light_callback(self, current_traffic_light, log_dir, base_log_name):
        state = None
        remain_time = None

        t1 = time.time()
                
        log_file_path = self.find_log_files_in_order(log_dir, base_log_name)[-1]
        print(log_file_path)
        t2 = time.time()

        print("log_file_path", t2 - t1)

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

        if state and remain_time:
            print(f"State: {state}, Remaining Time: {remain_time}")
        else:
            print("No matching data found for the current traffic light state.")


if __name__ == "__main__":
    s_time = time.time()
    log_dir = './log/'
    base_log_name = 'logback' 
    log_file_path = './log/logback.log'

    traffic_light_handler = Cits_traffic_light()

    # 트래픽 라이트 콜백 호출
    traffic_light_handler.traffic_light_callback(2, log_dir, base_log_name)

    e_time = time.time()
    print(f"Execution time: {e_time - s_time} seconds")
