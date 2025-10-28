import cv2  # OpenCV 라이브러리 임포트
import numpy as np  # NumPy 라이브러리 임포트
import matplotlib.pyplot as plt  # matplotlib 라이브러리 임포트
from matplotlib.pyplot import *  # matplotlib.pyplot 모듈 임포트
import math  # 수학 라이브러리 임포트

TOTAL_CNT = 50  # 총 카운트 설정

class SlideWindow:

    def __init__(self):
        self.current_line = "DEFAULT"  # 현재 라인 초기화
        self.left_fit = None  # 왼쪽 라인 피팅 초기화
        self.right_fit = None  # 오른쪽 라인 피팅 초기화
        self.leftx = None  # 왼쪽 x 좌표 초기화
        self.rightx = None  # 오른쪽 x 좌표 초기화
        self.lhd = 240  # ?
        self.left_cnt = 25  # 왼쪽 카운트 초기화
        self.right_cnt = 25  # 오른쪽 카운트 초기화

        self.x_previous = 320  # 이전 x 위치 초기화
        # self.x_temp = 320

        # 이전 차선 위치 저장 (강건성 향상)
        self.prev_left_x = None
        self.prev_right_x = None
        self.left_lane_inds_all = []
        self.right_lane_inds_all = []

    def slidewindow(self, img, is_detected, force_line=None):
        x_location = 320  # x 위치 초기화
        # 출력 이미지 초기화 및 높이, 너비 설정
        out_img = np.dstack((img, img, img)) * 255



        height = img.shape[0]  # 이미지 높이
        width = img.shape[1]  # 이미지 너비

        # 윈도우 개수 및 높이 초기화
        window_height = 15  # 윈도우 높이
        nwindows = 30  # 윈도우 개수
        
        # 이미지에서 비영점 위치 찾기
        nonzero = img.nonzero()  # 이미지에서 비영점 찾기
        nonzeroy = np.array(nonzero[0])  # y 좌표 배열로 변환
        nonzerox = np.array(nonzero[1])  # x 좌표 배열로 변환
        margin = 40  # 윈도우 마진 설정
        minpix = 0  # 최소 픽셀 수 설정
        left_lane_inds = []  # 왼쪽 차선 인덱스 초기화
        right_lane_inds = []  # 오른쪽 차선 인덱스 초기화

        # # 윈도우 설정 값 초기화 - original
        # win_h1 = 100  # 윈도우 높이 1
        # win_h2 = 150  # 윈도우 높이 2
        # win_l_w_l = 180 - 50  # 왼쪽 윈도우 왼쪽 경계
        # win_l_w_r = 180 + 50  # 왼쪽 윈도우 오른쪽 경계
        # win_r_w_l = 480 - 50  # 오른쪽 윈도우 왼쪽 경계
        # win_r_w_r = 480 + 50  # 오른쪽 윈도우 오른쪽 경계


        # 윈도우 설정 값 초기화
        # ----------------- 기본값 -------------------- #
        # win_h1 = height - 50  # 윈도우 높이 1
        # win_h2 = height  # 윈도우 높이 2
        # win_l_w_l = 180 - 95  # 왼쪽 윈도우 왼쪽 경계
        # win_l_w_r = 180 + 50  # 왼쪽 윈도우 오른쪽 경계
        # win_r_w_l = 480 - 65  # 오른쪽 윈도우 왼쪽 경계
        # win_r_w_r = 480 + 65  # 오른쪽 윈도우 오른쪽 경계
        # ---------------------------------------------

        win_h1 = height - 60  # 윈도우 높이 1
        win_h2 = height  # 윈도우 높이 2
        win_l_w_l = 180 - 140  # 왼쪽 윈도우 왼쪽 경계
        win_l_w_r = 180 + 140  # 왼쪽 윈도우 오른쪽 경계
        win_r_w_l = 480 - 140  # 오른쪽 윈도우 왼쪽 경계 (넓게)
        win_r_w_r = 480 + 140  # 오른쪽 윈도우 오른쪽 경계 (넓게)

        circle_height = 100  # 원 높이 설정
        
        road_width = 0.25  # 도로 너비 설정
        half_road_width = 0.5 * road_width  # 도로 반 너비 설정

        # 왼쪽 폴리라인 설정
        pts_left = np.array([[win_l_w_l, win_h2], [win_l_w_l, win_h1], [win_l_w_r, win_h1], [win_l_w_r, win_h2]], np.int32)
        cv2.polylines(out_img, [pts_left], False, (0, 255, 0), 1)
        
        # 오른쪽 폴리라인 설정
        pts_right = np.array([[win_r_w_l, win_h2], [win_r_w_l, win_h1], [win_r_w_r, win_h1], [win_r_w_r, win_h2]], np.int32)
        cv2.polylines(out_img, [pts_right], False, (255, 0, 0), 1)

        # 원 폴리라인 설정
        pts_catch = np.array([[0, circle_height], [width, circle_height]], np.int32)
        cv2.polylines(out_img, [pts_catch], False, (0, 120, 120), 1)

        # 좋은 왼쪽 및 오른쪽 인덱스 찾기
        good_left_inds = ((nonzerox >= win_l_w_l) & (nonzeroy <= win_h2) & (nonzeroy > win_h1) & (nonzerox <= win_l_w_r)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_r_w_l) & (nonzeroy <= win_h2) & (nonzeroy > win_h1) & (nonzerox <= win_r_w_r)).nonzero()[0]

        y_current = height - 1  # 현재 y 위치 초기화
        x_current = None  # 현재 x 위치 초기화

        # force_line 파라미터로 강제 선택
        if force_line == 'left' and len(good_left_inds) > 0:
            line_flag = 1  # 왼쪽 라인 플래그 설정
            x_current = int(np.mean(nonzerox[good_left_inds]))  # 현재 x 위치 설정
            y_current = int(np.mean(nonzeroy[good_left_inds]))  # 현재 y 위치 설정
        elif force_line == 'right' and len(good_right_inds) > 0:
            line_flag = 2  # 오른쪽 라인 플래그 설정
            x_current = int(np.mean(nonzerox[good_right_inds]))  # 현재 x 위치 설정
            y_current = int(np.mean(nonzeroy[good_right_inds]))  # 현재 y 위치 설정
        # None일 때 양쪽 차선 모두 사용 (중앙값)
        elif force_line is None and len(good_left_inds) > 0 and len(good_right_inds) > 0:
            line_flag = 3  # 양쪽 차선 모두 감지
            left_x = int(np.mean(nonzerox[good_left_inds]))
            right_x = int(np.mean(nonzerox[good_right_inds]))
            x_current = (left_x + right_x) // 2  # 양쪽 차선의 중간값
            y_current = height - 1
        # 더 많은 픽셀을 포함하는 라인을 찾기 (한쪽만 감지될 때)
        elif len(good_left_inds) > len(good_right_inds):
            line_flag = 1  # 왼쪽 라인 플래그 설정
            x_current = int(np.mean(nonzerox[good_left_inds]))  # 현재 x 위치 설정
            y_current = int(np.mean(nonzeroy[good_left_inds]))  # 현재 y 위치 설정
        elif len(good_left_inds) < len(good_right_inds):
            line_flag = 2  # 오른쪽 라인 플래그 설정
            x_current = int(np.mean(nonzerox[good_right_inds]))  # 현재 x 위치 설정
            y_current = int(np.mean(nonzeroy[good_right_inds]))  # 현재 y 위치 설정
        else:
            self.current_line = "MID"  # 중앙 라인 설정
            line_flag = 3  # 중앙 라인 플래그 설정

        # 왼쪽 라인 픽셀 시각화
        if line_flag == 1:
            for i in range(len(good_left_inds)):
                img = cv2.circle(out_img, (nonzerox[good_left_inds[i]], nonzeroy[good_left_inds[i]]), 1, (0, 255, 0), -1)
        # 오른쪽 라인 픽셀 시각화
        elif line_flag == 2:
            for i in range(len(good_right_inds)):
                img = cv2.circle(out_img, (nonzerox[good_right_inds[i]], nonzeroy[good_right_inds[i]]), 1, (255, 0, 0), -1)
        # 양쪽 라인 픽셀 시각화
        elif line_flag == 3:
            for i in range(len(good_left_inds)):
                img = cv2.circle(out_img, (nonzerox[good_left_inds[i]], nonzeroy[good_left_inds[i]]), 1, (0, 255, 0), -1)
            for i in range(len(good_right_inds)):
                img = cv2.circle(out_img, (nonzerox[good_right_inds[i]], nonzeroy[good_right_inds[i]]), 1, (255, 0, 0), -1)

        # 윈도우를 반복하며 라인 찾기
        for window in range(nwindows):
            win_y_low = height - (window + 1) * window_height  # 윈도우 아래쪽 y 좌표
            win_y_high = height - window * window_height  # 윈도우 위쪽 y 좌표

            if line_flag == 1:  # 왼쪽 라인일 경우
                win_x_low = x_current - margin  # 윈도우 왼쪽 x 좌표
                win_x_high = x_current + margin  # 윈도우 오른쪽 x 좌표
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low + int(width * road_width), win_y_low), (win_x_high + int(width * road_width), win_y_high), (255, 0, 0), 1)
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                if len(good_left_inds) > minpix:
                    x_current = int(np.mean(nonzerox[good_left_inds]))
                    left_lane_inds.append(good_left_inds)
                elif self.prev_left_x is not None:
                    # 차선 못 찾으면 이전 위치 사용
                    x_current = self.prev_left_x
                elif nonzeroy[left_lane_inds] != [] and nonzerox[left_lane_inds] != []:
                    p_left = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
                    x_current = int(np.polyval(p_left, win_y_high))
                if circle_height - 10 <= win_y_low < circle_height + 10:
                    x_location = int(x_current + width * half_road_width)
                    self.prev_left_x = x_current  # 현재 위치 저장
                    cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)

            elif line_flag == 2:  # 오른쪽 라인일 경우
                win_x_low = x_current - margin  # 윈도우 왼쪽 x 좌표
                win_x_high = x_current + margin  # 윈도우 오른쪽 x 좌표
                cv2.rectangle(out_img, (win_x_low - int(width * road_width), win_y_low), (win_x_high - int(width * road_width), win_y_high), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (255, 0, 0), 1)
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                if len(good_right_inds) > minpix:
                    x_current = int(np.mean(nonzerox[good_right_inds]))
                    right_lane_inds.append(good_right_inds)
                elif self.prev_right_x is not None:
                    # 차선 못 찾으면 이전 위치 사용
                    x_current = self.prev_right_x
                elif nonzeroy[right_lane_inds] != [] and nonzerox[right_lane_inds] != []:
                    p_right = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
                    x_current = int(np.polyval(p_right, win_y_high))
                if circle_height - 10 <= win_y_low < circle_height + 10:
                    x_location = int(x_current - width * half_road_width)
                    self.prev_right_x = x_current  # 현재 위치 저장
                    cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)

            else:  # 중앙 라인일 경우 (양쪽 차선 중간값)
                # 양쪽 차선을 각각 추적
                # 왼쪽 차선 윈도우
                win_x_low_left = left_x - margin
                win_x_high_left = left_x + margin
                cv2.rectangle(out_img, (win_x_low_left, win_y_low), (win_x_high_left, win_y_high), (0, 255, 0), 1)
                good_left_inds_new = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low_left) & (nonzerox < win_x_high_left)).nonzero()[0]
                if len(good_left_inds_new) > minpix:
                    left_x = int(np.mean(nonzerox[good_left_inds_new]))
                elif self.prev_left_x is not None:
                    left_x = self.prev_left_x  # 이전 위치 사용

                # 오른쪽 차선 윈도우
                win_x_low_right = right_x - margin
                win_x_high_right = right_x + margin
                cv2.rectangle(out_img, (win_x_low_right, win_y_low), (win_x_high_right, win_y_high), (255, 0, 0), 1)
                good_right_inds_new = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low_right) & (nonzerox < win_x_high_right)).nonzero()[0]
                if len(good_right_inds_new) > minpix:
                    right_x = int(np.mean(nonzerox[good_right_inds_new]))
                elif self.prev_right_x is not None:
                    right_x = self.prev_right_x  # 이전 위치 사용

                # 중간값 계산
                x_current = (left_x + right_x) // 2

                if circle_height - 10 <= win_y_low < circle_height + 10:
                    x_location = x_current  # 양쪽 차선 중간값 사용
                    self.prev_left_x = left_x  # 현재 위치 저장
                    self.prev_right_x = right_x
                    cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)

            # x 위치 설정
            if x_location == 320:
                x_location = self.x_previous
                cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)

            self.x_previous = x_location  # 이전 x 위치 갱신
        return out_img, x_location, self.current_line  # 출력 이미지, x 위치, 현재 라인 반환