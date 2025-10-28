#!/usr/bin/env python3
"""
Simple Path Editor for main_interpolation.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class PathEditor:
    def __init__(self, filename):
        self.filename = filename
        self.load_data()
        self.selected_point = None
        self.modified = False
        self.dragging = False
        self.panning = False
        self.pan_start = None
        self.selecting = False
        self.selection_start = None
        self.selection_rect = None
        self.selected_points = []
        self.group_dragging = False
        self.group_drag_start = None
        self.undo_stack = []
        self.zoom_selecting = False
        self.zoom_selection_start = None
        self.zoom_selection_rect = None
        self.initial_xlim = None
        self.initial_ylim = None
        self.rotation_mode = False
        self.rotation_pivot = None
        self.rotation_start_angle = None
        self.rotation_corners = []
        self.corner_markers = None
        self.brush_mode = False
        self.brush_radius = 50.0  # 브러쉬 영향 반경
        self.brush_strength = 0.5  # 브러쉬 강도 (0-1)
        self.brush_dragging = False
        self.brush_last_pos = None

    def load_data(self):
        data = []
        with open(self.filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    extra = ' '.join(parts[3:])
                    data.append([x, y, z, extra])
        self.data = np.array(data, dtype=object)
        self.original_data = self.data.copy()
        print(f"Loaded {len(self.data)} points")

    def save_data(self):
        from tkinter import Tk, simpledialog

        # Create a simple dialog to get filename
        root = Tk()
        root.withdraw()  # Hide the main window

        filename = simpledialog.askstring(
            "Save File",
            "Enter filename to save:",
            initialvalue=self.filename
        )

        root.destroy()

        if filename is None:  # User cancelled
            print("Save cancelled")
            return

        if not filename:  # Empty string
            filename = self.filename

        # Create backup of original file
        backup_file = self.filename + '.backup'
        with open(backup_file, 'w') as f:
            for row in self.original_data:
                f.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")

        # Save to new filename
        with open(filename, 'w') as f:
            for row in self.data:
                f.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")
        print(f"Saved to: {filename} (backup: {backup_file})")
        self.modified = False

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(18, 10))
        plt.subplots_adjust(bottom=0.15, left=0.18, right=0.82)

        # 버튼
        ax_save = plt.axes([0.2, 0.02, 0.06, 0.05])
        ax_undo = plt.axes([0.27, 0.02, 0.06, 0.05])
        ax_reset = plt.axes([0.34, 0.02, 0.07, 0.05])
        ax_smooth = plt.axes([0.42, 0.02, 0.07, 0.05])
        ax_zoom_in = plt.axes([0.55, 0.02, 0.07, 0.05])
        ax_zoom_out = plt.axes([0.63, 0.02, 0.07, 0.05])
        ax_zoom_reset = plt.axes([0.71, 0.02, 0.08, 0.05])

        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(lambda _: self.save_data())

        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_undo.on_clicked(lambda _: self.undo())

        self.btn_reset = Button(ax_reset, 'Reset All')
        self.btn_reset.on_clicked(lambda _: self.reset())

        self.btn_smooth = Button(ax_smooth, 'Smooth')
        self.btn_smooth.on_clicked(lambda _: self.smooth_selected())

        self.btn_zoom_in = Button(ax_zoom_in, 'Zoom In')
        self.btn_zoom_in.on_clicked(lambda _: self.zoom(0.7))

        self.btn_zoom_out = Button(ax_zoom_out, 'Zoom Out')
        self.btn_zoom_out.on_clicked(lambda _: self.zoom(1.3))

        self.btn_zoom_reset = Button(ax_zoom_reset, 'Zoom Reset')
        self.btn_zoom_reset.on_clicked(lambda _: self.reset_zoom())

        # 정보 텍스트
        self.info_text = self.fig.text(0.2, 0.08, 'Click and drag to move points', fontsize=11)

        # 도움말 텍스트 (왼쪽 - 기본 조작)
        basic_help = (
            "═══ BASIC CONTROLS ═══\n"
            "• Left Click + Drag\n"
            "  → Move single point\n\n"
            "• Right Click + Drag\n"
            "  → Select area (green box)\n\n"
            "• Middle Click + Drag\n"
            "  → Pan view\n\n"
            "• Mouse Wheel\n"
            "  → Zoom in/out\n\n"
            "• Ctrl + Middle + Drag\n"
            "  → Zoom to area\n\n"
            "• Delete Key\n"
            "  → Remove selected points\n\n"
            "═══ BUTTONS ═══\n"
            "• Undo: Undo last action\n"
            "• Reset All: Reset to original\n"
            "• Zoom Reset: Reset view\n"
            "• Save: Save to file"
        )
        self.basic_help_text = self.fig.text(0.02, 0.5, basic_help, fontsize=8.5,
                                            verticalalignment='center',
                                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 도움말 텍스트 (오른쪽 - 새로운 기능)
        new_features_help = (
            "═══ NEW FEATURES ═══\n\n"
            "【 ADD POINT 】\n"
            "• Shift + Left Click\n"
            "  → Insert new point\n"
            "     on path line\n"
            "  → Point added between\n"
            "     nearest two points\n\n"
            "【 BRUSH MODE 】\n"
            "• B Key: Toggle brush\n"
            "• Brush ON:\n"
            "  → Drag to smoothly\n"
            "     edit path\n"
            "  → Nearby points move\n"
            "     with weight\n"
            "• [ / ] Keys:\n"
            "  → Decrease/Increase\n"
            "     brush radius\n"
            "• - / + Keys:\n"
            "  → Decrease/Increase\n"
            "     brush strength\n\n"
            "【 SMOOTH FILTER 】\n"
            "• Select area + Smooth btn\n"
            "  → Apply smoothing filter\n"
            "  → Makes path smoother\n\n"
            "【 ROTATION MODE 】\n"
            "• R Key: Toggle rotation\n"
            "• Click cyan corner\n"
            "  → Set pivot (red X)\n"
            "• Drag to rotate\n"
            "  → Around pivot point"
        )
        self.new_features_text = self.fig.text(0.84, 0.5, new_features_help, fontsize=8.5,
                                               verticalalignment='center',
                                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        self.line = None
        self.points = None
        self.selected_marker = None

        self.update_plot()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def update_plot(self):
        x = self.data[:, 0].astype(float)
        y = self.data[:, 1].astype(float)

        # 현재 뷰 범위 저장
        if self.line is not None:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        else:
            xlim = None
            ylim = None

        self.ax.clear()

        # 경로 선 (초록색)
        self.line, = self.ax.plot(x, y, 'g-', linewidth=1.5, alpha=0.7, zorder=1)

        # 모든 포인트
        self.points = self.ax.scatter(x, y, c='blue', s=50, alpha=0.6, zorder=2, edgecolors='darkblue', linewidths=1)

        # 브러쉬 모드일 때 브러쉬 원 표시
        if self.brush_mode and self.brush_last_pos is not None:
            from matplotlib.patches import Circle
            brush_circle = Circle(self.brush_last_pos, self.brush_radius,
                                fill=False, edgecolor='purple', linewidth=2,
                                linestyle='--', alpha=0.7, zorder=3)
            self.ax.add_patch(brush_circle)

        # 선택된 그룹 포인트들
        if len(self.selected_points) > 0:
            selected_x = x[self.selected_points]
            selected_y = y[self.selected_points]
            self.ax.scatter(selected_x, selected_y, c='orange', s=150, zorder=4, edgecolors='darkorange', linewidths=2, alpha=0.7)

            # 회전 모드일 때 꼭짓점과 pivot 표시
            if self.rotation_mode and len(self.rotation_corners) == 4:
                corners_x = [c[0] for c in self.rotation_corners]
                corners_y = [c[1] for c in self.rotation_corners]
                # 꼭짓점 표시 (큰 사각형)
                self.ax.scatter(corners_x, corners_y, c='cyan', s=300, zorder=6,
                              edgecolors='blue', linewidths=3, marker='s', alpha=0.8)
                # pivot 표시 (빨간 X)
                if self.rotation_pivot is not None:
                    self.ax.scatter([self.rotation_pivot[0]], [self.rotation_pivot[1]],
                                  c='red', s=400, zorder=7, marker='x', linewidths=4)

        # 포인트 번호 표시 (일부만)
        step = max(1, len(x) // 50)
        for i in range(0, len(x), step):
            self.ax.text(x[i], y[i], str(i), fontsize=8, alpha=0.5, ha='right', va='bottom')

        # 선택 영역 표시
        if self.selecting and self.selection_start is not None and self.selection_rect is not None:
            from matplotlib.patches import Rectangle
            self.ax.add_patch(self.selection_rect)

        # 줌 선택 영역 표시
        if self.zoom_selecting and self.zoom_selection_start is not None and self.zoom_selection_rect is not None:
            from matplotlib.patches import Rectangle
            self.ax.add_patch(self.zoom_selection_rect)

        # 선택된 포인트
        if self.selected_point is not None:
            self.selected_marker = self.ax.scatter(
                x[self.selected_point],
                y[self.selected_point],
                c='red',
                s=200,
                zorder=5,
                edgecolors='darkred',
                linewidths=2,
                marker='o',
                alpha=0.8
            )
            self.info_text.set_text(
                f'Point #{self.selected_point} | X: {x[self.selected_point]:.2f}, Y: {y[self.selected_point]:.2f} | Press Delete to remove'
            )
        elif len(self.selected_points) > 0:
            if self.rotation_mode:
                if self.rotation_pivot is not None:
                    self.info_text.set_text(f'ROTATION MODE | Pivot set (red X) | Drag points to rotate | Press R to exit')
                else:
                    self.info_text.set_text(f'ROTATION MODE | Click cyan corner to set pivot | Press R to exit')
            else:
                self.info_text.set_text(f'Selected: {len(self.selected_points)} points | Left-drag: move | R key: rotation mode | Delete: remove')
        elif self.brush_mode:
            self.info_text.set_text(f'BRUSH MODE ON | Radius: {self.brush_radius:.1f} | Strength: {self.brush_strength:.2f} | Drag to edit path | B to exit')
        else:
            self.info_text.set_text('Shift+Click: add point | B: brush mode | Left-click: move | Right-drag: select | Middle-drag: pan')

        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_title('Path Editor (Left-drag: move | Right-drag: select | Middle-drag: pan | Ctrl+Middle: zoom area)', fontsize=13, pad=10)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_aspect('equal', adjustable='datalim')

        # 뷰 범위 복원
        if xlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        else:
            # 초기 줌 범위 저장
            if self.initial_xlim is None:
                self.initial_xlim = self.ax.get_xlim()
                self.initial_ylim = self.ax.get_ylim()

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Shift + 좌클릭 - 새로운 점 추가
        if event.button == 1 and event.key == 'shift':
            self.add_point_on_path(event.xdata, event.ydata)
            return

        # 브러쉬 모드 좌클릭 - 브러쉬 드래그 시작
        if event.button == 1 and self.brush_mode:
            self.brush_dragging = True
            self.brush_last_pos = (event.xdata, event.ydata)
            self.save_state()
            return

        # 회전 모드에서 꼭짓점 클릭 - pivot 설정
        if event.button == 1 and self.rotation_mode and len(self.rotation_corners) == 4:
            threshold = min(self.ax.get_xlim()[1] - self.ax.get_xlim()[0],
                          self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.02

            for corner in self.rotation_corners:
                dist = np.sqrt((corner[0] - event.xdata)**2 + (corner[1] - event.ydata)**2)
                if dist < threshold * 2:  # 꼭짓점은 조금 더 큰 영역
                    self.rotation_pivot = corner
                    self.update_plot()
                    return

        # 마우스 휠 버튼 (중간 버튼)
        if event.button == 2:
            # Ctrl + 휠버튼 - 줌 영역 선택
            if event.key == 'control':
                self.zoom_selecting = True
                self.zoom_selection_start = (event.xdata, event.ydata)
            # 휠버튼만 - 패닝
            else:
                self.panning = True
                self.pan_start = (event.xdata, event.ydata)
            return

        # 우클릭 - 영역 선택 시작
        if event.button == 3:
            self.selecting = True
            self.selection_start = (event.xdata, event.ydata)
            self.selected_points = []
            return

        # 좌클릭
        if event.button != 1:
            return

        x = self.data[:, 0].astype(float)
        y = self.data[:, 1].astype(float)

        # 그룹이 선택되어 있으면 그룹 드래그 또는 회전
        if len(self.selected_points) > 0:
            # 선택된 포인트 중 하나라도 클릭했는지 확인
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            threshold = min(x_range, y_range) * 0.02

            for idx in self.selected_points:
                dist = np.sqrt((x[idx] - event.xdata)**2 + (y[idx] - event.ydata)**2)
                if dist < threshold:
                    self.save_state()  # Save state before modification
                    self.group_dragging = True
                    self.group_drag_start = (event.xdata, event.ydata)

                    # 회전 모드이고 pivot이 설정되어 있으면 회전 시작 각도 계산
                    if self.rotation_mode and self.rotation_pivot is not None:
                        dx = event.xdata - self.rotation_pivot[0]
                        dy = event.ydata - self.rotation_pivot[1]
                        self.rotation_start_angle = np.arctan2(dy, dx)

                    return

        # 단일 포인트 선택
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        threshold = min(x_range, y_range) * 0.02

        distances = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
        nearest_idx = np.argmin(distances)

        if distances[nearest_idx] < threshold:
            self.save_state()  # Save state before modification
            self.selected_point = nearest_idx
            self.dragging = True
            self.selected_points = []
            self.update_plot()

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return

        # 브러쉬 모드일 때 커서 위치 업데이트
        if self.brush_mode:
            self.brush_last_pos = (event.xdata, event.ydata)
            if not self.brush_dragging:
                self.update_plot()  # 브러쉬 원 업데이트

        # 브러쉬 드래그 중
        if self.brush_dragging and self.brush_last_pos is not None:
            self.apply_brush(event.xdata, event.ydata)
            return

        # 패닝 중
        if self.panning and self.pan_start is not None:
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.ax.set_xlim([xlim[0] + dx, xlim[1] + dx])
            self.ax.set_ylim([ylim[0] + dy, ylim[1] + dy])
            self.fig.canvas.draw_idle()
            return

        # 줌 영역 선택 중
        if self.zoom_selecting and self.zoom_selection_start is not None:
            from matplotlib.patches import Rectangle
            x0, y0 = self.zoom_selection_start
            width = event.xdata - x0
            height = event.ydata - y0

            if self.zoom_selection_rect is not None:
                self.zoom_selection_rect.remove()

            self.zoom_selection_rect = Rectangle((x0, y0), width, height,
                                                 linewidth=2, edgecolor='blue',
                                                 facecolor='blue', alpha=0.2)
            self.ax.add_patch(self.zoom_selection_rect)
            self.fig.canvas.draw_idle()
            return

        # 영역 선택 중
        if self.selecting and self.selection_start is not None:
            from matplotlib.patches import Rectangle
            x0, y0 = self.selection_start
            width = event.xdata - x0
            height = event.ydata - y0

            if self.selection_rect is not None:
                self.selection_rect.remove()

            self.selection_rect = Rectangle((x0, y0), width, height,
                                           linewidth=2, edgecolor='green',
                                           facecolor='green', alpha=0.2)
            self.ax.add_patch(self.selection_rect)
            self.fig.canvas.draw_idle()
            return

        # 그룹 드래그 중 (회전 또는 이동)
        if self.group_dragging and self.group_drag_start is not None:
            # 회전 모드이고 pivot이 설정되어 있으면 회전
            if self.rotation_mode and self.rotation_pivot is not None and self.rotation_start_angle is not None:
                # 현재 각도 계산
                dx = event.xdata - self.rotation_pivot[0]
                dy = event.ydata - self.rotation_pivot[1]
                current_angle = np.arctan2(dy, dx)

                # 회전 각도
                angle = current_angle - self.rotation_start_angle

                # 각 포인트를 pivot 기준으로 회전
                pivot_x, pivot_y = self.rotation_pivot
                for idx in self.selected_points:
                    # pivot 기준 상대 좌표
                    rel_x = float(self.data[idx, 0]) - pivot_x
                    rel_y = float(self.data[idx, 1]) - pivot_y

                    # 회전 변환
                    new_x = rel_x * np.cos(angle) - rel_y * np.sin(angle)
                    new_y = rel_x * np.sin(angle) + rel_y * np.cos(angle)

                    # 절대 좌표로 변환
                    self.data[idx, 0] = new_x + pivot_x
                    self.data[idx, 1] = new_y + pivot_y

                # 다음 회전을 위해 시작 각도 업데이트
                self.rotation_start_angle = current_angle

            # 일반 이동 모드
            else:
                dx = event.xdata - self.group_drag_start[0]
                dy = event.ydata - self.group_drag_start[1]

                for idx in self.selected_points:
                    self.data[idx, 0] = float(self.data[idx, 0]) + dx
                    self.data[idx, 1] = float(self.data[idx, 1]) + dy

                self.group_drag_start = (event.xdata, event.ydata)

            self.modified = True
            self.update_plot()
            return

        # 단일 포인트 드래그 중
        if self.dragging and self.selected_point is not None:
            self.data[self.selected_point, 0] = event.xdata
            self.data[self.selected_point, 1] = event.ydata
            self.modified = True
            self.update_plot()

    def save_state(self):
        """Save current state for undo"""
        self.undo_stack.append(self.data.copy())

    def on_release(self, event):
        # 패닝 종료
        if event.button == 2 and self.panning:
            self.panning = False
            self.pan_start = None

        # 줌 영역 선택 종료
        if event.button == 2 and self.zoom_selecting:
            self.zoom_selecting = False

            if self.zoom_selection_start is not None and event.xdata is not None:
                x0, y0 = self.zoom_selection_start
                x1, y1 = event.xdata, event.ydata

                # 선택 영역 정규화
                x_min, x_max = min(x0, x1), max(x0, x1)
                y_min, y_max = min(y0, y1), max(y0, y1)

                # 영역이 충분히 크면 줌인
                if abs(x_max - x_min) > 0.1 and abs(y_max - y_min) > 0.1:
                    self.ax.set_xlim([x_min, x_max])
                    self.ax.set_ylim([y_min, y_max])

            self.zoom_selection_start = None
            if self.zoom_selection_rect is not None:
                self.zoom_selection_rect.remove()
                self.zoom_selection_rect = None
            self.fig.canvas.draw_idle()

        # 영역 선택 종료
        if event.button == 3 and self.selecting:
            self.selecting = False

            if self.selection_start is not None and event.xdata is not None:
                x0, y0 = self.selection_start
                x1, y1 = event.xdata, event.ydata

                # 선택 영역 정규화
                x_min, x_max = min(x0, x1), max(x0, x1)
                y_min, y_max = min(y0, y1), max(y0, y1)

                # 영역 내 포인트 찾기
                x = self.data[:, 0].astype(float)
                y = self.data[:, 1].astype(float)

                self.selected_points = []
                for i in range(len(x)):
                    if x_min <= x[i] <= x_max and y_min <= y[i] <= y_max:
                        self.selected_points.append(i)

                # 선택 박스의 꼭짓점 계산 (회전용)
                if len(self.selected_points) > 0:
                    self.rotation_corners = [
                        (x_min, y_min),  # 좌하단
                        (x_max, y_min),  # 우하단
                        (x_max, y_max),  # 우상단
                        (x_min, y_max),  # 좌상단
                    ]

            self.selection_start = None
            if self.selection_rect is not None:
                self.selection_rect.remove()
                self.selection_rect = None
            self.update_plot()

        # 브러쉬 드래그 종료
        if event.button == 1 and self.brush_dragging:
            self.brush_dragging = False

        # 그룹 드래그 종료
        if self.group_dragging:
            self.group_dragging = False
            self.group_drag_start = None

        # 단일 포인트 드래그 종료
        if self.dragging:
            self.dragging = False
            self.selected_point = None
            self.update_plot()

    def on_key(self, event):
        """Handle keyboard events"""
        # B 키 - 브러쉬 모드 토글
        if event.key == 'b':
            self.brush_mode = not self.brush_mode
            if self.brush_mode:
                print(f"Brush mode ON - Radius: {self.brush_radius:.1f}, Strength: {self.brush_strength:.2f}")
            else:
                print("Brush mode OFF")
                self.brush_last_pos = None
            self.update_plot()
            return

        # [ ] 키 - 브러쉬 반경 조절
        if event.key == '[' and self.brush_mode:
            self.brush_radius = max(10.0, self.brush_radius - 10.0)
            print(f"Brush radius: {self.brush_radius:.1f}")
            self.update_plot()
            return
        if event.key == ']' and self.brush_mode:
            self.brush_radius = min(200.0, self.brush_radius + 10.0)
            print(f"Brush radius: {self.brush_radius:.1f}")
            self.update_plot()
            return

        # - + 키 - 브러쉬 강도 조절
        if event.key == '-' and self.brush_mode:
            self.brush_strength = max(0.1, self.brush_strength - 0.1)
            print(f"Brush strength: {self.brush_strength:.2f}")
            self.update_plot()
            return
        if event.key == '+' and self.brush_mode:
            self.brush_strength = min(1.0, self.brush_strength + 0.1)
            print(f"Brush strength: {self.brush_strength:.2f}")
            self.update_plot()
            return

        # R 키 - 회전 모드 토글
        if event.key == 'r' and len(self.selected_points) > 0:
            self.rotation_mode = not self.rotation_mode
            if not self.rotation_mode:
                # 회전 모드 종료 시 초기화
                self.rotation_pivot = None
                self.rotation_start_angle = None
            self.update_plot()
            return

        # Delete key to remove points
        if event.key == 'delete':
            # Delete selected group
            if len(self.selected_points) > 0:
                self.save_state()  # Save state before modification
                # Remove points (in reverse order to maintain indices)
                for idx in sorted(self.selected_points, reverse=True):
                    self.data = np.delete(self.data, idx, axis=0)
                self.selected_points = []
                self.modified = True
                self.update_plot()
            # Delete single point
            elif self.selected_point is not None:
                self.save_state()  # Save state before modification
                self.data = np.delete(self.data, self.selected_point, axis=0)
                self.selected_point = None
                self.modified = True
                self.update_plot()

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 'up':
            self.zoom(0.8, event.xdata, event.ydata)
        elif event.button == 'down':
            self.zoom(1.25, event.xdata, event.ydata)

    def zoom(self, factor, x_center=None, y_center=None):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        if x_center is None:
            x_center = (xlim[0] + xlim[1]) / 2
        if y_center is None:
            y_center = (ylim[0] + ylim[1]) / 2

        x_range = (xlim[1] - xlim[0]) * factor
        y_range = (ylim[1] - ylim[0]) * factor

        self.ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
        self.ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
        self.fig.canvas.draw_idle()

    def undo(self):
        """Undo last operation"""
        if len(self.undo_stack) > 0:
            self.data = self.undo_stack.pop()
            self.selected_point = None
            self.selected_points = []
            self.modified = True
            self.update_plot()
            print(f"Undo - {len(self.undo_stack)} steps remaining")
        else:
            print("Nothing to undo")

    def reset_zoom(self):
        """Reset zoom to initial view"""
        if self.initial_xlim is not None and self.initial_ylim is not None:
            self.ax.set_xlim(self.initial_xlim)
            self.ax.set_ylim(self.initial_ylim)
            self.fig.canvas.draw_idle()
            print("Zoom reset to initial view")

    def reset(self):
        """Reset to original state"""
        self.data = self.original_data.copy()
        self.selected_point = None
        self.selected_points = []
        self.dragging = False
        self.group_dragging = False
        self.selecting = False
        self.modified = False
        self.undo_stack = []
        self.brush_mode = False
        self.brush_dragging = False
        self.update_plot()
        print("Reset to original state")

    def add_point_on_path(self, x, y):
        """Add a new point on the path near the clicked location"""
        if len(self.data) < 2:
            print("Need at least 2 points to add between")
            return

        self.save_state()

        # 클릭한 위치에서 가장 가까운 선분 찾기
        path_x = self.data[:, 0].astype(float)
        path_y = self.data[:, 1].astype(float)

        min_dist = float('inf')
        insert_idx = 1

        for i in range(len(self.data) - 1):
            x1, y1 = path_x[i], path_y[i]
            x2, y2 = path_x[i + 1], path_y[i + 1]

            # 선분까지의 거리 계산
            dx = x2 - x1
            dy = y2 - y1
            length_sq = dx*dx + dy*dy

            if length_sq == 0:
                dist = np.sqrt((x - x1)**2 + (y - y1)**2)
            else:
                t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.sqrt((x - proj_x)**2 + (y - proj_y)**2)

            if dist < min_dist:
                min_dist = dist
                insert_idx = i + 1
                # 선분 위의 투영점 계산
                if length_sq == 0:
                    new_x, new_y = x1, y1
                else:
                    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_sq))
                    new_x = x1 + t * dx
                    new_y = y1 + t * dy

        # 새 점 삽입
        z_value = self.data[insert_idx - 1, 2]  # 이전 점의 z 값 사용
        extra = self.data[insert_idx - 1, 3]  # 이전 점의 extra 데이터 사용
        new_point = np.array([[new_x, new_y, z_value, extra]], dtype=object)
        self.data = np.insert(self.data, insert_idx, new_point[0], axis=0)

        self.modified = True
        self.update_plot()
        print(f"Added point at index {insert_idx}: ({new_x:.2f}, {new_y:.2f})")

    def apply_brush(self, x, y):
        """Apply brush effect to nearby points"""
        if self.brush_last_pos is None:
            return

        prev_x, prev_y = self.brush_last_pos
        dx = x - prev_x
        dy = y - prev_y

        path_x = self.data[:, 0].astype(float)
        path_y = self.data[:, 1].astype(float)

        # 브러쉬 반경 내의 점들에 가중치 적용하여 이동
        for i in range(len(self.data)):
            dist = np.sqrt((path_x[i] - x)**2 + (path_y[i] - y)**2)

            if dist < self.brush_radius:
                # 가우시안 가중치 계산 (중심에서 멀수록 영향 감소)
                weight = np.exp(-(dist**2) / (2 * (self.brush_radius / 2)**2))
                weight *= self.brush_strength

                # 포인트 이동
                self.data[i, 0] = float(self.data[i, 0]) + dx * weight
                self.data[i, 1] = float(self.data[i, 1]) + dy * weight

        self.brush_last_pos = (x, y)
        self.modified = True
        self.update_plot()

    def smooth_selected(self):
        """Apply smoothing filter to selected points"""
        if len(self.selected_points) < 3:
            print("Need at least 3 selected points to smooth")
            return

        self.save_state()

        # 선택된 포인트들을 정렬
        sorted_indices = sorted(self.selected_points)

        # 이동평균 필터 적용 (window size = 3)
        path_x = self.data[:, 0].astype(float)
        path_y = self.data[:, 1].astype(float)

        new_x = path_x.copy()
        new_y = path_y.copy()

        for idx in sorted_indices:
            # 양 옆 포인트 찾기
            neighbors = []
            if idx > 0:
                neighbors.append(idx - 1)
            neighbors.append(idx)
            if idx < len(self.data) - 1:
                neighbors.append(idx + 1)

            # 이웃 포인트들의 평균
            if len(neighbors) > 1:
                new_x[idx] = np.mean(path_x[neighbors])
                new_y[idx] = np.mean(path_y[neighbors])

        # 업데이트
        for idx in sorted_indices:
            self.data[idx, 0] = new_x[idx]
            self.data[idx, 1] = new_y[idx]

        self.modified = True
        self.update_plot()
        print(f"Smoothed {len(sorted_indices)} points")

    def run(self):
        self.setup_plot()
        plt.show()

if __name__ == '__main__':
    import os
    # 스크립트가 있는 디렉토리 기준으로 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'kcity_semi.txt')
    editor = PathEditor(file_path)
    editor.run()
