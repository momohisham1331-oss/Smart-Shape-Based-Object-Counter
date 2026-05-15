import cv2
import numpy as np
from collections import defaultdict

class ObjectDetector:
    def __init__(self):
        self.tracked_objects = {}
        self.object_history = defaultdict(list)
        self.next_id = 0
    
    def detect_objects(self, frame, canny_low=50, canny_high=150,
                      hough_threshold=50, circles_param2=30, min_distance=50):
        """كشف الأجسام من الإطار"""
        
        # تحويل إلى رمادي
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny Edge Detection
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # البحث عن الدوائر
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_distance,
            param1=canny_high,
            param2=circles_param2,
            minRadius=8,
            maxRadius=100
        )
        
        # البحث عن الخطوط
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=hough_threshold,
            minLineLength=30,
            maxLineGap=10
        )
        
        return {
            'gray': gray,
            'blurred': blurred,
            'edges': edges,
            'circles': circles,
            'lines': lines,
            'frame': frame
        }
    
    def classify_objects(self, detection_result):
        """تصنيف الكائنات"""
        
        objects = []
        circles = detection_result['circles']
        lines = detection_result['lines']
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle_list = []
            
            for i in circles[0, :]:
                circle_list.append({
                    'x': i[0],
                    'y': i[1],
                    'r': i[2]
                })
            
            # البحث عن الدراجات (دائرتان)
            bicycles = self._find_bicycles(circle_list)
            for bicycle in bicycles:
                objects.append({
                    'type': 'bicycle',
                    'circles': bicycle,
                    'lines': []
                })
            
            # البحث عن السيارات (2+ دوائر + خطوط)
            if lines is not None and len(lines) > 0:
                cars = self._find_cars(circle_list, lines, bicycles)
                for car in cars:
                    objects.append({
                        'type': 'car',
                        'circles': car['circles'],
                        'lines': car['lines']
                    })
        
        return objects
    
    def _find_bicycles(self, circles):
        """البحث عن الدراجات"""
        bicycles = []
        used = set()
        
        for i, circle1 in enumerate(circles):
            if i in used:
                continue
            
            for j, circle2 in enumerate(circles):
                if i >= j or j in used:
                    continue
                
                # المسافة بين الدائرتين
                distance = np.sqrt(
                    (circle1['x'] - circle2['x'])**2 +
                    (circle1['y'] - circle2['y'])**2
                )
                
                # إذا كانا متقاربتان
                if 30 < distance < 250:
                    bicycles.append([circle1, circle2])
                    used.add(i)
                    used.add(j)
                    break
        
        return bicycles
    
    def _find_cars(self, circles, lines, used_bicycles):
        """البحث عن السيارات"""
        cars = []
        used_circle_indices = set()
        
        # تجميع الخطوط
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x2 != x1:
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            else:
                angle = 90
            
            # الخطوط الأفقية
            if angle < 30 or angle > 150:
                horizontal_lines.append(line[0])
            # الخطوط العمودية
            elif 60 < angle < 120:
                vertical_lines.append(line[0])
        
        # السيارة عادة تحتوي على خطوط
        if len(horizontal_lines) >= 1 and len(vertical_lines) >= 1:
            # استخدام أول دائرة لم تُستخدم في دراجة
            for idx, circle in enumerate(circles):
                is_in_bicycle = False
                
                for bicycle in used_bicycles:
                    for b_circle in bicycle:
                        if (circle['x'] == b_circle['x'] and
                            circle['y'] == b_circle['y']):
                            is_in_bicycle = True
                            break
                
                if not is_in_bicycle and idx not in used_circle_indices:
                    cars.append({
                        'circles': [circle],
                        'lines': [horizontal_lines[0], vertical_lines[0]]
                    })
                    used_circle_indices.add(idx)
                    break
        
        return cars
    
    def draw_results(self, frame, detection_result, objects,
                    canny_low=50, canny_high=150,
                    hough_threshold=50, circles_param2=30,
                    show_circles=True, show_lines=True,
                    show_edges=False):
        """رسم النتائج"""
        
        result_frame = frame.copy()
        circles = detection_result['circles']
        lines = detection_result['lines']
        edges = detection_result['edges']
        
        # رسم الحواف إن لزم الأمر
        if show_edges:
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            result_frame = cv2.addWeighted(result_frame, 0.7, edges_colored, 0.3, 0)
        
        # رسم الدوائر
        if show_circles and circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                cv2.circle(result_frame, center, radius, (0, 255, 0), 2)
                cv2.circle(result_frame, center, 2, (0, 255, 0), 3)
        
        # رسم الخطوط
        if show_lines and lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # رسم الكائنات المصنفة
        car_count = 0
        bicycle_count = 0
        
        for obj in objects:
            if obj['type'] == 'car':
                car_count += 1
                for circle in obj.get('circles', []):
                    x, y, r = circle['x'], circle['y'], circle['r']
                    cv2.rectangle(result_frame,
                                (x - r - 20, y - r - 20),
                                (x + r + 20, y + r + 20),
                                (0, 0, 255), 2)
                    cv2.putText(result_frame, 'CAR',
                              (x - 25, y - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              (0, 0, 255), 2)
            
            elif obj['type'] == 'bicycle':
                bicycle_count += 1
                for circle in obj.get('circles', []):
                    x, y, r = circle['x'], circle['y'], circle['r']
                    cv2.rectangle(result_frame,
                                (x - r - 20, y - r - 20),
                                (x + r + 20, y + r + 20),
                                (255, 0, 0), 2)
                
                if obj['circles']:
                    x = obj['circles'][0]['x']
                    y = obj['circles'][0]['y']
                    cv2.putText(result_frame, 'BICYCLE',
                              (x - 40, y - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              (255, 0, 0), 2)
        
        # كتابة الإحصائيات
        cv2.putText(result_frame, f"Cars: {car_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Bicycles: {bicycle_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return result_frame, car_count, bicycle_count
