import cv2
import numpy as np
import os
import sys
from collections import defaultdict
from datetime import datetime

class ObjectDetector:
    def __init__(self):
        self.car_count = 0
        self.bicycle_count = 0
        self.tracked_objects = {}
        self.object_history = defaultdict(list)
        self.next_id = 0
        self.frame_count = 0
        
    def detect_objects(self, frame, canny_low=50, canny_high=150, 
                      hough_threshold=50, circles_param2=30):
        """كشف السيارات والدراجات من الإطار"""
        
        # تحويل إلى رمادي
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # تطبيق Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny Edge Detection
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # البحث عن الدوائر (العجل)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=canny_high,
            param2=circles_param2,
            minRadius=10,
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
        """تصنيف الكائنات المكتشفة"""
        
        objects = []
        circles = detection_result['circles']
        lines = detection_result['lines']
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # تجميع الدوائر (العجل المتقارب)
            circle_list = []
            for i in circles[0, :]:
                circle_list.append((i[0], i[1], i[2]))
            
            # البحث عن الدراجات (2 دوائر متقاربة)
            bicycles = self._find_bicycles(circle_list)
            for bicycle in bicycles:
                objects.append({'type': 'bicycle', 'circles': bicycle})
            
            # البحث عن السيارات (2+ دوائر + خطوط)
            if lines is not None and len(lines) > 0:
                cars = self._find_cars(circle_list, lines)
                for car in cars:
                    objects.append({'type': 'car', 'circles': car['circles'], 'lines': car['lines']})
        
        return objects
    
    def _find_bicycles(self, circles):
        """البحث عن الدراجات (دائرتان متقاربتان)"""
        bicycles = []
        used = set()
        
        for i, (x1, y1, r1) in enumerate(circles):
            if i in used:
                continue
            
            for j, (x2, y2, r2) in enumerate(circles):
                if i >= j or j in used:
                    continue
                
                # المسافة بين الدائرتين
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                # إذا كانا متقاربتان (مثل عجلات الدراجة)
                if distance < 200 and distance > 30:
                    bicycles.append([(x1, y1, r1), (x2, y2, r2)])
                    used.add(i)
                    used.add(j)
                    break
        
        return bicycles
    
    def _find_cars(self, circles, lines):
        """البحث عن السيارات"""
        cars = []
        used_circles = set()
        
        if lines is None or len(lines) == 0:
            return cars
        
        # تجميع الخطوط (أفقية وعمودية)
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # حساب الزاوية
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
        
        # السيارة عادة تحتوي على 2+ خطوط و 2-4 دوائر
        if len(horizontal_lines) >= 1 and len(vertical_lines) >= 1:
            for circle in circles:
                if len(circles) >= 2:  # سيارة لها عجلتان على الأقل
                    cars.append({
                        'circles': [circle],
                        'lines': [horizontal_lines[0], vertical_lines[0]]
                    })
                    used_circles.add(circles.index(circle))
        
        return cars
    
    def draw_results(self, frame, detection_result, objects, 
                    canny_low=50, canny_high=150, 
                    hough_threshold=50, circles_param2=30):
        """رسم النتائج على الإطار"""
        
        result_frame = frame.copy()
        circles = detection_result['circles']
        lines = detection_result['lines']
        
        # رسم الدوائر (العجل)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                # دائرة خضراء للعجل
                cv2.circle(result_frame, center, radius, (0, 255, 0), 2)
                cv2.circle(result_frame, center, 2, (0, 255, 0), 3)
        
        # رسم الخطوط
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # خطوط زرقاء
                cv2.line(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # رسم البounding boxes والكائنات المصنفة
        car_count = 0
        bicycle_count = 0
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for idx, circle_group in enumerate(objects):
                if circle_group['type'] == 'car':
                    car_count += 1
                    # Bounding box أحمر للسيارات
                    for circle in circle_group.get('circles', []):
                        cv2.rectangle(result_frame, 
                                    (circle[0]-circle[2]-30, circle[1]-circle[2]-30),
                                    (circle[0]+circle[2]+30, circle[1]+circle[2]+30),
                                    (0, 0, 255), 2)
                        cv2.putText(result_frame, 'CAR', 
                                  (circle[0]-30, circle[1]-40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                elif circle_group['type'] == 'bicycle':
                    bicycle_count += 1
                    # Bounding box أزرق للدراجات
                    for circle in circle_group.get('circles', []):
                        cv2.rectangle(result_frame,
                                    (circle[0]-circle[2]-30, circle[1]-circle[2]-30),
                                    (circle[0]+circle[2]+30, circle[1]+circle[2]+30),
                                    (255, 0, 0), 2)
                    cv2.putText(result_frame, 'BICYCLE',
                              (circle_group['circles'][0][0]-50, circle_group['circles'][0][1]-40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # كتابة الإحصائيات
        cv2.putText(result_frame, f"Cars: {car_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Bicycles: {bicycle_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(result_frame, f"Frame: {self.frame_count}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame, car_count, bicycle_count


def main():
    if len(sys.argv) < 2:
        print("استخدام: python main.py <video_path>")
        print("مثال: python main.py cars.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ الملف غير موجود: {video_path}")
        sys.exit(1)
    
    # فتح الفيديو
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ لا يمكن فتح الفيديو: {video_path}")
        sys.exit(1)
    
    detector = ObjectDetector()
    
    # إنشاء نافذة للتحكم
    cv2.namedWindow("Object Detection Control")
    
    # Trackbars للتحكم
    cv2.createTrackbar("Canny Low", "Object Detection Control", 50, 300, lambda x: None)
    cv2.createTrackbar("Canny High", "Object Detection Control", 150, 300, lambda x: None)
    cv2.createTrackbar("Hough Threshold", "Object Detection Control", 50, 200, lambda x: None)
    cv2.createTrackbar("Circles Param2", "Object Detection Control", 30, 100, lambda x: None)
    
    paused = False
    total_cars = 0
    total_bicycles = 0
    
    print("🎬 تشغيل الفيديو...")
    print("🎮 التحكم:")
    print("  SPACE - توقيف/تشغيل")
    print("  R - تصفير العداد")
    print("  S - حفظ صورة")
    print("  ESC/Q - خروج")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("✅ انتهى الفيديو!")
                break
        
        # الحصول على قيم Trackbars
        canny_low = cv2.getTrackbarPos("Canny Low", "Object Detection Control")
        canny_high = cv2.getTrackbarPos("Canny High", "Object Detection Control")
        hough_threshold = cv2.getTrackbarPos("Hough Threshold", "Object Detection Control")
        circles_param2 = cv2.getTrackbarPos("Circles Param2", "Object Detection Control")
        
        # كشف الكائنات
        detection = detector.detect_objects(frame, canny_low, canny_high, 
                                          hough_threshold, circles_param2)
        
        # تصنيف الكائنات
        objects = detector.classify_objects(detection)
        
        # رسم النتائج
        result_frame, car_count, bicycle_count = detector.draw_results(
            frame, detection, objects,
            canny_low, canny_high, hough_threshold, circles_param2
        )
        
        total_cars = max(total_cars, car_count)
        total_bicycles = max(total_bicycles, bicycle_count)
        
        # عرض الكنترول
        control_frame = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(control_frame, f"Canny Low: {canny_low}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(control_frame, f"Canny High: {canny_high}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(control_frame, f"Hough Threshold: {hough_threshold}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(control_frame, f"Circles Param2: {circles_param2}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(control_frame, "Status: " + ("PAUSED" if paused else "PLAYING"),
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if paused else (0, 255, 0), 2)
        
        cv2.imshow("Object Detection Control", control_frame)
        cv2.imshow("Video Processing", result_frame)
        
        detector.frame_count += 1
        
        # معالجة المفاتيح
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:  # ESC أو Q
            print("👋 تم الخروج!")
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"{'⏸️ متوقف' if paused else '▶️ يعمل'}")
        elif key == ord('r'):  # R
            print("🔄 تم تصفير العداد!")
        elif key == ord('s'):  # S
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, result_frame)
            print(f"📸 تم حفظ الصورة: {filename}")
    
    print(f"\n📊 النتائج النهائية:")
    print(f"  🚗 السيارات: {total_cars}")
    print(f"  🚲 الدراجات: {total_bicycles}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
