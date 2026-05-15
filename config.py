# إعدادات المشروع

# Canny Edge Detection
CANNY_LOW_DEFAULT = 50
CANNY_HIGH_DEFAULT = 150

# Hough Transform
HOUGH_THRESHOLD_DEFAULT = 50
CIRCLES_PARAM2_DEFAULT = 30
MIN_DISTANCE_DEFAULT = 50

# معاملات البحث
MIN_RADIUS_CIRCLES = 8
MAX_RADIUS_CIRCLES = 100
MIN_LINE_LENGTH = 30
MAX_LINE_GAP = 10

# معاملات التتبع
MIN_DISTANCE_BETWEEN_CIRCLES = 30
MAX_DISTANCE_BETWEEN_CIRCLES = 250

# قيم الألوان (BGR)
COLOR_GREEN = (0, 255, 0)      # الدوائر والعجل
COLOR_BLUE = (255, 0, 0)       # الخطوط والدراجات
COLOR_RED = (0, 0, 255)        # السيارات والنصوص
COLOR_WHITE = (255, 255, 255)  # النصوص

# معاملات التطبيق
FRAME_SCALE = 0.5
FRAME_DISPLAY_SIZE = 640
