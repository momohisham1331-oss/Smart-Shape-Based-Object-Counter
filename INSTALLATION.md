# 📥 دليل التثبيت والتشغيل

## المتطلبات
- Python 3.8+
- pip (مدير الحزم)

## 🚀 خطوات التثبيت

### 1. استنساخ المشروع
```bash
git clone https://github.com/momohisham1331-oss/Smart-Shape-Based-Object-Counter.git
cd Smart-Shape-Based-Object-Counter
```

### 2. إنشاء بيئة افتراضية (اختياري لكن مُنصح)
```bash
python -m venv venv

# على Windows:
venv\Scripts\activate

# على Mac/Linux:
source venv/bin/activate
```

### 3. تثبيت المكتبات
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. تشغيل التطبيق
```bash
streamlit run app.py
```

## ✅ النتيجة المتوقعة
- سيفتح متصفح تلقائياً
- العنوان: `http://localhost:8501`
- واجهة سهلة الاستخدام

## 🎯 الاستخدام

1. **اختر الفيديو**: من الـ Sidebar أيمن
2. **اضبط المعاملات**:
   - Canny Low/High
   - Hough Threshold
   - Circles Param2
3. **اضغط**: 🚀 معالجة الفيديو
4. **شاهد**: النتائج والإحصائيات

## 🐛 حل المشاكل

### لا تعمل المكتبات؟
```bash
pip install --upgrade pip setuptools wheel
pip install --only-binary :all: numpy
pip install -r requirements.txt
```

### الفيديو بطيء جداً؟
- استخدم فيديو بدقة أقل
- قلل عدد الإطارات المعالجة

### لا يتم الكشف؟
- اضبط Canny Low و Canny High
- قلل Hough Threshold
- زد Circles Param2

---

**تم! الآن أنت جاهز للبدء! 🚀**
