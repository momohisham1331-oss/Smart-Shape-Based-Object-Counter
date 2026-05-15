import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
import tempfile
import os
from detector import ObjectDetector

st.set_page_config(
    page_title="Smart Object Counter",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.title("🚗 Smart Shape-Based Object Counter & Analyzer")
st.markdown("---")

# الـ Sidebar
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # Upload الفيديو
    uploaded_file = st.file_uploader(
        "📹 اختر الفيديو",
        type=["mp4", "avi", "mov", "mkv", "flv", "wmv"]
    )
    
    st.markdown("---")
    st.subheader("🎛️ معاملات الكشف")
    
    # Canny parameters
    canny_low = st.slider(
        "Canny Low Threshold",
        0, 300, 50,
        help="الحد الأدنى لكشف الحواف"
    )
    
    canny_high = st.slider(
        "Canny High Threshold",
        0, 300, 150,
        help="الحد الأقصى لكشف الحواف"
    )
    
    # Hough Transform parameters
    hough_threshold = st.slider(
        "Hough Threshold (Lines)",
        0, 200, 50,
        help="حساسية كشف الخطوط"
    )
    
    circles_param2 = st.slider(
        "Circles Param2",
        0, 100, 30,
        help="حساسية كشف الدوائر (العجل)"
    )
    
    min_distance = st.slider(
        "Min Distance Between Circles",
        10, 200, 50,
        help="الحد الأدنى للمسافة بين الدوائر"
    )
    
    st.markdown("---")
    st.subheader("📊 خيارات العرض")
    
    show_circles = st.checkbox("عرض الدوائر (العجل)", value=True)
    show_lines = st.checkbox("عرض الخطوط", value=True)
    show_edges = st.checkbox("عرض حواف Canny", value=False)
    show_stats = st.checkbox("عرض الإحصائيات", value=True)
    
    st.markdown("---")
    process_button = st.button(
        "🚀 معالجة الفيديو",
        use_container_width=True,
        type="primary"
    )

# المحتوى الرئيسي
if uploaded_file is not None:
    # حفظ الفيديو مؤقتاً
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name
    
    if process_button:
        st.info("⏳ جاري معالجة الفيديو... يرجى الانتظار")
        
        # فتح الفيديو
        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            st.error("❌ لا يمكن فتح الفيديو")
        else:
            # الحصول على معلومات الفيديو
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # عمود للفيديو الأصلي
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📹 الفيديو الأصلي")
                video_placeholder = st.empty()
            
            with col2:
                st.subheader("🎯 النتائج المعالجة")
                result_placeholder = st.empty()
            
            # إنشاء المكتشف
            detector = ObjectDetector()
            
            # إنشاء عداد وإحصائيات
            stats_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_count = 0
            total_cars = 0
            total_bicycles = 0
            frame_cars = defaultdict(int)
            frame_bicycles = defaultdict(int)
            
            processed_frames = []
            
            # معالجة الإطارات
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # تقليل حجم الإطار للسرعة
                scale = 0.5
                frame_resized = cv2.resize(frame, (int(width*scale), int(height*scale)))
                
                # الكشف
                detection_result = detector.detect_objects(
                    frame_resized,
                    canny_low, canny_high,
                    hough_threshold, circles_param2,
                    min_distance
                )
                
                # التصنيف
                objects = detector.classify_objects(detection_result)
                
                # الرسم
                result_frame, car_count, bicycle_count = detector.draw_results(
                    frame_resized, detection_result, objects,
                    canny_low, canny_high, hough_threshold, circles_param2,
                    show_circles, show_lines, show_edges
                )
                
                total_cars = max(total_cars, car_count)
                total_bicycles = max(total_bicycles, bicycle_count)
                frame_cars[frame_count] = car_count
                frame_bicycles[frame_count] = bicycle_count
                
                # تحويل BGR إلى RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # عرض الإطارات
                with col1:
                    video_placeholder.image(frame_rgb, use_column_width=True)
                
                with col2:
                    result_placeholder.image(result_rgb, use_column_width=True)
                
                # عرض الإحصائيات
                if show_stats:
                    with stats_placeholder.container():
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            st.metric(
                                "🚗 السيارات",
                                total_cars,
                                delta=None
                            )
                        
                        with stat_col2:
                            st.metric(
                                "🚲 الدراجات",
                                total_bicycles,
                                delta=None
                            )
                        
                        with stat_col3:
                            st.metric(
                                "📹 الإطار",
                                f"{frame_count}/{total_frames}",
                                delta=None
                            )
                        
                        with stat_col4:
                            st.metric(
                                "⏱️ الوقت",
                                f"{frame_count/fps:.1f}s",
                                delta=None
                            )
                
                # تحديث شريط التقدم
                progress = min(frame_count / max(total_frames, 1), 1.0)
                progress_bar.progress(progress)
                
                frame_count += 1
            
            cap.release()
            
            # النتائج النهائية
            st.markdown("---")
            st.subheader("📊 النتائج النهائية")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "🚗 إجمالي السيارات",
                    total_cars,
                    "+" + str(total_cars) if total_cars > 0 else "0"
                )
            
            with col_b:
                st.metric(
                    "🚲 إجمالي الدراجات",
                    total_bicycles,
                    "+" + str(total_bicycles) if total_bicycles > 0 else "0"
                )
            
            with col_c:
                st.metric(
                    "📹 عدد الإطارات",
                    frame_count,
                    f"{frame_count/total_frames*100:.1f}%" if total_frames > 0 else "0%"
                )
            
            # رسم بياني للإحصائيات
            st.subheader("📈 الرسم البياني للكشف")
            
            import pandas as pd
            import plotly.graph_objects as go
            
            df_data = {
                'Frame': list(frame_cars.keys()),
                'Cars': list(frame_cars.values()),
                'Bicycles': list(frame_bicycles.values())
            }
            
            df = pd.DataFrame(df_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Frame'],
                y=df['Cars'],
                mode='lines+markers',
                name='السيارات',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=df['Frame'],
                y=df['Bicycles'],
                mode='lines+markers',
                name='الدراجات',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title="عدد الكائنات المكتشفة عبر الإطارات",
                xaxis_title="الإطار",
                yaxis_title="العدد",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 اضبط الإعدادات واضغط 'معالجة الفيديو' للبدء")
    
    # حذف الملف المؤقت
    try:
        os.unlink(temp_video_path)
    except:
        pass

else:
    st.warning("📹 يرجى اختيار فيديو من الـ Sidebar")
    
    # معلومات المشروع
    st.markdown("""
    ## 📚 معلومات المشروع
    
    ### 🎯 المميزات:
    - ✅ كشف السيارات والدراجات من الفيديوهات
    - ✅ تحليل متقدم للأشكال (دوائر وخطوط)
    - ✅ عداد ذكي بدون تكرار
    - ✅ Trackbars تفاعلية للتحكم
    - ✅ عرض فوري للنتائج
    - ✅ رسوم بيانية للإحصائيات
    
    ### 🎛️ كيفية الاستخدام:
    1. اختر الفيديو من الـ Sidebar
    2. اضبط معاملات الكشف حسب الحاجة
    3. اضغط 'معالجة الفيديو'
    4. شاهد النتائج والإحصائيات
    
    ### 📖 المعاملات:
    - **Canny Low/High**: التحكم بكشف الحواف
    - **Hough Threshold**: حساسية كشف الخطوط
    - **Circles Param2**: حساسية كشف الدوائر
    - **Min Distance**: الحد الأدنى للمسافة بين الكائنات
    
    ---
    
    **🚀 ابدأ الآن بتحميل فيديو!**
    """)
