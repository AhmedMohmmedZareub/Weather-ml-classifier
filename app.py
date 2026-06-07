import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. عنوان الصفحة النهائية على المتصفح
st.set_page_config(page_title="Weather ML Classifier", page_icon="🌤️")
st.title("🌤️ نظام التنبؤ الذكي بحالة الطقس - غزة")
st.write("هذا التطبيق يعتمد على خوارزمية Random Forest للتنبؤ بصلاحية الطقس للأنشطة الخارجية.")

# 2. تدريب النموذج خلف الكواليس بناءً على البيانات
data = {
    'Temperature': [25, 29, 15, 18, 32, 12, 22, 28, 14, 30],
    'WindSpeed': [12, 17, 25, 22, 10, 30, 15, 14, 28, 11],
    'Humidity': [60, 65, 80, 75, 50, 90, 55, 62, 85, 48],
    'GoOut': [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)
X = df[['Temperature', 'WindSpeed', 'Humidity']]
y = df['GoOut']

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 3. تصميم واجهة المستخدم (المدخلات)
st.sidebar.header("📊 أدخل مؤشرات الطقس الحالية:")
temp = st.sidebar.slider("درجة الحرارة (C°)", 10, 40, 25)
wind = st.sidebar.slider("سرعة الرياح (km/h)", 5, 50, 15)
humidity = st.sidebar.slider("نسبة الرطوبة (%)", 30, 100, 60)

# 4. زر التنبؤ والمخرجات النهائية
if st.button("شغل نموذج الذكاء الاصطناعي وافحص الحالة"):
    prediction = model.predict([[temp, wind, humidity]])
    
    st.subheader("🤖 النتيجة النهائية للتنبؤ:")
    if prediction[0] == 1:
        st.success("✅ الطقس ممتاز ومناسب جداً للخروج والأنشطة الخارجية!")
    else:
        st.error("❌ تحذير: الطقس غير مناسب للخروج بناءً على تحليلات النموذج.")