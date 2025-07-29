import streamlit as st
import pandas as pd
import joblib
import numpy as np


# --- Helper Function for the new report ---
def get_credit_grade(score):
    """Maps a credit score to a grade based on NCB criteria."""
    if 753 <= score <= 900:
        return "AA", "ดีเยี่ยม"
    elif 725 <= score <= 752:
        return "BB", "ดี"
    elif 699 <= score <= 724:
        return "CC", "ดีพอใช้"
    elif 681 <= score <= 698:
        return "DD", "ปานกลาง"
    elif 666 <= score <= 680:
        return "EE", "ควรปรับปรุง"
    elif 616 <= score <= 665:
        return "FF", "ต้องปรับปรุง"
    elif 300 <= score <= 615:
        return "HH", "มีความเสี่ยงสูง"
    else:
        return "N/A", "ไม่สามารถประเมินได้"


def get_credit_reasons(score, data):
    """Generates plausible and consistent reasons based on the score and input data."""
    reasons = []

    # --- Logic for HIGH credit score ---
    if score >= 680:
        # Check for positive factors
        if data['membership_duration_months'] > 120:
            reasons.append("✅ มีประวัติการเป็นสมาชิกที่ยาวนาน")
        if data['job_completion_rate'] > 95.0:
            reasons.append("✅ มีอัตราการทำงานสำเร็จในระดับสูง")
        if data['customer_rating_avg'] > 4.5:
            reasons.append("✅ ได้รับคะแนนเฉลี่ยจากลูกค้าในระดับดีเยี่ยม")
        if data['work_consistency_index'] > 0.9:
            reasons.append("✅ มีความสม่ำเสมอในการทำงานสูง")

        # Default positive reason if no specific positive factors are met
        if not reasons:
            reasons.append("✅ มีประวัติทางการเงินและประสิทธิภาพการทำงานโดยรวมอยู่ในเกณฑ์ดี")

    # --- Logic for LOW credit score ---
    else:  # score < 680
        # Check for negative factors
        if data['Loan_Amount'] > (data['Monthly_Income'] * 5) and data['Monthly_Income'] > 0:
            reasons.append("⚠️ สัดส่วนยอดหนี้สินเชื่อต่อรายได้ค่อนข้างสูง")
        if data['Work_Experience'] < 2:
            reasons.append("⚠️ มีประสบการณ์ทำงานค่อนข้างน้อย")
        if data['job_cancellation_count'] > 10:
            reasons.append("⚠️ มีประวัติการยกเลิกงานที่เกิดขึ้นบ่อยครั้ง")
        if data['inactive_days_last_30'] > 15:
            reasons.append("⚠️ มีจำนวนวันที่ไม่ทำงานในช่วง 30 วันที่ผ่านมาค่อนข้างสูง")
        if data['rejected_jobs_last_30'] > 5:
            reasons.append("⚠️ มีประวัติการปฏิเสธงานที่เกิดขึ้นบ่อยครั้ง")

        # Default negative reason if no specific negative factors are met
        if not reasons:
            reasons.append("⚠️ ควรพิจารณาปรับปรุงประวัติทางการเงินและประสิทธิภาพการทำงาน")

    return reasons[:4]  # Return max 4 reasons

# --- 1. โหลด Model และกำหนด Mapping ---
# ใช้ try-except เพื่อป้องกันข้อผิดพลาดหากหาไฟล์ไม่เจอ
try:
    model = joblib.load("loan_model_extended_muticlass_randomforest_credit_score.pkl")
except FileNotFoundError:
    st.error("ไม่พบไฟล์โมเดลที่จำเป็น (loan_model...pkl). กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ 'models'")
    st.stop()  # หยุดการทำงานของแอปถ้าไม่มีโมเดล



# Manual mapping สำหรับแปลงค่าจากข้อความเป็นตัวเลข
education_map = {'Vocational': 0, 'Secondary': 1, 'Primary': 2, 'None': 3}
loan_purpose_map = {'business': 0, 'personal': 1}
home_ownership_map = {'own': 0, 'rent': 1}
certificate_map = {'Yes': 0, 'No': 1}
gender_map = {'Male': 0, 'Female': 1}
marital_status_map = {'Single': 0, 'Married': 1, 'Divorced': 2}
region_map = {'North': 0, 'Central': 1, 'South': 2, 'East': 3, 'West': 4}
occupation_map = {'Private': 0, 'Government': 1, 'Freelancer': 2, 'Unemployed': 3}

# --- 2. ตั้งค่าหน้าจอและหัวข้อ ---
st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
st.title("🎯 Project: AI-Powered Credit rating service (3-Class)")

# --- 3. สร้าง Form เพื่อรับข้อมูลทั้งหมดในครั้งเดียว ---
with st.form("loan_application_form"):
    st.subheader("📋 Credit Rating Service")

    # === ข้อมูลผู้สมัคร ===
   # col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
   # col1.write("Feature Type")
   # col2.write("Feature (English)")
   # col3.write("คำแปลภาษาไทย")
   # worker_id = col4.text_input("worker_id_input", label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("ข้อมูลผู้สมัคร")
    # col2.write("Gender")
    col3.write("เพศ")
    Gender = col4.selectbox("gender_input", list(gender_map.keys()), label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("")
   # col2.write("Age")
    col3.write("อายุ")
    Age = col4.slider("age_input", 18, 70, 30, label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("")
   # col2.write("Marital Status")
    col3.write("สถานภาพสมรส")
    Marital_Status = col4.selectbox("marital_status_input", list(marital_status_map.keys()),
                                    label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("")
   # col2.write("Education")
    col3.write("ระดับการศึกษา")
    Education = col4.selectbox("education_input", list(education_map.keys()), label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("")
   # col2.write("Occupation")
    col3.write("อาชีพ")
    Occupation = col4.selectbox("occupation_input", list(occupation_map.keys()), label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("")
  #  col2.write("Work Experience")
    col3.write("ประสบการณ์ทำงาน (ปี)")
    Work_Experience = col4.slider("work_experience_input", 0, 40, 5, label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write(" ")
   # col2.write("Certificate")
    col3.write("ใบรับรอง")
    Certificate = col4.selectbox("certificate_input", list(certificate_map.keys()), label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("")
   # col2.write("Region")
    col3.write("ภูมิภาค")
    Region = col4.selectbox("region_input", list(region_map.keys()), label_visibility="collapsed")

    # === ข้อมูลการเงินและสินเชื่อ ===
    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write(" ")
  #  col2.write("Monthly Income")
    col3.write("รายได้ต่อเดือน")
    Monthly_Income = col4.number_input("monthly_income_input", min_value=0.0, value=25000.0,
                                       label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("")
   # col2.write("Home Ownership")
    col3.write("สถานะที่อยู่อาศัย")
    home_ownership = col4.selectbox("home_ownership_input", list(home_ownership_map.keys()),
                                    label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("")
   # col2.write("Dependents")
    col3.write("จำนวนผู้อยู่ในอุปการะ")
    dependents = col4.slider("dependents_input", 0, 10, 1, label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("ข้อมูลสินเชื่อที่ต้องการ")
   # col2.write("Loan Amount")
    col3.write("จำนวนเงินที่ขอกู้")
    Loan_Amount = col4.number_input("loan_amount_input", min_value=0.0, value=10000.0, label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("")
   # col2.write("Loan Purpose")
    col3.write("วัตถุประสงค์ของการกู้")
    loan_purpose = col4.selectbox("loan_purpose_input", list(loan_purpose_map.keys()), label_visibility="collapsed")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    col1.write("ข้อมูลเครดิตบูโร")
  #  col2.write("simulated credit score")
    col3.write("้คะแนนเครดิต")
    simulated_credit_score = col4.slider("simulated_credit_score_input", 400, 900, 600, label_visibility="collapsed")

  #  st.divider()
  #  st.subheader("📊 ข้อมูลทางเลือก (Performance Metrics)")

    # === Performance/Activity Metrics ===
    metrics_values = {}
  #  metrics_config = [
  #      ("job_completion_rate", "Job Completion Rate (%)", "อัตราสำเร็จงาน", 0.0, 100.0, 85.0),
  #      ("on_time_rate", "On Time Rate (%)", "อัตราตรงเวลา", 0.0, 100.0, 90.0),
  #      ("avg_response_time_mins", "Avg. Response Time (mins)", "เวลาตอบกลับเฉลี่ย (นาที)", 0.0, 120.0, 10.0),
  #      ("customer_rating_avg", "Customer Rating Avg.", "คะแนนเฉลี่ยจากลูกค้า", 0.0, 5.0, 4.2),
  #      ("job_acceptance_rate", "Job Acceptance Rate (%)", "อัตรารับงาน", 0.0, 100.0, 80.0),
  #      ("job_cancellation_count", "Job Cancellation Count", "จำนวนยกเลิกงาน", 0, 100, 2),
  #      ("weekly_active_days", "Weekly Active Days", "วันทำงานต่อสัปดาห์", 0, 7, 5),
  #      ("membership_duration_months", "Membership Duration (months)", "ระยะเวลาสมาชิก (เดือน)", 0, 240, 24),
  #     # ("simulated_credit_score", "Simulated Credit Score", "คะแนนเครดิตจำลอง", 300, 850, 650),
  #      ("work_consistency_index", "Work Consistency Index", "ดัชนีความสม่ำเสมอ", 0.0, 1.0, 0.75),
  #      ("inactive_days_last_30", "Inactive Days (last 30)", "วันที่ไม่ทำงานใน 30 วัน", 0, 30, 3),
  #      ("rejected_jobs_last_30", "Rejected Jobs (last 30)", "จำนวนงานที่ปฏิเสธใน 30 วัน", 0, 30, 1),
  #  ]

    metrics_config = [
         ("job_completion_rate", " ", "อัตราสำเร็จงาน", 0.0, 100.0, 85.0),
         ("on_time_rate", " ", "อัตราตรงเวลา", 0.0, 100.0, 90.0),
         ("avg_response_time_mins", " ", "เวลาตอบกลับเฉลี่ย (นาที)", 0.0, 120.0, 10.0),
         ("customer_rating_avg", " ", "คะแนนเฉลี่ยจากลูกค้า", 0.0, 5.0, 4.2),
         ("job_acceptance_rate", " ", "อัตรารับงาน", 0.0, 100.0, 80.0),
         ("job_cancellation_count", " ", "จำนวนยกเลิกงาน", 0, 100, 2),
         ("weekly_active_days", " ", "วันทำงานต่อสัปดาห์", 0, 7, 5),
         ("membership_duration_months", " ", "ระยะเวลาสมาชิก (เดือน)", 0, 240, 24),
    #     # ("simulated_credit_score", "Simulated Credit Score", "คะแนนเครดิตจำลอง", 300, 850, 650),
         ("work_consistency_index", " ", "ดัชนีความสม่ำเสมอ", 0.0, 1.0, 0.75),
         ("inactive_days_last_30", " ", "วันที่ไม่ทำงานใน 30 วัน", 0, 30, 3),
         ("rejected_jobs_last_30", " ", "จำนวนงานที่ปฏิเสธใน 30 วัน", 0, 30, 1),
      ]

    # --- สร้าง Header ของตารางข้อมูลทางเลือก (ทำครั้งเดียว) ---
   # col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
   # col1.write("ข้อมูลทางเลือก")
   # col2.write("**Metric (EN)**")
   # col3.write("**เมตริก (TH)**")

    first_time = True

    for var_name, label_en, label_th, min_val, max_val, default in metrics_config:
        col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
        if first_time:
            col1.write("ข้อมูลทางเลือก")
            first_time = False
        else:
            col1.write("")  # หรือเว้นว่างไม่แสดงอะไร

        col2.write(label_en)
        col3.write(label_th)
        if isinstance(default, float):
            metrics_values[var_name] = col4.slider(var_name, float(min_val), float(max_val), float(default),
                                                   label_visibility="collapsed")
        else:
            metrics_values[var_name] = col4.slider(var_name, int(min_val), int(max_val), int(default), label_visibility="collapsed")

        # --- CSS for the submit button ---
        st.markdown("""
        <style>
        div.stButton > button {
            background-color: #FF8C00; /* Orange color */
            color: white;
            width: 100%;
            border-radius: 5px;
            border: none;
        }
        </style>
        """, unsafe_allow_html=True)

    colss = st.columns([6, 1])
    with colss[1]:
         submitted = st.form_submit_button("ประเมินการขอสินเชื่อ")

# --- 4. การประมวลผลจะเกิดขึ้นหลังกด Submit เท่านั้น ---
if submitted:
    # สร้าง Dictionary ของข้อมูลทั้งหมดเพื่อสร้าง DataFrame
    data_to_predict = {
        "Gender": gender_map[Gender],
        "Age": Age,
        "Occupation": occupation_map[Occupation],
        "Education": education_map[Education],
        "Marital_Status": marital_status_map[Marital_Status],
        "Work_Experience": Work_Experience,
        "Certificate": certificate_map[Certificate],
        "Region": region_map[Region],
        "Monthly_Income": Monthly_Income,
        "Loan_Amount": Loan_Amount,
        "loan_purpose": loan_purpose_map[loan_purpose],
        "home_ownership": home_ownership_map[home_ownership],
        "dependents": dependents,
        "job_completion_rate": metrics_values["job_completion_rate"],
        "on_time_rate": metrics_values["on_time_rate"],
        "avg_response_time_mins": metrics_values["avg_response_time_mins"],
        "customer_rating_avg": metrics_values["customer_rating_avg"],
        "job_acceptance_rate": metrics_values["job_acceptance_rate"],
        "job_cancellation_count": metrics_values["job_cancellation_count"],
        "weekly_active_days": metrics_values["weekly_active_days"],
        "membership_duration_months": metrics_values["membership_duration_months"],
        "simulated_credit_score": simulated_credit_score,
        "work_consistency_index": metrics_values["work_consistency_index"],
        "inactive_days_last_30": metrics_values["inactive_days_last_30"],
        "rejected_jobs_last_30": metrics_values["rejected_jobs_last_30"]
       # **metrics_values  # นำค่าจาก sliders ทั้งหมดมารวมกัน
    }

    input_df = pd.DataFrame([data_to_predict])

    st.write("---")
    st.subheader("ผลการทำนาย (Prediction Result)")

    # ทำนายผล
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # แสดงผลลัพธ์
    ##    st.success(f"**ผลการประเมินสถานะ: {prediction}**")

        # แสดงความน่าจะเป็น
    ##    proba_df = pd.DataFrame({
    ##        'สถานะ (Status)': model.classes_,
    ##        'ความน่าจะเป็น (Probability)': prediction_proba
    ##    })
    ##    st.write("รายละเอียดความน่าจะเป็น:")
    ##    st.dataframe(proba_df.style.format({'ความน่าจะเป็น (Probability)': '{:.2%}'}))

    ## except Exception as e:
    ##    st.error(f"เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")
        # --- ส่วนแสดงผลที่ออกแบบใหม่ ---
        st.markdown("""
            <style>
            .report-container {
                border: 2px solid #1E90FF;
                border-radius: 10px;
                padding: 20px;
                background-color: #F0F8FF;
            }
            .report-header {
                color: #1E90FF;
                text-align: center;
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown('<h2 class="report-header">ตัวอย่างรายงานเครดิตบูโร (คะแนนเครดิต)</h2>', unsafe_allow_html=True)

            # --- NEW LAYOUT PART 1: Top metrics ---
            score = simulated_credit_score
            grade, grade_desc = get_credit_grade(score)

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(label=f"คะแนนเครดิต (เกรด: {grade})", value=score)

            with res_col2:
                status_map = {0: "มีความเสี่ยงสูง (ไม่อนุมัติ)", 1: "มีความเสี่ยงต่ำ (อนุมัติ)", 2: "รอการตรวจสอบเพิ่มเติม"}
                status_color = {0: "red", 1: "green", 2: "orange"}
                #status_map = {0: "มีความเสี่ยงต่ำ (อนุมัติ)", 1: "รอการตรวจสอบเพิ่มเติม", 2: "มีความเสี่ยงสูง (ไม่อนุมัติ)"}
                #status_color = {0: "green", 1: "orange", 2: "red"}
                st.markdown("##### **ผลการประเมินโดย AI**")
                st.markdown(
                    f"<h4 style='color:{status_color.get(prediction, 'black')};'>{status_map.get(prediction, 'N/A')}</h4>",
                    unsafe_allow_html=True)

            with res_col3:
                prob_default_index = np.where(model.classes_ == 0)[0][0]
                #prob_default_index = np.where(model.classes_ == 2)[0][0]
                prob_default = prediction_proba[prob_default_index]
                st.metric(label="ความน่าจะเป็นในการผิดนัดชำระ", value=f"{prob_default:.2%}")

            st.markdown("<br>", unsafe_allow_html=True)  # Add some space

            # --- NEW LAYOUT PART 2: Table on the left ---
            table_col1, table_col2 = st.columns([1, 1])
            with table_col1:
                st.markdown("##### **ตารางคะแนนเครดิต**")
                score_table = {
                    "เกรด": ["AA", "BB", "CC", "DD", "EE", "FF", "HH"],
                    "ช่วงคะแนน": ["753-900", "725-752", "699-724", "681-698", "666-680", "616-665", "300-615"]
                }
                score_df = pd.DataFrame(score_table)


                def highlight_grade(s):
                    return ['background-color: #1E90FF; color: white' if s.เกรด == grade else '' for i in s]


                st.dataframe(score_df.style.apply(highlight_grade, axis=1), use_container_width=True)

            with table_col2:
                # --- NEW REASON SECTION ---
                st.markdown("##### **เหตุผลประกอบคะแนนเครดิต**")
                reasons = get_credit_reasons(score, data_to_predict)
                st.markdown('<div class="reason-box">', unsafe_allow_html=True)
                for reason in reasons:
                    st.write(reason)
                st.markdown('</div>', unsafe_allow_html=True)

            #with table_col3:
            #    st.write("")  # Empty column for spacing

            st.markdown("---")
            st.info("**หมายเหตุ:** รายงานนี้เป็นผลการประเมินเบื้องต้นโดยใช้ข้อมูลที่ท่านกรอกและโมเดลปัญญาประดิษฐ์เท่านั้น")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")
