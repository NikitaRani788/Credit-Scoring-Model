import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="wide"
)
# -----------------------------
#Load the model and feature names
# -----------------------------
model = joblib.load("models/best_credit_score.pkl")
feature_names = joblib.load("models/feature_names.pkl")
#st.success("✅ Model Loaded Successfully!")

#st.write("Number of Features:", len(feature_names))
#st.subheader("Features Used by the Model")
#st.write(feature_names)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("💳 Credit Score Prediction")

st.sidebar.markdown("---")
# Navigation
page = st.sidebar.radio(
    "📌 Navigation",
    [
        "🏠 Credit Score Prediction",
        "📊 Analytics Dashboard",
        "ℹ️ About Project"
    ]
)
st.sidebar.subheader("**MODEL**")
st.sidebar.write("Tuned Random Forest")
st.sidebar.write("**Accuracy:** 79.47%") 

st.sidebar.markdown("---")

st.sidebar.subheader("Developer")
st.sidebar.write("Nikita Rani")

st.sidebar.markdown("---")

if page == "🏠 Credit Score Prediction":

        # -----------------------------
    # Main Page
    # -----------------------------
    st.title("💳 Credit Score Prediction System")

    st.markdown("""
    Welcome to the **Credit Score Prediction System**.

    Predict customer credit scores using a **Tuned Random Forest Classifier** based on financial and credit behaviour.

    ### Credit Score Categories

    - 🟢 Good
                
    Excellent financial behaviour.
    - 🟡 Standard
                
    Moderate credit profile.
    - 🔴 Poor
                
    Needs financial improvement.

    Please enter the customer details in the input fields to get the prediction.
    """)

    st.markdown("---")

    #Personal details
    st.header("👤 Personal Information")

    with st.expander("Enter Personal Details", expanded=True):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=25
            )

            annual_income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=50000.0
            )

        with col2:
            monthly_salary = st.number_input(
                "Monthly In-hand Salary",
                min_value=0.0,
                value=4000.0
            )

    #Banking detail
    st.header("🏦 Banking Details")

    with st.expander("Bank Information", expanded=True):

        col1, col2 = st.columns(2)

        with col1:

            num_bank_accounts = st.number_input(
                "Number of Bank Accounts",
                min_value=0,
                value=3
            )

            num_credit_cards = st.number_input(
                "Number of Credit Cards",
                min_value=0,
                value=2
            )

        with col2:

            num_loans = st.number_input(
                "Number of Loans",
                min_value=0,
                value=1
            )

            interest_rate = st.number_input(
                "Interest Rate",
                min_value=0.0,
                value=8.0
            )

    #Debt information
    st.header("💰 Debt Information")

    with st.expander("Debt Details", expanded=True):

        col1, col2 = st.columns(2)

        with col1:

            outstanding_debt = st.number_input(
                "Outstanding Debt",
                min_value=0.0,
                value=500.0
            )

            total_emi = st.number_input(
                "Total EMI per Month",
                min_value=0.0,
                value=100.0
            )

        with col2:

            amount_invested = st.number_input(
                "Amount Invested Monthly",
                min_value=0.0,
                value=50.0
            )

            monthly_balance = st.number_input(
                "Monthly Balance",
                value=500.0
            )

    #Credit behaviour
    st.header("📊 Credit Behaviour")

    with st.expander("Credit Behaviour", expanded=True):

        col1, col2 = st.columns(2)

        with col1:

            delay_due = st.number_input(
                "Delay From Due Date",
                min_value=0,
                value=5
            )

            delayed_payments = st.number_input(
                "Number of Delayed Payments",
                min_value=0,
                value=2
            )

            credit_inquiries = st.number_input(
                "Credit Inquiries",
                min_value=0,
                value=3
            )

        with col2:

            credit_utilization = st.number_input(
                "Credit Utilization Ratio",
                min_value=0.0,
                value=30.0
            )

            changed_credit_limit = st.number_input(
                "Changed Credit Limit",
                value=5.0
            )

    #Customer profile
    st.header("📝 Customer Profile")

    with st.expander("Profile Details", expanded=True):

        col1, col2 = st.columns(2)

        with col1:

            occupation = st.selectbox(
                "Occupation",
                [
                    "Doctor",
                    "Engineer",
                    "Lawyer",
                    "Scientist",
                    "Teacher",
                    "Developer",
                    "Media_Manager",
                    "Mechanic",
                    "Entrepreneur",
                    "Writer",
                    "Architect",
                    "Journalist",
                    "Musician",
                    "Manager"
                ]
            )

            credit_mix = st.selectbox(
                "Credit Mix",
                [
                    "Good",
                    "Standard",
                    "Bad"
                ]
            )

        with col2:

            payment_minimum = st.selectbox(
                "Payment of Minimum Amount",
                [
                    "Yes",
                    "No"
                ]
            )

            payment_behaviour = st.selectbox(
                "Payment Behaviour",
                [
                    "High_spent_Small_value_payments",
                    "High_spent_Medium_value_payments",
                    "High_spent_Large_value_payments",
                    "Low_spent_Small_value_payments",
                    "Low_spent_Medium_value_payments",
                    "Low_spent_Large_value_payments"
                ]
            )

    #Predict button
    st.markdown("---")
    predict = st.button(
        "🔍 Predict Credit Score",
        use_container_width=True
    )

    # -----------------------------
    # Compute engineered features
    # -----------------------------
    if predict:
            # Step 1: Feature Engineering
            debt_to_income = outstanding_debt / annual_income
            emi_ratio = total_emi / monthly_salary
            investment_ratio = amount_invested / monthly_salary
            loan_per_account = num_loans / num_bank_accounts
            # Step 2: Create DataFrame
            input_data = {
            ...
            }
            input_df = pd.DataFrame([input_data])
            # Step 3: Encoding
            input_df = pd.get_dummies(input_df)
            #Align columns
            input_df = input_df.reindex(columns=feature_names, fill_value=0)
            # Step 4: Prediction
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            #Display the class label
            credit_labels = {
                0: "Poor",
                1: "Standard",
                2: "Good"
                }
            result = credit_labels[int(prediction)]
            #Progress bars
            labels = ["Poor", "Standard", "Good"]
            values = prob * 100
            colors = ["#e74c3c", "#f1c40f", "#2ecc71"]

            fig = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:.2f}%" for v in values],
                textposition="outside"
            ))

            fig.update_layout(
                title="Prediction Confidence",
                xaxis=dict(range=[0,100]),
                height=300,
                showlegend=False,
                margin=dict(l=30, r=30, t=50, b=30)
            )

            st.plotly_chart(fig, use_container_width=True) 
            #Prediction result
            st.success(f"Predicted Credit Score: {result}")
            highest = prob.max()*100

            if highest >= 80:
                st.success(f"✅ High confidence prediction ({highest:.1f}%)")

            elif highest >= 60:
                st.info(f"ℹ️ Moderate confidence prediction ({highest:.1f}%)")

            else:
                st.warning(f"⚠️ Low confidence prediction ({highest:.1f}%). The model finds this customer close to multiple credit categories.")
            # -----------------------------
            # Credit Improvement Advisor
            # -----------------------------
            st.markdown("---")
            st.header("💡 Credit Improvement Advisor")
            prediction_result = prediction      # Poor / Standard / Good

            # -------------------------------
            # Financial Health Score
            # -------------------------------

            score = 100

            if outstanding_debt > 5000:
                score -= 15

            if debt_to_income > 0.40:
                score -= 15

            if credit_utilization > 30:
                score -= 10

            if delayed_payments > 3:
                score -= 15

            if delay_due > 10:
                score -= 10

            if interest_rate > 15:
                score -= 10

            if credit_inquiries > 5:
                score -= 10

            if num_loans > 4:
                score -= 5

            if payment_minimum == "Yes":
                score -= 5

            if score < 0:
                score = 0

            # -------------------------------
            # Risk Banner
            # -------------------------------
            if prediction_result == "Poor":
                  st.error("""
            ### 🔴 High Credit Risk
                           
            Several financial factors are negatively affecting your credit score.
            """)
             

            elif prediction_result == "Standard":
                st.warning(f"""
            ### 🟡 Average Financial Profile

            **Predicted Credit Score : {prediction_result}**

            Your credit profile is stable, but there is room for improvement.
            """)

            else:
                   st.success(f"""
            ### 🟢 Excellent Financial Profile

            **Predicted Credit Score : {prediction_result}**

            Your financial behaviour is healthy. Continue maintaining your excellent habits.
            """)
            # -------------------------------
            # Financial Health Score
            # -------------------------------

            st.subheader("📊 Financial Health Score")

            st.progress(score / 100)

            st.metric("Overall Score", f"{score}/100")

            # -------------------------------
            # Recommendation Card Function
            # -------------------------------

            def recommendation(priority, color, icon, title, message):

                st.markdown(
                    f"""
            <div style="
            background-color:#f8f9fa;
            padding:18px;
            border-radius:12px;
            border-left:8px solid {color};
            margin-bottom:15px;
            box-shadow:2px 2px 8px rgba(0,0,0,0.15);
            ">

            <h4>{icon} {title}</h4>

            <b>Priority :</b> {priority}

            <br><br>

            {message}

            </div>
            """,
                    unsafe_allow_html=True,
                )

            # -------------------------------
            # Recommendations
            # -------------------------------

            st.subheader("🎯 Personalized Recommendations")

            recommendation_found = False

            # Outstanding Debt

            if outstanding_debt > 5000:

                recommendation_found = True

                recommendation(
                    "High",
                    "#ff4b4b",
                    "💰",
                    "Reduce Outstanding Debt",
                    "Your outstanding debt is high. Paying off high-interest debt first can significantly improve your credit score."
                )

            # Credit Utilization

            if credit_utilization > 30:

                recommendation_found = True

                recommendation(
                    "High",
                    "#ff4b4b",
                    "💳",
                    "Lower Credit Utilization",
                    "Keep your credit utilization below 30% by paying credit card bills before the due date."
                )

            # Delayed Payments

            if delayed_payments > 3:

                recommendation_found = True

                recommendation(
                    "High",
                    "#ff4b4b",
                    "⏰",
                    "Avoid Delayed Payments",
                    "Pay all EMIs and bills on time. Timely payments are one of the most important factors in credit scoring."
                )

            # Delay from Due Date

            if delay_due > 10:

                recommendation_found = True

                recommendation(
                    "Medium",
                    "#ff9800",
                    "📅",
                    "Reduce Payment Delay",
                    "Pay your dues before the due date to avoid penalties and improve your repayment history."
                )

            # Debt to Income

            if debt_to_income > 0.40:

                recommendation_found = True

                recommendation(
                    "High",
                    "#ff4b4b",
                    "📉",
                    "Improve Debt-to-Income Ratio",
                    "Reduce existing debt or increase your income to improve your financial stability."
                )

            # Interest Rate

            if interest_rate > 15:

                recommendation_found = True

                recommendation(
                    "Medium",
                    "#ff9800",
                    "📈",
                    "Reduce High Interest Debt",
                    "Consider refinancing high-interest loans whenever possible."
                )

            # Credit Inquiries

            if credit_inquiries > 5:

                recommendation_found = True

                recommendation(
                    "Medium",
                    "#ff9800",
                    "🔍",
                    "Avoid Multiple Credit Applications",
                    "Applying for many loans or credit cards within a short period may negatively affect your credit score."
                )

            # Number of Loans

            if num_loans > 4:

                recommendation_found = True

                recommendation(
                    "Medium",
                    "#ff9800",
                    "🏦",
                    "Reduce Active Loans",
                    "Managing fewer active loans generally leads to a healthier credit profile."
                )

            # Credit Mix

            if credit_mix == "Bad":

                recommendation_found = True

                recommendation(
                    "Medium",
                    "#ff9800",
                    "📂",
                    "Improve Credit Mix",
                    "Maintain a balanced mix of secured and unsecured credit for better credit health."
                )

            # Minimum Payment

            if payment_minimum == "Yes":

                recommendation_found = True

                recommendation(
                    "Low",
                    "#4CAF50",
                    "💵",
                    "Pay More Than Minimum Due",
                    "Whenever possible, pay the full outstanding amount instead of only the minimum due."
                )

            # Investment Ratio

            if investment_ratio < 0.10:

                recommendation_found = True

                recommendation(
                    "Low",
                    "#4CAF50",
                    "📊",
                    "Increase Investments",
                    "Increasing your monthly investments strengthens your long-term financial stability."
                )

            # Monthly Balance

            if monthly_balance < 0:

                recommendation_found = True

                recommendation(
                    "High",
                    "#ff4b4b",
                    "⚠",
                    "Improve Monthly Balance",
                    "Your monthly balance is negative. Consider reducing unnecessary expenses."
                )

            # -------------------------------
            # Positive Message
            # -------------------------------

            if not recommendation_found:

                st.success("""
            ### 🎉 Excellent Financial Behaviour

            No major financial issues were detected.

            Keep following these good financial habits:

            ✅ Pay EMIs on time

            ✅ Maintain low credit utilization

            ✅ Avoid unnecessary loans

            ✅ Continue investing regularly

            ✅ Maintain your excellent credit history
            """)
                pass

elif page == "📊 Analytics Dashboard":
        
    # -------------------------------
    # Analytical Dashboard
    # -------------------------------
    st.title("📊 Analytics Dashboard")
    import pandas as pd

    df = pd.read_csv("data/processed/cleaned.csv")
    #Display first five rows
    st.subheader("Dataset Preview")
    #Add Dataset Preview Checkbox
    if st.checkbox("Show Dataset"):

        st.dataframe(df.head(10))
    #dataset Overview
    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    card_style = """
    <style>
    .metric-card {
        background-color: #ffffff;
        border: 2px solid #E5E7EB;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 8px 20px rgba(0,0,0,0.18);
        border-color: #4F46E5;
    }

    .metric-title {
        font-size:16px;
        color:#6B7280;
        font-weight:600;
    }

    .metric-value {
        font-size:28px;
        font-weight:bold;
        color:#111827;
        margin-top:8px;
    }
    </style>
    """

    st.markdown(card_style, unsafe_allow_html=True)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">👥 Total Customers</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Features</div>
            <div class="metric-value">{df.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Average Income</div>
            <div class="metric-value">₹{df['Annual_Income'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💸 Average Debt</div>
            <div class="metric-value">₹{df['Outstanding_Debt'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Credit Score Distribution
    st.subheader("Credit Score Distribution")

    # Count each credit score category
    credit_count = (
        df["Credit_Score"]
        .value_counts()
        .reset_index()
    )

    credit_count.columns = ["Credit Score", "Count"]

    # If Credit_Score is encoded as 0,1,2 uncomment this:
    # credit_count["Credit Score"] = credit_count["Credit Score"].replace({
    #     0: "Poor",
    #     1: "Standard",
    #     2: "Good"
    # })

    fig1 = px.pie(
        credit_count,
        names="Credit Score",
        values="Count",
        hole=0.6,      # Makes it a donut chart
        color="Credit Score",
        color_discrete_map={
            "Poor": "#EF4444",        # Red
            "Standard": "#FBBF24",    # Yellow
            "Good": "#22C55E"         # Green
        }
    )

    fig1.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Percentage: %{percent}<extra></extra>"
    )

    fig1.update_layout(
        title={
            "text": "Credit Score Distribution",
            "x": 0.5
        },
        annotations=[
            dict(
                text=f"<b>{len(df):,}</b><br>Customers",
                x=0.5,
                y=0.5,
                showarrow=False,
                font_size=18
            )
        ],
        legend_title="",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig1, use_container_width=True)
    #Income distribution
    st.subheader("Annual Income Distribution")

    fig2 = px.histogram(
        df,
        x="Annual_Income",
        nbins=40,
        title="Annual Income Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)
    
    #Outstanding debt
    st.subheader("Outstanding Debt Distribution")

    fig = px.histogram(
        df,
        x="Outstanding_Debt",
        nbins=40,
        title="Outstanding Debt Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    #Credit Mix
    st.subheader("Credit Mix")

    credit_mix = df["Credit_Mix"].value_counts()

    fig = px.pie(
        values=credit_mix.values,
        names=credit_mix.index,
        title="Credit Mix Distribution",
        color=credit_mix.index,
        color_discrete_map={
            "Poor": "#EF4444",        # Red
            "Standard": "#FBBF24",    # Yellow
            "Good": "#22C55E"         # Green
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    #Occupation Distribution
    heat = pd.crosstab(
        df["Occupation"],
        df["Credit_Score"]
    )

    heat.rename(columns={
        0: "Poor",
        1: "Standard",
        2: "Good"
    }, inplace=True)

    fig = px.imshow(
        heat,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Occupation vs Credit Score"
    )

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)
    #Feature Importance
    feature_imp = pd.read_csv("data/processed/cleaned.csv")
    st.subheader("Top Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
     by="Importance",
     ascending=False
    )
    fig = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(fig, use_container_width=True)
        
elif page == "ℹ️ About Project":

    st.header("ℹ️ About the Project")

    st.markdown("""
    This application predicts the **credit score category** of a customer (**Good**, **Standard**, or **Poor**) using Machine Learning.

    The project demonstrates a complete end-to-end machine learning pipeline, including data preprocessing, feature engineering, model training, hyperparameter tuning, prediction, personalized recommendations, and interactive analytics.
    """)

    st.markdown("---")

    # Dataset
    st.subheader("📂 Dataset")

    st.info("""
    **Credit Score Classification Dataset**

    -  Total Records: **100,000**
    -  Target Classes: **Good, Standard, Poor**
    -  Original Features: **24**
    -  Engineered Features: **50+**
    """)

    st.markdown("---")

    # Model Performance
    st.subheader("🏆 Best Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "79.47%")

    with col2:
        st.metric("Precision", "79%")

    with col3:
        st.metric("Recall", "79%")

    with col4:
        st.metric("F1 Score", "79%")

    st.success("✅ Best Performing Model: **Tuned Random Forest Classifier**")

    st.markdown("---")

    # Features Used
    st.subheader("📋 Key Features Used")

    st.markdown("""
    -  Age
    -  Annual Income
    -  Outstanding Debt
    -  Credit Utilization Ratio
    -  Number of Credit Cards
    -  Interest Rate
    -  Credit Mix
    -  Credit History Age
    -  EMI per Month
    -  Debt-to-Income Ratio
    -  And several engineered features
    """)

    st.markdown("---")

    # Technologies
    st.subheader("🛠 Technologies Used")

    st.markdown("""
    -  Python
    -  Pandas & NumPy
    -  Scikit-learn
    -  Plotly
    -  Streamlit
    -  Joblib
    """)

    st.markdown("---")

    # Workflow
    st.subheader("🚀 Machine Learning Pipeline")

    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">

    <div style="background:#E3F2FD;padding:15px;border-radius:12px;font-weight:bold;">Data<br>Ingestion<br></div>

    <div style="font-size:30px;">➡️</div>

    <div style="background:#FFF3CD;padding:15px;border-radius:12px;font-weight:bold;"><br>EDA<br></div>

    <div style="font-size:30px;">➡️</div>

    <div style="background:#E8F5E9;padding:15px;border-radius:12px;font-weight:bold;"><br>Preprocessing<br></div>

    <div style="font-size:30px;">➡️</div>

    <div style="background:#EDE7F6;padding:15px;border-radius:12px;font-weight:bold;">Feature<br>Engineering<br></div>

    <div style="font-size:30px;">➡️</div>

    <div style="background:#E0F7FA;padding:15px;border-radius:12px;font-weight:bold;"><br>Tuned RF<br></div>

    <div style="font-size:30px;">➡️</div>

    <div style="background:#F3E5F5;padding:15px;border-radius:12px;font-weight:bold;"><br>Prediction<br></div>

    <div style="font-size:30px;">➡️</div>

    <div style="background:#FFF8E1;padding:15px;border-radius:12px;font-weight:bold;"><br>Advisor<br></div>

    <div style="font-size:30px;">➡️</div>

    <div style="background:#E8F5E9;padding:15px;border-radius:12px;font-weight:bold;"><br>Dashboard<br></div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.success(
        "This project demonstrates a complete Machine Learning workflow for "
        "credit score prediction with an interactive web application, "
        "personalized recommendations, and business analytics."
    )

# ------------------------------------
# Footer
# ------------------------------------

st.markdown("""
<hr style="margin-top:50px;margin-bottom:15px;">

<div style="text-align:center; color:#808080; font-size:15px;">

<b>Credit Score Prediction System</b><br><br>

👩‍💻 Developed by <b>Nikita Rani</b><br>

🎓 B.Tech Computer Science & Engineering<br>

🤖 Machine Learning Project | 2026

</div>
""", unsafe_allow_html=True)