import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AI/ML Salary Prediction",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "salary_prediction_pipeline.pkl"
DATA_PATH = BASE_DIR / "ai_job_market_cleaned.csv"
RESULTS_PATH = BASE_DIR / "model_results.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
results_df = pd.read_csv(RESULTS_PATH)

st.title("AI/ML Job Salary Prediction Dashboard")

st.write(
    "Enter the job details below to estimate the average salary for an AI/ML job listing."
)

st.divider()

# -----------------------------
# Helper functions
# -----------------------------

def get_unique_values(column_name):
    return sorted(df[column_name].dropna().astype(str).unique())


def split_values(series):
    values = (
        series.astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .dropna()
        .unique()
    )
    return sorted([value for value in values if value and value.lower() != "nan"])


def has_any(selected_items, keywords):
    selected_items = [item.lower() for item in selected_items]
    return int(any(keyword.lower() in selected_items for keyword in keywords))


def create_prediction_input(
    industry,
    job_title,
    experience_level,
    employment_type,
    country_code,
    company_size,
    posted_year,
    posted_month,
    selected_skills,
    selected_tools
):
    all_selected_items = selected_skills + selected_tools

    posted_quarter = ((posted_month - 1) // 3) + 1

    experience_mapping = {
        "Entry": 1,
        "Mid": 2,
        "Senior": 3
    }

    company_size_mapping = {
        "Startup": 1,
        "Mid": 2,
        "Large": 3
    }

    has_python = has_any(all_selected_items, ["python"])
    has_sql = has_any(all_selected_items, ["sql"])
    has_r = has_any(all_selected_items, ["r"])
    has_excel = has_any(all_selected_items, ["excel"])
    has_pandas = has_any(all_selected_items, ["pandas"])
    has_numpy = has_any(all_selected_items, ["numpy"])
    has_scikit_learn = has_any(all_selected_items, ["scikit-learn"])
    has_tensorflow = has_any(all_selected_items, ["tensorflow"])
    has_pytorch = has_any(all_selected_items, ["pytorch"])
    has_keras = has_any(all_selected_items, ["keras"])
    has_huggingface = has_any(all_selected_items, ["hugging face"])
    has_langchain = has_any(all_selected_items, ["langchain"])
    has_fastapi = has_any(all_selected_items, ["fastapi"])
    has_flask = has_any(all_selected_items, ["flask"])
    has_powerbi = has_any(all_selected_items, ["power bi"])
    has_mlflow = has_any(all_selected_items, ["mlflow"])
    has_cuda = has_any(all_selected_items, ["cuda"])
    has_aws = has_any(all_selected_items, ["aws"])
    has_azure = has_any(all_selected_items, ["azure"])
    has_gcp = has_any(all_selected_items, ["gcp"])
    has_bigquery = has_any(all_selected_items, ["bigquery"])
    has_kdb = has_any(all_selected_items, ["kdb+"])
    has_reinforcement_learning = has_any(all_selected_items, ["reinforcement learning"])

    has_cloud = max(has_aws, has_azure, has_gcp)
    has_deep_learning = max(has_tensorflow, has_pytorch, has_keras)
    has_nlp_related = max(has_huggingface, has_langchain)

    input_data = pd.DataFrame([{
        "industry": industry,
        "job_title": job_title,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "country_code": country_code,
        "company_size": company_size,
        "posted_year": posted_year,
        "posted_month": posted_month,
        "posted_quarter": posted_quarter,
        "skill_count": len(selected_skills),
        "tool_count": len(selected_tools),
        "experience_level_num": experience_mapping.get(experience_level, 0),
        "company_size_num": company_size_mapping.get(company_size, 0),
        "has_python": has_python,
        "has_sql": has_sql,
        "has_r": has_r,
        "has_excel": has_excel,
        "has_pandas": has_pandas,
        "has_numpy": has_numpy,
        "has_scikit_learn": has_scikit_learn,
        "has_tensorflow": has_tensorflow,
        "has_pytorch": has_pytorch,
        "has_keras": has_keras,
        "has_huggingface": has_huggingface,
        "has_langchain": has_langchain,
        "has_fastapi": has_fastapi,
        "has_flask": has_flask,
        "has_powerbi": has_powerbi,
        "has_mlflow": has_mlflow,
        "has_cuda": has_cuda,
        "has_aws": has_aws,
        "has_azure": has_azure,
        "has_gcp": has_gcp,
        "has_bigquery": has_bigquery,
        "has_kdb": has_kdb,
        "has_reinforcement_learning": has_reinforcement_learning,
        "has_cloud": has_cloud,
        "has_deep_learning": has_deep_learning,
        "has_nlp_related": has_nlp_related
    }])

    return input_data


# -----------------------------
# Input options
# -----------------------------

job_titles = get_unique_values("job_title")
industries = get_unique_values("industry")
experience_levels = get_unique_values("experience_level")
employment_types = get_unique_values("employment_type")
company_sizes = get_unique_values("company_size")
country_codes = get_unique_values("country_code")
years = sorted(df["posted_year"].dropna().unique())

skills_options = split_values(df["skills_required"])
tools_options = split_values(df["tools_preferred"])

# -----------------------------
# Dashboard layout
# -----------------------------

st.subheader("Job Information")

col1, col2, col3 = st.columns(3)

with col1:
    job_title = st.selectbox("Job Role", job_titles)
    experience_level = st.selectbox("Experience Level", experience_levels)

with col2:
    industry = st.selectbox("Industry", industries)
    employment_type = st.selectbox("Employment Type", employment_types)

with col3:
    company_size = st.selectbox("Company Size", company_sizes)
    country_code = st.selectbox("Location Code", country_codes)

st.subheader("Skills and Tools")

col4, col5 = st.columns(2)

with col4:
    selected_skills = st.multiselect(
        "Required Skills",
        skills_options,
        default=["Python", "SQL"] if "Python" in skills_options and "SQL" in skills_options else []
    )

with col5:
    selected_tools = st.multiselect(
        "Preferred Tools",
        tools_options,
        default=["TensorFlow"] if "TensorFlow" in tools_options else []
    )

with st.expander("Optional posting date settings"):
    col6, col7 = st.columns(2)

    with col6:
        posted_year = st.selectbox("Posted Year", years, index=len(years) - 1)

    with col7:
        posted_month = st.selectbox("Posted Month", list(range(1, 13)), index=0)

st.divider()

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Salary", use_container_width=True):
    input_data = create_prediction_input(
        industry=industry,
        job_title=job_title,
        experience_level=experience_level,
        employment_type=employment_type,
        country_code=country_code,
        company_size=company_size,
        posted_year=posted_year,
        posted_month=posted_month,
        selected_skills=selected_skills,
        selected_tools=selected_tools
    )

    predicted_salary = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    col8, col9, col10 = st.columns(3)

    col8.metric("Predicted Average Salary", f"${predicted_salary:,.2f}")
    col9.metric("Selected Job Role", job_title)
    col10.metric("Experience Level", experience_level)

    st.success("Salary prediction generated successfully.")

# -----------------------------
# Simple results section
# -----------------------------

st.divider()

st.subheader("Model Performance Summary")

best_model = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]

col11, col12, col13, col14 = st.columns(4)

col11.metric("Best Model", best_model["Model"])
col12.metric("MAE", f"{best_model['MAE']:,.2f}")
col13.metric("RMSE", f"{best_model['RMSE']:,.2f}")
col14.metric("R² Score", f"{best_model['R2 Score']:.4f}")

with st.expander("View full model comparison table"):
    st.dataframe(results_df)