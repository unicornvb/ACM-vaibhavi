"""
Streamlit web application for competitive programming difficulty prediction.
Allows users to input problem descriptions and get real-time difficulty predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import add_meta_features

# Page configuration
st.set_page_config(
    page_title="CP Difficulty Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E88E5 0%, #7B1FA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1E88E5;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
        padding-left: 12px;
    }
    
    /* Result boxes */
    .result-container {
        margin-top: 3rem;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .easy-result {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border: 3px solid #4CAF50;
    }
    
    .medium-result {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border: 3px solid #FF9800;
    }
    
    .hard-result {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border: 3px solid #F44336;
    }
    
    .result-class {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .result-score {
        font-size: 2rem;
        font-weight: 500;
        color: #555;
        margin-bottom: 0.5rem;
    }
    
    .result-subtitle {
        font-size: 1.1rem;
        color: #777;
        font-style: italic;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(120deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 16px rgba(30, 136, 229, 0.3);
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(120deg, #1565C0 0%, #0D47A1 100%);
        box-shadow: 0 6px 24px rgba(30, 136, 229, 0.4);
        transform: translateY(-2px);
    }
    
    /* Text areas */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
        font-size: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #1E88E5;
        box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.1);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1E88E5;
        margin: 1.5rem 0;
    }
    
    .info-box-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0.8rem;
    }
    
    .info-box-content {
        color: #424242;
        line-height: 1.8;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    /* Example box */
    .example-box {
        background: #F5F5F5;
        border: 2px dashed #BDBDBD;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.95rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        padding: 2rem;
        margin-top: 4rem;
        border-top: 1px solid #E0E0E0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_models():
    """Load trained classification and regression models."""
    artifact_dir = Path("artifacts")
    
    try:
        clf_path = artifact_dir / "classification_model.joblib"
        reg_path = artifact_dir / "regression_model.joblib"
        
        if not clf_path.exists() or not reg_path.exists():
            return None, None, "Model files not found. Please train models first."
        
        clf_artifacts = joblib.load(clf_path)
        reg_artifacts = joblib.load(reg_path)
        
        return clf_artifacts, reg_artifacts, None
        
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"


def prepare_problem_input(description: str, input_desc: str, output_desc: str) -> pd.DataFrame:
    """
    Prepare problem text for prediction.
    
    Args:
        description: Main problem statement
        input_desc: Input format description
        output_desc: Output format description
        
    Returns:
        DataFrame with combined text and extracted features
    """
    df = pd.DataFrame({
        "description": [description],
        "input_description": [input_desc],
        "output_description": [output_desc]
    })
    
    # Combine text columns
    df["combined_text"] = (
        df["description"].fillna("") + "\n" +
        df["input_description"].fillna("") + "\n" +
        df["output_description"].fillna("")
    )
    
    # Extract meta features
    df = add_meta_features(df)
    
    return df


def make_prediction(df: pd.DataFrame, clf_artifacts: dict, reg_artifacts: dict) -> tuple:
    """
    Make difficulty predictions using loaded models.
    
    Args:
        df: Input DataFrame with features
        clf_artifacts: Classification model artifacts
        reg_artifacts: Regression model artifacts
        
    Returns:
        Tuple of (predicted_class, predicted_score)
    """
    meta_cols = clf_artifacts["meta_cols"]
    
    # Ensure all meta columns exist
    for col in meta_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Transform text features
    X_text_clf = clf_artifacts["text_pipeline"].transform(df["combined_text"])
    X_text_reg = reg_artifacts["text_pipeline"].transform(df["combined_text"])
    
    # Get meta features
    X_meta = df[meta_cols].values
    
    # Combine features for classification
    X_combined_clf = np.hstack([X_text_clf, X_meta])
    X_combined_clf = clf_artifacts["variance_filter"].transform(X_combined_clf)
    X_combined_clf = clf_artifacts["scaler"].transform(X_combined_clf)
    
    # Combine features for regression
    X_combined_reg = np.hstack([X_text_reg, X_meta])
    X_combined_reg = reg_artifacts["variance_filter"].transform(X_combined_reg)
    X_combined_reg = reg_artifacts["scaler"].transform(X_combined_reg)
    
    # Make predictions
    pred_class = clf_artifacts["model"].predict(X_combined_clf)[0]
    pred_score = reg_artifacts["model"].predict(X_combined_reg)[0]
    
    # Clip score to valid range
    pred_score = np.clip(pred_score, 0, 10)
    
    return pred_class, pred_score


def display_prediction_result(pred_class: str, pred_score: float):
    """
    Display prediction results with beautiful formatting.
    
    Args:
        pred_class: Predicted difficulty class
        pred_score: Predicted difficulty score
    """
    # Determine styling
    if pred_class == "Easy":
        box_class = "easy-result"
        color = "#4CAF50"
        emoji = "🟢"
        description = "Great for beginners and practice!"
    elif pred_class == "Medium":
        box_class = "medium-result"
        color = "#FF9800"
        emoji = "🟡"
        description = "Requires solid algorithmic knowledge"
    else:  # Hard
        box_class = "hard-result"
        color = "#F44336"
        emoji = "🔴"
        description = "Advanced problem solving required"
    
    # Main result display
    st.markdown(f"""
        <div class="result-container {box_class}">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{emoji}</div>
            <div class="result-class" style="color: {color};">{pred_class}</div>
            <div class="result-score">Difficulty Score: <strong>{pred_score:.2f}</strong> / 10</div>
            <div class="result-subtitle">{description}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Detailed metrics
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Difficulty Class",
            value=pred_class,
            delta=None
        )
    
    with col2:
        st.metric(
            label="🎯 Score",
            value=f"{pred_score:.2f}",
            delta=None
        )
    
    with col3:
        percentile = (pred_score / 10) * 100
        st.metric(
            label="📈 Percentile",
            value=f"{percentile:.0f}%",
            delta=None
        )
    
    with col4:
        if pred_score <= 3:
            complexity = "O(n) or better"
        elif pred_score <= 6:
            complexity = "O(n log n)"
        elif pred_score <= 8:
            complexity = "O(n²) or DP"
        else:
            complexity = "Advanced algorithms"
        
        st.metric(
            label="💡 Likely Complexity",
            value=complexity,
            delta=None
        )


def display_detected_features(df: pd.DataFrame):
    """Display detected features from the problem."""
    st.markdown("### 🔍 Feature Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📝 Text Features**")
        st.write(f"• Text Length: {int(df['text_len'].values[0])} chars")
        st.write(f"• Lines: {int(df['num_lines_est'].values[0])}")
        st.write(f"• Tokens: {int(df['token_count'].values[0])}")
        st.write(f"• Avg Word Length: {df['avg_word_len'].values[0]:.2f}")
    
    with col2:
        st.markdown("**🔢 Problem Structure**")
        st.write(f"• Keywords Found: {int(df['num_keywords'].values[0])}")
        st.write(f"• Math Symbols: {int(df['num_math_symbols'].values[0])}")
        st.write(f"• Constraints: {int(df['num_constraints'].values[0])}")
        st.write(f"• Examples: {int(df['num_examples'].values[0])}")
    
    with col3:
        st.markdown("**🎓 Algorithm Hints**")
        st.write(f"• Graph Keywords: {int(df['graph_kw_count'].values[0])}")
        st.write(f"• DP Keywords: {int(df['dp_kw_count'].values[0])}")
        st.write(f"• Data Structure Keywords: {int(df['advanced_ds_kw_count'].values[0])}")
        st.write(f"• Max N: {int(df['estimated_max_n'].values[0])}")


def display_example_problems():
    """Display example problems for user reference."""
    st.markdown("""
        <div class="info-box">
            <div class="info-box-title">📚 Example Problems</div>
            <div class="info-box-content">
                Click on an example to auto-fill:
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    examples = {
        "🟢 Easy: Two Sum": {
            "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution.",
            "input": "First line contains n (1 ≤ n ≤ 1000). Second line contains n space-separated integers. Third line contains target integer.",
            "output": "Print two space-separated indices (0-indexed)."
        },
        "🟡 Medium: Longest Increasing Subsequence": {
            "description": "Given an integer array nums, return the length of the longest strictly increasing subsequence. A subsequence is derived from the array by deleting some or no elements without changing the order of the remaining elements.",
            "input": "First line contains n (1 ≤ n ≤ 10^5). Second line contains n space-separated integers (-10^9 ≤ nums[i] ≤ 10^9).",
            "output": "Print a single integer - the length of the longest increasing subsequence."
        },
        "🔴 Hard: Minimum Cost to Merge Stones": {
            "description": "There are n piles of stones arranged in a row. The i-th pile has stones[i] stones. A move consists of merging exactly k consecutive piles into one pile, and the cost of this move is equal to the total number of stones in these k piles. Return the minimum cost to merge all piles of stones into one pile. If it is impossible, return -1.",
            "input": "First line contains n and k (1 ≤ n ≤ 30, 2 ≤ k ≤ n). Second line contains n space-separated integers (1 ≤ stones[i] ≤ 100).",
            "output": "Print the minimum cost, or -1 if impossible."
        }
    }
    
    cols = st.columns(len(examples))
    
    for idx, (title, content) in enumerate(examples.items()):
        with cols[idx]:
            if st.button(title, key=f"example_{idx}", use_container_width=True):
                st.session_state.example_desc = content["description"]
                st.session_state.example_input = content["input"]
                st.session_state.example_output = content["output"]
                st.rerun()


def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<div class="main-header">🎯 CP Difficulty Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Predict competitive programming problem difficulty using machine learning</div>',
        unsafe_allow_html=True
    )
    
    # Load models
    with st.spinner("Loading models..."):
        clf_artifacts, reg_artifacts, error = load_models()
    
    if error:
        st.error(f"❌ {error}")
        st.info("""
            **To train the models:**
            1. Ensure `problems_data_acm.jsonl` is in the project directory
            2. Run: `python src/train_models.py`
            3. Wait for training to complete
            4. Refresh this page
        """)
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Example problems
    with st.expander("💡 See Example Problems", expanded=False):
        display_example_problems()
    
    # Main input section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="section-header">📝 Problem Statement</div>', unsafe_allow_html=True)
        
        # Check if example was clicked
        default_desc = st.session_state.get("example_desc", "")
        default_input = st.session_state.get("example_input", "")
        default_output = st.session_state.get("example_output", "")
        
        description = st.text_area(
            "Main Description",
            value=default_desc,
            height=200,
            placeholder="Enter the problem statement here...\n\nExample: Given an array of integers, find the maximum sum of any contiguous subarray.",
            help="Describe what the problem asks to solve"
        )
        
        input_desc = st.text_area(
            "Input Format",
            value=default_input,
            height=120,
            placeholder="Describe the input format...\n\nExample: First line contains n (1 ≤ n ≤ 10^5)\nSecond line contains n space-separated integers",
            help="Explain the structure and constraints of the input"
        )
        
        output_desc = st.text_area(
            "Output Format",
            value=default_output,
            height=120,
            placeholder="Describe the expected output...\n\nExample: Print a single integer - the maximum sum",
            help="Explain what the output should contain"
        )
    
    with col2:
        st.markdown('<div class="section-header">💡 Tips for Best Results</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <div class="info-box-content">
                    <strong>Include these for accurate predictions:</strong><br><br>
                    ✓ Complete problem statement<br>
                    ✓ Input/output constraints (n ≤ 10^5)<br>
                    ✓ Time and memory limits<br>
                    ✓ Sample test cases<br>
                    ✓ Algorithm hints (if any)<br>
                    ✓ Expected complexity
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📊 Difficulty Scale</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div style="background: #FAFAFA; padding: 1.5rem; border-radius: 12px;">
                <div style="margin-bottom: 1rem;">
                    <span style="color: #4CAF50; font-weight: bold; font-size: 1.2rem;">🟢 Easy</span>
                    <span style="color: #666; float: right;">Score ≤ 5.0</span>
                </div>
                <div style="margin-bottom: 1rem;">
                    <span style="color: #FF9800; font-weight: bold; font-size: 1.2rem;">🟡 Medium</span>
                    <span style="color: #666; float: right;">Score 5.1 - 8.5</span>
                </div>
                <div>
                    <span style="color: #F44336; font-weight: bold; font-size: 1.2rem;">🔴 Hard</span>
                    <span style="color: #666; float: right;">Score > 8.5</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Clear example button
    if st.session_state.get("example_desc"):
        if st.button("🗑️ Clear Example", use_container_width=False):
            st.session_state.example_desc = ""
            st.session_state.example_input = ""
            st.session_state.example_output = ""
            st.rerun()
    
    # Predict button
    st.markdown("---")
    predict_clicked = st.button("🚀 Predict Difficulty", type="primary")
    
    if predict_clicked:
        # Validate input
        if not description.strip():
            st.warning("⚠️ Please enter at least a problem description.")
            return
        
        # Make prediction
        with st.spinner("🔮 Analyzing problem difficulty..."):
            try:
                # Prepare input
                df = prepare_problem_input(description, input_desc, output_desc)
                
                # Make prediction
                pred_class, pred_score = make_prediction(df, clf_artifacts, reg_artifacts)
                
                # Display results
                st.markdown("## 🎯 Prediction Results")
                display_prediction_result(pred_class, pred_score)
                
                # Show detected features
                with st.expander("🔍 View Detected Features", expanded=False):
                    display_detected_features(df)
                
                # Clear example after prediction
                if "example_desc" in st.session_state:
                    del st.session_state.example_desc
                    del st.session_state.example_input
                    del st.session_state.example_output
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("""
        <div class="footer">
            Built with ❤️ using Streamlit and scikit-learn<br>
            <small>Competitive Programming Difficulty Predictor v1.0</small>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()