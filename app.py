import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from source import *

# Page Config
st.set_page_config(page_title="QuLab: Lab 12: Automation Project Evaluation", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 12: Automation Project Evaluation")
st.divider()

# Helper function to update derived columns
def _update_derived_columns():
    df = st.session_state.project_scores_df.copy()

    # Apply suitability calculations
    df['suit_efficiency'] = df.apply(calculate_weighted_score, axis=1, weights=st.session_state.weights_efficiency_first)
    df['suit_risk'] = df.apply(calculate_weighted_score, axis=1, weights=st.session_state.weights_risk_first)

    # Apply tier classification
    df['tier'] = df.apply(classify_tier, axis=1)
    df['color'] = df['tier'].map(st.session_state.tier_colors_map)

    # Map hours_saved_month and effort, then categorize timeline
    full_hours_saved_map = st.session_state.hours_saved_month_map_full.copy()
    full_effort_map = st.session_state.effort_map_full.copy()
    
    for task_id in df.index:
        if task_id not in full_hours_saved_map:
            full_hours_saved_map[task_id] = 7 # Default as per source.py logic
        if task_id not in full_effort_map:
            full_effort_map[task_id] = 'Med' # Default as per source.py logic
    
    df['hours_saved_month'] = df.index.map(full_hours_saved_map)
    df['effort'] = df.index.map(full_effort_map)
    df['timeline'] = df.apply(categorize_timeline, axis=1)

    st.session_state.project_scores_df = df

# Initialize Session State
def _initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.current_page = "Introduction"
        st.session_state.scorecard_dimensions = SCORECARD_DIMENSIONS
        st.session_state.score_columns = SCORE_COLUMNS
        st.session_state.tier_colors_map = tier_colors_map
        st.session_state.hybrid_routing_df = pd.DataFrame(hybrid_routing_data)
        
        st.session_state.hours_saved_month_map_full = hours_saved_month_map.copy()
        st.session_state.effort_map_full = effort_map.copy()
        
        # Build initial DataFrame
        # data is imported from source.py
        df_base = pd.DataFrame(data).T
        
        # Add custom tasks from source.py
        c1 = pd.DataFrame([custom_task_1_scores], index=['T9: Custom Task 1'])
        c2 = pd.DataFrame([custom_task_2_scores], index=['T10: Custom Task 2'])
        
        st.session_state.project_scores_df = pd.concat([df_base, c1, c2])
        st.session_state.project_scores_df.index.name = 'Project_ID'
        
        st.session_state.weights_efficiency_first = weights_efficiency_first.copy()
        st.session_state.weights_risk_first = weights_risk_first.copy()
        
        # Initialize selection to first project if available
        if not st.session_state.project_scores_df.empty:
            st.session_state.selected_top_project_id = st.session_state.project_scores_df.index[0]
        else:
            st.session_state.selected_top_project_id = None
            
        # ROI defaults
        st.session_state.roi_annual_hours_saved = 96
        st.session_state.roi_cost_per_hour = 150
        st.session_state.roi_development_cost = 1200
        st.session_state.roi_annual_operating_cost = 100
        
        st.session_state.initialized = True
        _update_derived_columns()

_initialize_session_state()

# Sidebar Navigation
st.sidebar.title("Automation Scorecard Navigator")
page_selection = st.sidebar.selectbox(
    "Choose a section:",
    [
        "Introduction",
        "1. Define Dimensions",
        "2. Score Projects",
        "3. Prioritize with Weights",
        "4. Classify Tiers",
        "5. Visualize Landscape",
        "6. Deep Dive & ROI",
        "7. Automation Roadmap"
    ],
    key="current_page"
)

# Page: Introduction
if st.session_state.current_page == "Introduction":
    st.title("Automation Project Evaluation: A CFA Institute Scorecard Approach for Investment Professionals")
    st.markdown(f"## Case Study: Systematically Prioritizing AI/ML Projects at Prosperity Global Asset Management")
    st.markdown(f"**Persona:** Alex Chen, CFA, Investment Operations Manager at Prosperity Global Asset Management.")
    st.markdown(f"**Organization:** Prosperity Global Asset Management (PGAM), a mid-sized asset management firm.")
    st.markdown(f"**Scenario:**")
    st.markdown(f"Alex Chen, CFA, leads a team at Prosperity Global Asset Management responsible for optimizing investment workflows. Recently, PGAM has been exploring the integration of AI/ML to enhance operational efficiency and innovation. However, ad-hoc automation efforts have led to mixed results, with some projects failing to deliver expected value due to misalignment with strategic goals or underestimation of risks.")
    st.markdown(f"To address this, Alex is tasked with implementing a structured framework for evaluating and prioritizing potential AI/ML automation initiatives. He aims to move beyond intuitive decisions and apply a systematic approach inspired by the CFA Institute's Automation Scorecard. This will enable him to identify high-value, feasible, and low-risk automation opportunities, ensuring strategic alignment and maximizing return on investment. The goal is to build a clear, prioritized automation roadmap that resonates with various stakeholders across portfolio management, compliance, and IT.")
    st.markdown(f"This application will guide Alex through a step-by-step workflow, from defining key evaluation dimensions to classifying projects and estimating their potential ROI, all within a practical, hands-on environment.")
    st.markdown(f"---")
    st.markdown(f"### Workflow Overview")
    st.markdown(f"1. **Define Dimensions**: Understand the scorecard criteria.")
    st.markdown(f"2. **Score Projects**: Input scores for candidate tasks.")
    st.markdown(f"3. **Prioritize with Weights**: Apply strategic weighting profiles.")
    st.markdown(f"4. **Classify Tiers**: Assign projects to automation categories.")
    st.markdown(f"5. **Visualize Landscape**: Gain insights through interactive plots.")
    st.markdown(f"6. **Deep Dive & ROI**: Explore hybrid solutions and financial justification.")
    st.markdown(f"7. **Automation Roadmap**: Generate a prioritized implementation plan.")

# Page: 1. Define Dimensions
elif st.session_state.current_page == "1. Define Dimensions":
    st.title("1. Defining the Automation Scorecard Dimensions")
    st.markdown(f"To systematically evaluate automation projects, Alex needs to clearly define the key dimensions that will be used. These dimensions are crucial for consistent scoring and strategic alignment, based on the CFA Institute's Automation Scorecard framework.")
    st.markdown(f"Alex understands that a robust framework requires clear definitions for each evaluation criterion. This helps ensure consistency in scoring across different projects and reduces subjectivity in the evaluation process.")
    st.markdown(f"The weighted suitability score will be calculated using the Multi-Criteria Decision Analysis (MCDA) formula:")
    st.markdown(r"$$S_j = \sum_{d=1}^{D} W_d \cdot X_{jd}$$")
    st.markdown(r"where $S_j$ is the suitability score for project $j$, $W_d$ is the weight assigned to dimension $d$, and $X_{jd}$ is the raw score of project $j$ on dimension $d$. A positive weight means a higher score on this dimension contributes positively to suitability; a negative weight means a higher score contributes negatively. $D$ is the total number of dimensions.")

    st.markdown(f"---")
    st.markdown(f"### Automation Scorecard Dimensions")
    st.markdown(f"Alex needs to programmatically define the 6 key dimensions for evaluating automation suitability. These definitions guide how each project will be scored, ensuring a standardized approach. He also considers the scoring scale for each dimension.")
    st.table(pd.DataFrame(st.session_state.scorecard_dimensions.items(), columns=['Dimension', 'Description']))

    st.markdown(f"**Explanation of Execution:**")
    st.markdown(f"By explicitly defining each dimension and its scoring scale, Alex establishes a common language for discussing automation suitability within PGAM. This clarity is vital for gathering consistent inputs from different department heads and ensuring that all stakeholders evaluate projects using the same criteria. This structured definition forms the foundation for objective analysis.")

# Page: 2. Score Projects
elif st.session_state.current_page == "2. Score Projects":
    st.title("2. Scoring Candidate Automation Projects")
    st.markdown(f"Alex identifies several potential automation projects within PGAM's investment workflows. He then scores each project against the defined dimensions, using a synthetic dataset that represents typical tasks. This hands-on scoring exercise is crucial for applying the framework to real-world scenarios.")
    st.markdown(f"Alex has compiled a list of tasks that his team performs regularly. To apply the scorecard, he needs to assign numerical scores to each task across the 6 dimensions. This initial scoring forms the raw data for the Multi-Criteria Decision Analysis (MCDA).")
    st.markdown(f"He will start with a pre-defined set of 8 tasks and their scores, representing common investment operations. The app also includes 2 pre-simulated custom tasks (T9, T10). You can add more custom tasks below!")
    st.markdown(f"**Important Note:** As highlighted by the CFA Institute, scores are subjective inputs. Different analysts or departments might score the same task differently based on their specific context or risk tolerance. The value of this scorecard lies in structuring the conversation and surfacing these disagreements early, *before* committing significant resources.")

    st.markdown(f"---")
    st.subheader("Current Project Scores")
    # Display and allow editing of scores
    edited_df = st.data_editor(
        st.session_state.project_scores_df[st.session_state.score_columns],
        column_config={
            dim: st.column_config.NumberColumn(
                label=dim.replace('_', ' ').title(),
                min_value=0 if dim == 'human_oversight' else 1,
                max_value=1 if dim == 'human_oversight' else 5,
                step=1,
                format="%d",
                help=st.session_state.scorecard_dimensions[dim]
            ) for dim in st.session_state.score_columns
        },
        key="project_scores_editor",
        num_rows="dynamic",
        use_container_width=True
    )
    
    if not edited_df.equals(st.session_state.project_scores_df[st.session_state.score_columns]):
        updated_df_scores_only = edited_df.copy()
        
        # Merge updated scores back into the session state DataFrame
        if len(updated_df_scores_only) > len(st.session_state.project_scores_df):
            # Identify new rows
            new_rows = updated_df_scores_only.loc[~updated_df_scores_only.index.isin(st.session_state.project_scores_df.index)]
            # Give new rows a temporary ID
            new_row_ids = [f"UserTask_{len(st.session_state.project_scores_df) + i + 1}" for i in range(len(new_rows))]
            new_rows.index = new_row_ids
            # Concatenate
            st.session_state.project_scores_df = pd.concat([st.session_state.project_scores_df, new_rows])
            st.session_state.project_scores_df = st.session_state.project_scores_df.combine_first(updated_df_scores_only)
        else:
             st.session_state.project_scores_df.update(updated_df_scores_only)

        _update_derived_columns()
        st.rerun()

    st.markdown(f"---")
    st.subheader("Add a New Custom Automation Task")
    with st.form("new_task_form"):
        new_task_id = st.text_input("Project ID (e.g., 'T11: My New Task')", max_chars=50)
        new_task_description = st.text_area("Brief Description", max_chars=200)
        new_task_scores = {}
        st.markdown("Enter scores (1-5, or 0/1 for human_oversight):")
        for dim, desc in st.session_state.scorecard_dimensions.items():
            if dim == 'human_oversight':
                new_task_scores[dim] = st.radio(f"**{dim.replace('_', ' ').title()}** ({desc})", options=[0, 1], key=f"new_score_{dim}")
            else:
                new_task_scores[dim] = st.slider(f"**{dim.replace('_', ' ').title()}** ({desc})", min_value=1, max_value=5, value=3, key=f"new_score_{dim}")
        
        submitted = st.form_submit_button("Add Project")
        if submitted:
            if new_task_id and new_task_id not in st.session_state.project_scores_df.index:
                new_task_df = pd.DataFrame([new_task_scores], index=[new_task_id], columns=st.session_state.score_columns)
                st.session_state.project_scores_df = pd.concat([st.session_state.project_scores_df, new_task_df])
                st.session_state.hours_saved_month_map_full[new_task_id] = 7 # Default
                st.session_state.effort_map_full[new_task_id] = 'Med' # Default
                _update_derived_columns()
                st.success(f"Project '{new_task_id}' added!")
                st.rerun()
            elif new_task_id in st.session_state.project_scores_df.index:
                st.error(f"Error: Project ID '{new_task_id}' already exists. Please choose a different one.")
            else:
                st.error("Please enter a Project ID.")

    st.markdown(f"---")
    st.subheader("Scorecard Heatmap (V1)")
    st.markdown(f"This heatmap visually represents the scores of all projects across the defined dimensions.")
    
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, max(8, len(st.session_state.project_scores_df) * 0.7)))
    sns.heatmap(st.session_state.project_scores_df[st.session_state.score_columns], annot=True, fmt='.0f', cmap='viridis', linewidths=.5, linecolor='lightgray', ax=ax_heatmap)
    ax_heatmap.set_title('Automation Scorecard Heatmap: Task Scores Across Dimensions', fontsize=16)
    ax_heatmap.set_xlabel('Score Dimensions', fontsize=12)
    ax_heatmap.set_ylabel('Candidate Projects', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig_heatmap)
    plt.close(fig_heatmap)

    st.markdown(f"**Explanation of Execution:**")
    st.markdown(f"Alex has now populated his scorecard with initial ratings for various tasks. The inclusion of custom tasks demonstrates the framework's flexibility to incorporate new ideas directly from his team's experience. This scored matrix serves as the foundation for quantitative analysis, enabling him to move towards calculating overall suitability. The raw scores immediately highlight tasks that might be complex, high-risk, or have significant efficiency potential, providing initial insights even before weighted calculations.")

# Page: 3. Prioritize with Weights
elif st.session_state.current_page == "3. Prioritize with Weights":
    st.title("3. Prioritizing with Weighted Suitability Scores (MCDA) and Sensitivity Analysis")
    st.markdown(f"Understanding that different strategic priorities exist within PGAM, Alex needs to evaluate projects based on varying criteria. He will use Multi-Criteria Decision Analysis (MCDA) with different weight profiles to compute suitability scores and perform a sensitivity analysis. This allows him to see how project rankings change based on PGAM's strategic focus (e.g., 'Efficiency-First' versus 'Risk-First').")
    st.markdown(f"Alex knows that PGAM's priorities can shift. Sometimes, the firm needs to prioritize quick wins and efficiency gains. At other times, minimizing risk is paramount. To account for these strategic nuances, he defines two distinct weight profiles for the scorecard dimensions: 'Efficiency-First' and 'Risk-First'.")
    st.markdown(f"The formula for calculating the weighted suitability score, $S_j = \sum_{{d=1}}^{{D}} W_d \cdot X_{{jd}}$, allows for flexibility.")
    st.markdown(r"where $S_j$ is the suitability score for project $j$, $W_d$ is the weight assigned to dimension $d$, and $X_{jd}$ is the raw score of project $j$ on dimension $d$. A higher positive weight on 'efficiency_impact', for instance, means tasks with high efficiency potential will rank higher under an 'Efficiency-First' profile. Conversely, a higher negative weight on 'risk_level' means high-risk tasks will be penalized more heavily under a 'Risk-First' profile.")
    st.markdown(f"This sensitivity analysis is a powerful tool to understand the robustness of recommendations and to facilitate discussions among stakeholders about trade-offs.")

    st.markdown(f"---")
    st.subheader("Configure Weight Profiles")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Efficiency-First Weights")
        current_weights_eff = st.session_state.weights_efficiency_first.copy()
        for dim, desc in st.session_state.scorecard_dimensions.items():
            current_weights_eff[dim] = st.slider(f"**{dim.replace('_', ' ').title()}** (Efficiency-First)",
                                                 min_value=-1.0, max_value=1.0, value=current_weights_eff.get(dim, 0.0), step=0.05,
                                                 key=f"eff_weight_{dim}")
        if current_weights_eff != st.session_state.weights_efficiency_first:
            st.session_state.weights_efficiency_first = current_weights_eff
            _update_derived_columns()
            st.rerun()

    with col2:
        st.markdown("#### Risk-First Weights")
        current_weights_risk = st.session_state.weights_risk_first.copy()
        for dim, desc in st.session_state.scorecard_dimensions.items():
            current_weights_risk[dim] = st.slider(f"**{dim.replace('_', ' ').title()}** (Risk-First)",
                                                  min_value=-1.0, max_value=1.0, value=current_weights_risk.get(dim, 0.0), step=0.05,
                                                  key=f"risk_weight_{dim}")
        if current_weights_risk != st.session_state.weights_risk_first:
            st.session_state.weights_risk_first = current_weights_risk
            _update_derived_columns()
            st.rerun()
    
    st.markdown(f"---")
    st.subheader("Weighted Suitability Scores")
    st.markdown(f"Below are the calculated suitability scores based on the configured weight profiles.")

    st.dataframe(st.session_state.project_scores_df[['suit_efficiency', 'suit_risk']].sort_values(
        'suit_efficiency', ascending=False
    ).round(2), use_container_width=True)

    st.markdown(f"**Explanation of Execution:**")
    st.markdown(f"By applying two different weight profiles, Alex can observe how project priorities shift based on strategic focus. For example, a task that ranks high under 'Efficiency-First' might drop significantly under 'Risk-First' if it involves high inherent risk. This sensitivity analysis is critical for Alex to present a nuanced view to PGAM's leadership, facilitating discussions on strategic trade-offs and ensuring that automation efforts align with current firm objectives. The numerical scores now provide a quantitative basis for comparing and ranking projects.")

# Page: 4. Classify Tiers
elif st.session_state.current_page == "4. Classify Tiers":
    st.title("4. Classifying Projects into Automation Tiers")
    st.markdown(f"Not all automation is created equal. Alex needs to categorize each project into the most appropriate automation tier to guide technology choices and implementation strategies. This involves applying rule-based logic to the project scores.")
    st.markdown(f"After calculating suitability scores, Alex's next step is to classify each project into one of four automation tiers:")
    st.markdown(f"1.  **Traditional Automation:** Rules-based, structured, objective tasks.")
    st.markdown(f"2.  **GenAI / LLM Automation:** Tasks involving unstructured data, context, and generation.")
    st.markdown(f"3.  **Human Intervention Required:** High-risk or highly subjective tasks requiring human oversight.")
    st.markdown(f"4.  **Hybrid (Traditional + GenAI):** Tasks that combine elements of both structured processing and generative capabilities.")
    st.markdown(f"This classification is crucial for determining the right technological approach and resource allocation. The classification logic is based on a decision tree derived from the CFA Institute framework:")
    st.markdown(f"*   **Traditional Automation** if `Task_Complexity <= 2` AND `Data_Structure <= 2` AND `Output_Objectivity <= 2`.")
    st.markdown(f"*   **Human Intervention Required** if `Risk_Level >= 4` OR (`Output_Objectivity >= 5` AND `Task_Complexity >= 5`).")
    st.markdown(f"*   **GenAI / LLM Automation** if `Data_Structure >= 4` OR `Task_Complexity >= 3`.")
    st.markdown(f"*   **Hybrid (Traditional + GenAI)** for all other cases.")
    st.markdown(f"This ordered logic ensures that high-risk tasks requiring human judgment are identified first, followed by clear cases for traditional or GenAI, with remaining tasks falling into a hybrid category.")

    st.markdown(f"---")
    st.subheader("Projects Classified into Automation Tiers")
    
    st.dataframe(st.session_state.project_scores_df[['tier', 'suit_efficiency', 'suit_risk']].sort_values(
        'suit_efficiency', ascending=False
    ).round(2), use_container_width=True)

    st.markdown(f"---")
    st.subheader("Tier Distribution")
    st.dataframe(st.session_state.project_scores_df['tier'].value_counts().reset_index().rename(columns={'index': 'Tier', 'tier': 'Count'}), use_container_width=True)

    st.markdown(f"**Explanation of Execution:**")
    st.markdown(f"Alex now has a clear classification for each potential automation project, indicating the most appropriate technological approach. This direct categorization helps him in initial resource planning and in guiding discussions with IT and engineering teams about the required infrastructure and expertise. For instance, tasks classified as 'Human Intervention Required' will prompt further investigation into necessary controls and human-in-the-loop processes, aligning with PGAM's ethical guidelines and risk management policies.")

# Page: 5. Visualize Landscape
elif st.session_state.current_page == "5. Visualize Landscape":
    st.title("5. Visualizing the Automation Project Landscape")
    st.markdown(f"To effectively communicate his findings to PGAM's leadership and other stakeholders, Alex needs compelling visualizations. These plots will summarize the project evaluations, highlight key trade-offs, and present the prioritized roadmap clearly.")
    st.markdown(f"Alex understands that visual representations are much more impactful than raw tables for strategic decision-making. He plans to generate several plots to illustrate:")
    st.markdown(f"1.  **Scorecard Heatmap (V1):** A visual overview of how each task scores across all dimensions, quickly identifying patterns.")
    st.markdown(f"2.  **Impact vs. Risk Quadrant Plot (V2):** To map projects based on their efficiency potential and risk level, categorizing them into strategic quadrants (e.g., Quick Wins, Caution Areas).")
    st.markdown(f"3.  **Suitability Ranking Bar Chart (V3):** A direct comparison of projects based on their weighted suitability, color-coded by automation tier.")
    st.markdown(f"4.  **Radar Charts (V4):** Detailed profiles for a few selected tasks, allowing for a deeper comparison across dimensions.")
    st.markdown(f"These visualizations will help stakeholders quickly grasp the key insights from the scorecard analysis and facilitate discussions on project prioritization.")

    st.markdown(f"---")
    st.subheader("Impact vs. Risk Quadrant Plot (V2)")
    fig_quadrant, ax_quadrant = plt.subplots(figsize=(12, 10))
    for task_id, row in st.session_state.project_scores_df.iterrows():
        ax_quadrant.scatter(row['efficiency_impact'], row['risk_level'],
                    color=row['color'], s=300, edgecolors='black', alpha=0.8, label=row['tier'] if row['tier'] not in ax_quadrant.get_legend_handles_labels()[1] else "")
        ax_quadrant.annotate(task_id.split(':')[0], # Display short ID
                     (row['efficiency_impact'] + 0.05, row['risk_level'] + 0.1),
                     fontsize=9, weight='bold')

    ax_quadrant.axhline(y=2.5, color='gray', linestyle='--', alpha=0.6) # Mid-point for Risk
    ax_quadrant.axvline(x=3.0, color='gray', linestyle='--', alpha=0.6) # Mid-point for Efficiency Impact

    ax_quadrant.text(4.5, 1.2, 'Quick Wins', fontsize=12, color='green', fontweight='bold', ha='center', va='center')
    ax_quadrant.text(1.5, 1.2, 'Low Impact / Low Risk', fontsize=12, color='gray', fontweight='bold', ha='center', va='center')
    ax_quadrant.text(1.5, 4.0, 'CAUTION (High Risk)', fontsize=12, color='red', fontweight='bold', ha='center', va='center')
    ax_quadrant.text(4.5, 4.0, 'Strategic / High Risk', fontsize=12, color='purple', fontweight='bold', ha='center', va='center')

    ax_quadrant.set_xlabel('Efficiency Impact (1-5)', fontsize=12)
    ax_quadrant.set_ylabel('Risk Level (1-5)', fontsize=12)
    ax_quadrant.set_title('Automation Priority: Impact vs. Risk Quadrant Plot', fontsize=16)
    ax_quadrant.set_xticks(np.arange(1, 6, 1))
    ax_quadrant.set_yticks(np.arange(1, 6, 1))
    ax_quadrant.grid(True, linestyle=':', alpha=0.7)

    handles, labels = ax_quadrant.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(ul)] for ul in unique_labels]
    ax_quadrant.legend(unique_handles, unique_labels, title="Automation Tier", bbox_to_anchor=(1.05, 1), loc='upper left')

    fig_quadrant.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig_quadrant)
    plt.close(fig_quadrant)

    st.markdown(f"---")
    st.subheader("Suitability Ranking Bar Chart (V3)")
    st.markdown(f"Projects ranked by their 'Efficiency-First' suitability score.")
    fig_bar, ax_bar = plt.subplots(figsize=(12, max(8, len(st.session_state.project_scores_df) * 0.7)))
    ranked_projects_eff = st.session_state.project_scores_df.sort_values('suit_efficiency', ascending=True)
    sns.barplot(x='suit_efficiency', y=ranked_projects_eff.index, data=ranked_projects_eff,
                palette=ranked_projects_eff['color'].tolist(), hue='tier', dodge=False, ax=ax_bar)

    ax_bar.set_xlabel('Weighted Suitability Score (Efficiency-First)', fontsize=12)
    ax_bar.set_ylabel('Candidate Projects', fontsize=12)
    ax_bar.set_title('Suitability Ranking: Projects by Efficiency-First Score', fontsize=16)
    ax_bar.legend(title="Automation Tier", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_bar.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    st.markdown(f"---")
    st.subheader("Radar Charts for Selected Tasks (V4)")
    st.markdown(f"Select 2-3 projects to compare their score profiles across dimensions.")
    
    # Multiselect for radar chart projects
    all_projects = st.session_state.project_scores_df.index.tolist()
    selected_tasks_for_radar = st.multiselect(
        "Select projects for Radar Chart comparison (max 3):",
        options=all_projects,
        default=all_projects[:min(3, len(all_projects))], # Default to first 3 if available
        key="radar_chart_selection"
    )

    if len(selected_tasks_for_radar) > 0:
        num_vars = len(st.session_state.score_columns)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1] # Complete the circle

        fig_radar, axes_radar = plt.subplots(1, len(selected_tasks_for_radar), figsize=(5 * len(selected_tasks_for_radar), 6), subplot_kw=dict(polar=True))
        if len(selected_tasks_for_radar) == 1: # Handle single subplot case
            axes_radar = [axes_radar]

        for i, task_id in enumerate(selected_tasks_for_radar):
            values = st.session_state.project_scores_df.loc[task_id, st.session_state.score_columns].tolist()
            values += values[:1] # Complete the circle for plotting

            ax = axes_radar[i]
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_yticklabels([]) # Hide radial labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(st.session_state.score_columns, fontsize=9, rotation=45, ha='right')
            ax.set_rlim(0, 5) # Radial limits up to 5

            ax.plot(angles, values, linewidth=2, linestyle='solid', label=task_id, color=st.session_state.project_scores_df.loc[task_id, 'color'])
            ax.fill(angles, values, color=st.session_state.project_scores_df.loc[task_id, 'color'], alpha=0.25)
            ax.set_title(f"Profile: {task_id.split(':')[0]} ({st.session_state.project_scores_df.loc[task_id, 'tier']})", va='bottom', fontsize=12)
        
        fig_radar.tight_layout()
        st.pyplot(fig_radar)
        plt.close(fig_radar)
    else:
        st.info("Please select at least one project for the Radar Chart.")

    st.markdown(f"---")
    st.subheader("Sensitivity Tornado (V5) - Conceptual")
    st.markdown(f"A conceptual visualization demonstrating how project suitability scores and rankings change when individual dimension weights are perturbed (e.g., ±20%).")
    st.markdown(f"*(Note: While a full sensitivity tornado chart requires more complex simulation, the current application demonstrates sensitivity by allowing comparison between 'Efficiency-First' and 'Risk-First' weight profiles. A full tornado chart would graphically show the impact of perturbing each dimension's weight on the overall suitability score for a selected project.)*")
    
    st.markdown(f"**Explanation of Execution:**")
    st.markdown(f"These visualizations provide Alex and PGAM's leadership with a comprehensive and intuitive overview of the automation landscape. The **Scorecard Heatmap** (displayed on the 'Score Projects' page) quickly reveals which tasks are complex or structured. The **Impact vs. Risk Quadrant Plot** immediately highlights potential 'Quick Wins' (high impact, low risk) and 'Caution' areas (high risk), enabling strategic resource allocation. The **Suitability Ranking Bar Chart** offers a clear prioritization list, while the **Radar Charts** allow for detailed task-level analysis, showing the unique profile of each project across the evaluation dimensions. This visual storytelling helps to foster better understanding and consensus among diverse stakeholders.")

# Page: 6. Deep Dive & ROI
elif st.session_state.current_page == "6. Deep Dive & ROI":
    st.title("6. Deep Dive: Hybrid Routing & ROI for the Top Project")
    st.markdown(f"For complex tasks, Alex knows that pure automation might not be feasible or desirable. He needs to consider a 'Hybrid Routing' approach, breaking down complex tasks into sub-tasks suitable for different automation tiers. Additionally, he must justify the top-ranked project with a solid business case, including ROI estimation.")
    st.markdown(f"Alex recognizes that many investment workflows are not monolithic; they consist of multiple sub-tasks. For tasks classified as 'Hybrid' or even 'GenAI/LLM Automation', a sophisticated approach might involve decomposing them into sub-tasks that can be handled by Traditional Automation, GenAI/LLM, or Human Intervention. This 'Hybrid Routing' maximizes automation potential while maintaining human oversight where critical.")
    st.markdown(f"Simultaneously, to secure budget and buy-in for the top-ranked project, Alex must present a compelling business case, including a clear Return on Investment (ROI) estimation. The ROI calculation is a critical metric for financial professionals, directly linking automation efforts to financial value.")
    st.markdown(f"The ROI formula for the top candidate project is:")
    st.markdown(r"$$ROI = \frac{\text{Annual Hours Saved} \cdot \text{Cost_per_Hour}}{\text{Development_Cost} + \text{Annual_Operating_Cost}} - 1$$")
    st.markdown(r"where **Annual Hours Saved** is the total hours saved per year by automating the task, **Cost_per_Hour** is the average cost of human labor per hour for the task, **Development_Cost** represents one-time costs to develop and implement the automation, and **Annual_Operating_Cost** represents recurring costs for maintaining the automation.")

    st.markdown(f"---")
    st.subheader("Conceptual Hybrid Routing Table")
    st.markdown(f"This demonstrates conceptually how a complex task can be broken down into sub-tasks and assigned appropriate automation tiers. (Example for 'T4: Monthly Performance Report')")
    st.dataframe(st.session_state.hybrid_routing_df, hide_index=True, use_container_width=True)

    total_manual_time = st.session_state.hybrid_routing_df['Time (Manual, min)'].sum()
    total_hybrid_time = st.session_state.hybrid_routing_df['Time (Auto, min)'].sum()
    if total_manual_time > 0:
        time_reduction_percent = (1 - (total_hybrid_time / total_manual_time)) * 100
    else:
        time_reduction_percent = 0
    st.markdown(f"Total manual time for T4: **{total_manual_time:.2f} minutes**")
    st.markdown(f"Total hybrid-routed time for T4: **{total_hybrid_time:.2f} minutes**")
    st.markdown(f"Estimated time reduction: **{time_reduction_percent:.2f}%**")

    st.markdown(f"---")
    st.subheader("ROI Estimation for Top-Ranked Project")
    
    # Select the top project (based on Efficiency-First suitability)
    sorted_projects = st.session_state.project_scores_df.sort_values('suit_efficiency', ascending=False)
    
    if not sorted_projects.empty:
        # Use st.selectbox to allow selecting any project, default to top-ranked
        st.session_state.selected_top_project_id = st.selectbox(
            "Select Project for ROI Analysis:",
            options=sorted_projects.index.tolist(),
            index=sorted_projects.index.tolist().index(sorted_projects.index[0]), # Default to top-ranked
            key="roi_project_selector"
        )
        
        top_project_data = st.session_state.project_scores_df.loc[st.session_state.selected_top_project_id]

        st.markdown(f"**Selected Project:** '{st.session_state.selected_top_project_id}'")
        st.markdown(f"**Automation Tier:** {top_project_data['tier']}")
        st.markdown(f"**Efficiency-First Suitability Score:** {top_project_data['suit_efficiency']:.2f}")

        st.markdown(f"---")
        st.markdown(f"#### Estimated Financial Impact Parameters")
        col_roi1, col_roi2 = st.columns(2)
        with col_roi1:
            st.session_state.roi_annual_hours_saved = st.number_input(
                "Annual Hours Saved:", min_value=0, value=st.session_state.roi_annual_hours_saved, step=1, key="roi_hours"
            )
            st.session_state.roi_cost_per_hour = st.number_input(
                "Cost per Hour ($):", min_value=0, value=st.session_state.roi_cost_per_hour, step=10, key="roi_cost_per_hour"
            )
        with col_roi2:
            st.session_state.roi_development_cost = st.number_input(
                "Development Cost ($):", min_value=0, value=st.session_state.roi_development_cost, step=100, key="roi_dev_cost"
            )
            st.session_state.roi_annual_operating_cost = st.number_input(
                "Annual Operating Cost ($):", min_value=0, value=st.session_state.roi_annual_operating_cost, step=10, key="roi_op_cost"
            )
        
        # Calculate ROI
        if (st.session_state.roi_development_cost + st.session_state.roi_annual_operating_cost) > 0:
            roi_calculated = (st.session_state.roi_annual_hours_saved * st.session_state.roi_cost_per_hour) / \
                             (st.session_state.roi_development_cost + st.session_state.roi_annual_operating_cost) - 1
        else:
            roi_calculated = 0 # Prevent division by zero

        st.markdown(f"---")
        st.markdown(f"#### Business Case Summary for '{st.session_state.selected_top_project_id}'")
        st.markdown(f"Estimated Annual Savings: **${st.session_state.roi_annual_hours_saved * st.session_state.roi_cost_per_hour:,.2f}**")
        st.markdown(f"Calculated ROI: **{roi_calculated:.2%}**")

        st.markdown(f"---")
        st.markdown(f"#### Conceptual Timeline & Risk Mitigation:")
        st.markdown(f"  Timeline: Estimated 4-6 weeks for development and initial rollout.")
        st.markdown(f"  Risk Mitigation: Implement robust data validation checks, phased rollout, and human-in-the-loop review for exceptions.")
    else:
        st.info("No projects available for ROI analysis. Please add projects on the 'Score Projects' page.")

    st.markdown(f"**Explanation of Execution:**")
    st.markdown(f"The **Hybrid Routing Table** clearly demonstrates to PGAM's technical teams how a seemingly complex task like a performance report can be efficiently broken down, leveraging different AI/ML capabilities while retaining human judgment for critical review. This optimizes automation efforts and ensures scalability.")
    st.markdown(f"The **ROI Estimation** provides a clear financial justification for pursuing the top-ranked project. By quantifying the financial benefits against costs, Alex can make a compelling case to the firm's management, highlighting the tangible value of strategic automation. This directly supports resource allocation decisions and demonstrates the practical application of his analysis in driving business value.")

# Page: 7. Automation Roadmap
elif st.session_state.current_page == "7. Automation Roadmap":
    st.title("7. Prioritized Automation Roadmap and Conclusion")
    st.markdown(f"Alex consolidates all findings into a comprehensive automation roadmap, categorizing projects into 'Quick Wins', 'Medium-Term', and 'Strategic' initiatives. This final output provides PGAM with a clear, actionable plan for its AI/ML automation journey.")
    st.markdown(f"Alex's ultimate goal is to present a clear, actionable automation roadmap to PGAM's executive committee. This roadmap summarizes all the analysis performed: the scores, suitability rankings, automation tiers, and a conceptual timeline. Categorizing projects by effort/timeline (Quick Wins, Medium-Term, Strategic) helps manage expectations and sequence implementation, ensuring a phased approach to AI adoption.")
    st.markdown(f"This roadmap serves as a strategic document, guiding the firm's investment in AI/ML and fostering a common understanding across different departments. It transforms raw data and complex analysis into a digestible, decision-driving output.")

    st.markdown(f"---")
    st.subheader("Prioritized Automation Roadmap")
    
    final_roadmap = st.session_state.project_scores_df.sort_values('suit_efficiency', ascending=False)
    
    st.dataframe(final_roadmap[['timeline', 'hours_saved_month', 'tier', 'suit_efficiency', 'suit_risk']].round(2), use_container_width=True)

    st.markdown(f"---")
    st.subheader("Final Recommendations")
    st.markdown(f"Alex has successfully applied the CFA Institute Automation Scorecard framework to prioritize AI/ML automation projects at Prosperity Global Asset Management.")
    st.markdown(f"The analysis provides:")
    st.markdown(f"1. A clear ranking of projects based on weighted suitability, allowing for strategic alignment.")
    st.markdown(f"2. Classification into appropriate automation tiers, guiding technology selection.")
    st.markdown(f"3. Visualizations that facilitate stakeholder communication and highlight key trade-offs.")
    st.markdown(f"4. A conceptual ROI for top projects, enabling data-driven investment decisions.")
    st.markdown(f"This structured approach moves PGAM beyond ad-hoc automation, ensuring that future AI/ML investments target high-value, feasible, and risk-managed opportunities.")

    st.markdown(f"**Explanation of Execution:**")
    st.markdown(f"This final roadmap is Alex's central deliverable. It provides PGAM's leadership with a clear, concise, and actionable plan. Projects are categorized to manage expectations regarding development timelines and required effort, fostering a realistic approach to AI adoption. The roadmap ensures that PGAM's automation efforts are strategically aligned, financially justifiable, and systematically managed, moving the firm closer to its goals of operational efficiency and innovation while mitigating risks. This completes Alex's objective of moving from ad-hoc decisions to a structured, data-driven framework for AI/ML project prioritization.")


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
