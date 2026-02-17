import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from typing import Dict, List, Any, Tuple

# --- 1. Global Constants for the Scorecard ---
# These constants define the core structure and criteria of the automation scorecard.
# They are defined globally as they are intrinsic to this specific analytical framework.

SCORECARD_DIMENSIONS = {
    'task_complexity': 'Repetitive (1) to highly variable (5)',
    'output_objectivity': 'Objective/verifiable (1) to subjective (5)',
    'data_structure': 'Structured/tabular (1) to unstructured/text (5)',
    'risk_level': 'Low risk of failure (1) to high risk (5)',
    'human_oversight': 'No review needed (0) to sign-off required (1)',
    'efficiency_impact': 'Minimal savings (1) to massive savings (5)'
}

SCORE_COLUMNS = list(SCORECARD_DIMENSIONS.keys())

# Define two distinct weight profiles for sensitivity analysis.
# Weights determine how each dimension contributes to the overall suitability score.
weights_efficiency_first = {
    'task_complexity': -0.10,          # Penalize complexity
    'output_objectivity': -0.10,       # Penalize subjectivity
    'data_structure': 0.00,            # Neutral on data structure for this profile
    'risk_level': -0.15,               # Penalize risk moderately
    'human_oversight': -0.15,          # Penalize required oversight moderately
    'efficiency_impact': 0.50          # Heavily reward high efficiency impact
}

weights_risk_first = {
    'task_complexity': -0.10,          # Penalize complexity
    'output_objectivity': -0.15,       # Penalize subjectivity more
    'data_structure': -0.05,           # Slightly penalize unstructured data
    'risk_level': -0.35,               # Heavily penalize risk
    'human_oversight': -0.15,          # Penalize required oversight moderately
    'efficiency_impact': 0.20          # Moderately reward efficiency impact
}

# Define a consistent color map for automation tiers across visualizations.
tier_colors_map = {
    'Traditional Automation': 'green',
    'GenAI / LLM Automation': 'blue',
    'Hybrid (Traditional + GenAI)': 'orange',
    'Human Intervention Required': 'red'
}

# --- 2. Core Data Processing Functions ---

def initialize_scorecard_data(raw_data: Dict[str, List[int]]) -> pd.DataFrame:
    """
    Initializes the DataFrame for automation project scores from raw data.

    Args:
        raw_data (Dict[str, List[int]]): A dictionary where keys are project IDs
                                         and values are lists of scores for each dimension,
                                         ordered according to SCORE_COLUMNS.

    Returns:
        pd.DataFrame: A DataFrame with projects as index and dimensions as columns.
    """
    scores_df = pd.DataFrame(raw_data, index=SCORE_COLUMNS).T
    scores_df.index.name = 'Project_ID'
    scores_df.columns.name = 'Dimension'
    return scores_df

def add_new_task(df: pd.DataFrame, task_id: str, scores: Dict[str, int]) -> pd.DataFrame:
    """
    Adds a new automation task to the scorecard DataFrame. This function assumes
    input scores are validated before being passed.

    Args:
        df (pd.DataFrame): The current DataFrame of project scores.
        task_id (str): A unique identifier for the new project.
        scores (Dict[str, int]): A dictionary of scores for the new task,
                                 mapping dimension names to their integer scores.

    Returns:
        pd.DataFrame: The updated DataFrame with the new task added.
                      Returns the original DataFrame if task_id already exists.
    """
    if task_id in df.index:
        print(f"Warning: Project ID '{task_id}' already exists. Skipping addition.")
        return df

    # Ensure scores dictionary contains all SCORE_COLUMNS for consistency
    validated_scores = {dim: scores.get(dim, 0) for dim in SCORE_COLUMNS}
    new_task = pd.DataFrame([validated_scores], index=[task_id], columns=SCORE_COLUMNS)
    return pd.concat([df, new_task])

def _calculate_weighted_score(row: pd.Series, weights: Dict[str, float]) -> float:
    """
    Helper function to calculate the weighted suitability score for a single project row.
    """
    score = sum(row[dim] * weight for dim, weight in weights.items() if dim in row.index)
    return score

def calculate_suitability_scores(
    df: pd.DataFrame,
    weights_efficiency: Dict[str, float] = weights_efficiency_first,
    weights_risk: Dict[str, float] = weights_risk_first
) -> pd.DataFrame:
    """
    Calculates weighted suitability scores for projects based on predefined weight profiles.

    Args:
        df (pd.DataFrame): The DataFrame of project scores.
        weights_efficiency (Dict[str, float]): Weights for the efficiency-first profile.
        weights_risk (Dict[str, float]): Weights for the risk-first profile.

    Returns:
        pd.DataFrame: The DataFrame with 'suit_efficiency' and 'suit_risk' columns added.
    """
    df_copy = df.copy() # Operate on a copy to avoid modifying the original DataFrame
    df_copy['suit_efficiency'] = df_copy.apply(lambda row: _calculate_weighted_score(row, weights_efficiency), axis=1)
    df_copy['suit_risk'] = df_copy.apply(lambda row: _calculate_weighted_score(row, weights_risk), axis=1)
    return df_copy

def _classify_tier_logic(row: pd.Series) -> str:
    """
    Helper function to classify a single project into an automation tier based on defined rules.
    """
    # Rule 1: Human Intervention Required (Highest priority for risk management)
    if row['risk_level'] >= 4 or (row['output_objectivity'] >= 5 and row['task_complexity'] >= 5):
        return 'Human Intervention Required'
    # Rule 2: Traditional Automation (Simple, structured, objective)
    elif row['task_complexity'] <= 2 and row['data_structure'] <= 2 and row['output_objectivity'] <= 2:
        return 'Traditional Automation'
    # Rule 3: GenAI / LLM Automation (Unstructured data or higher complexity)
    elif row['data_structure'] >= 4 or row['task_complexity'] >= 3:
        return 'GenAI / LLM Automation'
    # Rule 4: Hybrid (Default for remaining cases)
    else:
        return 'Hybrid (Traditional + GenAI)'

def classify_automation_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies each project into an automation tier and assigns a color based on the tier.

    Args:
        df (pd.DataFrame): The DataFrame of project scores, including all dimensions.

    Returns:
        pd.DataFrame: The DataFrame with 'tier' and 'color' columns added.
    """
    df_copy = df.copy()
    df_copy['tier'] = df_copy.apply(_classify_tier_logic, axis=1)
    df_copy['color'] = df_copy['tier'].map(tier_colors_map)
    return df_copy

# --- 3. Visualization Functions ---
# Each visualization function generates a matplotlib Figure object, suitable for embedding
# into web applications (e.g., by converting to a BytesIO object).

def generate_heatmap(df: pd.DataFrame, dimensions_keys: List[str] = SCORE_COLUMNS) -> plt.Figure:
    """
    Generates a heatmap of automation task scores across dimensions.

    Args:
        df (pd.DataFrame): DataFrame containing project scores.
        dimensions_keys (List[str]): List of dimension column names to include in the heatmap.

    Returns:
        plt.Figure: A matplotlib Figure object containing the heatmap.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(df[dimensions_keys], annot=True, fmt='.0f', cmap='viridis', linewidths=.5, linecolor='lightgray', ax=ax)
    ax.set_title('Automation Scorecard Heatmap: Task Scores Across Dimensions', fontsize=16)
    ax.set_xlabel('Score Dimensions', fontsize=12)
    ax.set_ylabel('Candidate Projects', fontsize=12)
    plt.tight_layout()
    return fig

def generate_impact_risk_quadrant_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Generates an Impact vs. Risk quadrant plot for automation projects.

    Args:
        df (pd.DataFrame): DataFrame containing project scores, including 'efficiency_impact',
                           'risk_level', 'tier', and 'color' columns.

    Returns:
        plt.Figure: A matplotlib Figure object containing the quadrant plot.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    for task_id, row in df.iterrows():
        ax.scatter(row['efficiency_impact'], row['risk_level'],
                   color=row['color'], s=300, edgecolors='black', alpha=0.8,
                   label=row['tier'] if row['tier'] not in ax.get_legend_handles_labels()[1] else "")
        ax.annotate(task_id.split(':')[0],
                    (row['efficiency_impact'] + 0.05, row['risk_level'] + 0.1),
                    fontsize=9, weight='bold')

    # Quadrant lines and labels
    ax.axhline(y=2.5, color='gray', linestyle='--', alpha=0.6)
    ax.axvline(x=3.0, color='gray', linestyle='--', alpha=0.6)

    ax.text(4.5, 1.2, 'Quick Wins', fontsize=12, color='green', fontweight='bold', ha='center', va='center')
    ax.text(1.5, 1.2, 'Low Impact / Low Risk', fontsize=12, color='gray', fontweight='bold', ha='center', va='center')
    ax.text(1.5, 4.0, 'CAUTION (High Risk)', fontsize=12, color='red', fontweight='bold', ha='center', va='center')
    ax.text(4.5, 4.0, 'Strategic / High Risk', fontsize=12, color='purple', fontweight='bold', ha='center', va='center')

    ax.set_xlabel('Efficiency Impact (1-5)', fontsize=12)
    ax.set_ylabel('Risk Level (1-5)', fontsize=12)
    ax.set_title('Automation Priority: Impact vs. Risk Quadrant Plot', fontsize=16)
    ax.set_xticks(np.arange(1, 6, 1))
    ax.set_yticks(np.arange(1, 6, 1))
    ax.grid(True, linestyle=':', alpha=0.7)

    # Custom legend for unique tier labels
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(ul)] for ul in unique_labels]
    ax.legend(unique_handles, unique_labels, title="Automation Tier", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

def generate_suitability_ranking_bar_chart(df: pd.DataFrame) -> plt.Figure:
    """
    Generates a horizontal bar chart showing project suitability ranking (Efficiency-First).

    Args:
        df (pd.DataFrame): DataFrame containing project scores, including 'suit_efficiency',
                           'tier', and 'color' columns.

    Returns:
        plt.Figure: A matplotlib Figure object containing the bar chart.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    # Sort by 'suit_efficiency' for ranking, ascending for horizontal bar plot order
    ranked_projects_eff = df.sort_values('suit_efficiency', ascending=True)
    sns.barplot(x='suit_efficiency', y=ranked_projects_eff.index, data=ranked_projects_eff,
                palette=ranked_projects_eff['color'].tolist(), hue='tier', dodge=False, ax=ax)

    ax.set_xlabel('Weighted Suitability Score (Efficiency-First)', fontsize=12)
    ax.set_ylabel('Candidate Projects', fontsize=12)
    ax.set_title('Suitability Ranking: Projects by Efficiency-First Score', fontsize=16)
    ax.legend(title="Automation Tier", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

def generate_radar_charts(df: pd.DataFrame, selected_tasks: List[str], dimensions_keys: List[str] = SCORE_COLUMNS) -> plt.Figure:
    """
    Generates radar charts for a selection of automation tasks.

    Args:
        df (pd.DataFrame): DataFrame containing project scores, including 'color' and 'tier' columns.
        selected_tasks (List[str]): List of project IDs for which to generate radar charts.
        dimensions_keys (List[str]): List of dimension column names to include in the radar chart.

    Returns:
        plt.Figure: A matplotlib Figure object containing the radar charts.
                    Returns None if no valid tasks are selected.
    """
    valid_selected_tasks = [task for task in selected_tasks if task in df.index]
    if not valid_selected_tasks:
        return None

    num_vars = len(dimensions_keys)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1] # Complete the circle for plotting

    fig, axes = plt.subplots(1, len(valid_selected_tasks), figsize=(5 * len(valid_selected_tasks), 6), subplot_kw=dict(polar=True))
    if len(valid_selected_tasks) == 1: # Handle single subplot case
        axes = [axes]

    max_score = 5 # Assuming 1-5 scale for all dimensions for radar chart visualization

    for i, task_id in enumerate(valid_selected_tasks):
        values = df.loc[task_id, dimensions_keys].tolist()
        values += values[:1] # Complete the circle for plotting

        ax = axes[i]
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_yticklabels([]) # Hide radial labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions_keys, fontsize=9, rotation=45, ha='right')
        ax.set_rlim(0, max_score) # Set radial limits
        ax.set_rlabel_position(0) # Hide default radial axis labels

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=task_id, color=df.loc[task_id, 'color'])
        ax.fill(angles, values, color=df.loc[task_id, 'color'], alpha=0.25)
        ax.set_title(f"Profile: {task_id.split(':')[0]} ({df.loc[task_id, 'tier']})", va='bottom', fontsize=12)

    plt.tight_layout()
    return fig

# --- 4. Business Case & Roadmap Functions ---

def generate_conceptual_hybrid_routing_table(task_id: str = 'T4: Performance Report') -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Generates a conceptual hybrid routing table for a given complex task.
    This function uses hardcoded data for demonstration purposes, modeling how
    a complex task might be broken down into sub-tasks and automated.

    Args:
        task_id (str): The ID of the task for which to generate the routing table.
                       (Currently, the data is specific to 'T4: Performance Report').

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]: A tuple containing:
            - pd.DataFrame: The hybrid routing table.
            - Dict[str, float]: A dictionary with calculated total manual time, total hybrid time,
                                and estimated time reduction percentage.
    """
    hybrid_routing_data = {
        'Sub-Task': [
            'Retrieve holdings/prices',
            'Calculate returns/attribution',
            'Generate charts from data',
            'Draft market commentary',
            'Customize for client (template)',
            'Review and approve final report'
        ],
        'Tier': [
            'Traditional Automation',
            'Traditional Automation',
            'GenAI / LLM Automation',
            'GenAI / LLM Automation',
            'GenAI / LLM Automation',
            'Human Intervention Required'
        ],
        'Time (Manual, min)': [15, 30, 20, 45, 20, 15],
        'Time (Auto, min)':   [0.17, 0.08, 0.17, 0.5, 0.25, 10] # Automated/human intervention time
    }
    hybrid_routing_df = pd.DataFrame(hybrid_routing_data)

    total_manual_time = hybrid_routing_df['Time (Manual, min)'].sum()
    total_hybrid_time = hybrid_routing_df['Time (Auto, min)'].sum()
    time_reduction_percent = (1 - (total_hybrid_time / total_manual_time)) * 100 if total_manual_time > 0 else 0

    metrics = {
        'total_manual_time': total_manual_time,
        'total_hybrid_time': total_hybrid_time,
        'time_reduction_percent': time_reduction_percent
    }
    return hybrid_routing_df, metrics

def calculate_roi_for_project(
    df: pd.DataFrame,
    project_id: str,
    annual_hours_saved: float,
    cost_per_hour: float,
    development_cost: float,
    annual_operating_cost: float
) -> Dict[str, Any]:
    """
    Calculates and returns ROI details for a specific project.
    Conceptual values for ROI calculation (example based on T1: Data Ingestion from PDF).

    Args:
        df (pd.DataFrame): DataFrame containing project data, including 'tier' and 'suit_efficiency'.
        project_id (str): The ID of the project for which to calculate ROI.
        annual_hours_saved (float): Estimated annual hours saved by automating this project.
        cost_per_hour (float): Burdened cost per hour of the resource whose time is saved.
        development_cost (float): One-time cost to develop the automation solution.
        annual_operating_cost (float): Annual cost for maintenance, monitoring, etc.

    Returns:
        Dict[str, Any]: A dictionary containing ROI metrics and project details.
    """
    if project_id not in df.index:
        return {"error": f"Project ID '{project_id}' not found in DataFrame."}

    project_data = df.loc[project_id]

    annual_savings = annual_hours_saved * cost_per_hour
    total_cost = development_cost + annual_operating_cost
    roi = (annual_savings / total_cost) - 1 if total_cost > 0 else float('inf')

    return {
        'project_id': project_id,
        'automation_tier': project_data.get('tier', 'N/A'),
        'efficiency_suitability_score': project_data.get('suit_efficiency', 0.0),
        'annual_hours_saved': annual_hours_saved,
        'cost_per_hour': cost_per_hour,
        'annual_savings': annual_savings,
        'development_cost': development_cost,
        'annual_operating_cost': annual_operating_cost,
        'calculated_roi': roi,
        'conceptual_timeline': "Estimated 4-6 weeks for development and initial rollout.",
        'risk_mitigation': "Implement robust data validation checks, phased rollout, and human-in-the-loop review for exceptions."
    }

def generate_roadmap(
    df: pd.DataFrame,
    initial_hours_saved_map: Dict[str, int],
    initial_effort_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Generates a prioritized automation roadmap by adding 'hours_saved_month',
    'effort', and 'timeline' columns and sorting by efficiency suitability.

    Args:
        df (pd.DataFrame): DataFrame containing project data, including suitability scores.
        initial_hours_saved_map (Dict[str, int]): Initial mapping of project IDs to monthly hours saved.
        initial_effort_map (Dict[str, str]): Initial mapping of project IDs to effort levels (Low, Med, High).

    Returns:
        pd.DataFrame: A DataFrame representing the prioritized roadmap with key details.
    """
    df_copy = df.copy()

    # Apply initial hours saved and effort, with defaults for new/custom tasks
    df_copy['hours_saved_month'] = df_copy.index.map(lambda x: initial_hours_saved_map.get(x, 7)) # Default 7 for custom tasks
    df_copy['effort'] = df_copy.index.map(lambda x: initial_effort_map.get(x, 'Med')) # Default 'Med' for custom tasks

    def _categorize_timeline(row: pd.Series) -> str:
        """Helper to assign timeline categories based on effort."""
        if row['effort'] == 'Low':
            return 'Quick Win (<1 week)'
        elif row['effort'] == 'Med':
            return 'Medium-Term (1-4 weeks)'
        else: # 'High' effort
            return 'Strategic (1-3 months)'

    df_copy['timeline'] = df_copy.apply(_categorize_timeline, axis=1)

    # Sort the roadmap by efficiency suitability to prioritize
    final_roadmap = df_copy.sort_values('suit_efficiency', ascending=False)
    return final_roadmap[['timeline', 'hours_saved_month', 'tier']]

# --- 5. Orchestrator Function ---

def run_full_analysis(
    initial_raw_data: Dict[str, List[int]],
    custom_tasks_data: List[Dict[str, Any]],
    roi_parameters: Dict[str, Any],
    roadmap_initial_data: Dict[str, Dict[str, Any]],
    selected_radar_tasks: List[str] = None
) -> Dict[str, Any]:
    """
    Orchestrates the full automation scorecard analysis, including data processing,
    suitability scoring, tier classification, visualization generation, business case
    summaries, and roadmap planning.

    Args:
        initial_raw_data (Dict[str, List[int]]): Initial synthetic dataset of candidate tasks.
        custom_tasks_data (List[Dict[str, Any]]): A list of dictionaries, each containing
                                                  'task_id' and 'scores' for custom tasks.
        roi_parameters (Dict[str, Any]): Parameters for ROI calculation, including
                                         'annual_hours_saved', 'cost_per_hour',
                                         'development_cost', 'annual_operating_cost'.
        roadmap_initial_data (Dict[str, Dict[str, Any]]): Contains 'hours_saved_month_map'
                                                          and 'effort_map' for roadmap generation.
        selected_radar_tasks (List[str], optional): List of task IDs for which to generate
                                                     radar charts. Defaults to a selection.

    Returns:
        Dict[str, Any]: A dictionary containing all generated DataFrames, matplotlib Figure objects,
                        and summary metrics from the analysis.
    """
    if selected_radar_tasks is None:
        selected_radar_tasks = ['T1: Data Ingestion', 'T2: 10-K Risk Extract', 'T8: Thesis Drafting']

    results = {}

    # 1. Initialize data
    scores_df = initialize_scorecard_data(initial_raw_data)
    results['initial_scores_df'] = scores_df.copy()

    # 2. Add custom tasks
    for task_info in custom_tasks_data:
        scores_df = add_new_task(scores_df, task_info['task_id'], task_info['scores'])
    results['scores_df_with_custom_tasks'] = scores_df.copy()

    # 3. Calculate suitability scores
    scores_df = calculate_suitability_scores(scores_df)
    results['scores_df_with_suitability'] = scores_df.copy()

    results['ranking_efficiency'] = scores_df[['suit_efficiency', 'suit_risk']].sort_values(
        'suit_efficiency', ascending=False
    )
    results['ranking_risk'] = scores_df[['suit_efficiency', 'suit_risk']].sort_values(
        'suit_risk', ascending=False
    )

    # 4. Classify automation tiers
    scores_df = classify_automation_tier(scores_df)
    results['scores_df_with_tiers'] = scores_df.copy()
    results['tier_distribution'] = scores_df['tier'].value_counts()

    # 5. Generate Visualizations (Figures will be returned as matplotlib Figure objects)
    results['heatmap_fig'] = generate_heatmap(scores_df, dimensions_keys=SCORE_COLUMNS)
    results['quadrant_plot_fig'] = generate_impact_risk_quadrant_plot(scores_df)
    results['suitability_bar_chart_fig'] = generate_suitability_ranking_bar_chart(scores_df)
    results['radar_charts_fig'] = generate_radar_charts(scores_df, selected_radar_tasks, dimensions_keys=SCORE_COLUMNS)

    # 6. Hybrid Routing Table for a conceptual task (e.g., T4)
    hybrid_routing_df, hybrid_metrics = generate_conceptual_hybrid_routing_table('T4: Performance Report')
    results['hybrid_routing_table_T4'] = hybrid_routing_df
    results['hybrid_routing_metrics_T4'] = hybrid_metrics

    # 7. ROI Estimation for the top-ranked project
    # Ensure there's at least one project to select
    if not scores_df.empty:
        top_project_id = scores_df.sort_values('suit_efficiency', ascending=False).index[0]
        roi_summary = calculate_roi_for_project(
            scores_df,
            top_project_id,
            **roi_parameters
        )
        results['roi_summary'] = roi_summary
    else:
        results['roi_summary'] = {"error": "No projects available for ROI calculation."}


    # 8. Prioritized Automation Roadmap
    prioritized_roadmap_df = generate_roadmap(
        scores_df,
        roadmap_initial_data['hours_saved_month_map'],
        roadmap_initial_data['effort_map']
    )
    results['prioritized_roadmap'] = prioritized_roadmap_df

    # Final recommendations text
    results['final_recommendations'] = (
        "Alex has successfully applied the CFA Institute Automation Scorecard framework to prioritize AI/ML automation projects at Prosperity Global Asset Management.\n"
        "The analysis provides:\n"
        "1. A clear ranking of projects based on weighted suitability, allowing for strategic alignment.\n"
        "2. Classification into appropriate automation tiers, guiding technology selection.\n"
        "3. Visualizations that facilitate stakeholder communication and highlight key trade-offs.\n"
        "4. A conceptual ROI for top projects, enabling data-driven investment decisions.\n\n"
        "This structured approach moves PGAM beyond ad-hoc automation, ensuring that future AI/ML investments target high-value, feasible, and risk-managed opportunities."
    )

    return results
