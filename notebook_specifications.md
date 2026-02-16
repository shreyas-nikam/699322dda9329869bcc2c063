
# Automation Project Evaluation: A CFA Institute Scorecard Approach for Investment Professionals

## Case Study: Systematically Prioritizing AI/ML Projects at Prosperity Global Asset Management

**Persona:** Alex Chen, CFA, Investment Operations Manager at Prosperity Global Asset Management.  
**Organization:** Prosperity Global Asset Management (PGAM), a mid-sized asset management firm.

**Scenario:**
Alex Chen, CFA, leads a team at Prosperity Global Asset Management responsible for optimizing investment workflows. Recently, PGAM has been exploring the integration of AI/ML to enhance operational efficiency and innovation. However, ad-hoc automation efforts have led to mixed results, with some projects failing to deliver expected value due to misalignment with strategic goals or underestimation of risks.

To address this, Alex is tasked with implementing a structured framework for evaluating and prioritizing potential AI/ML automation initiatives. He aims to move beyond intuitive decisions and apply a systematic approach inspired by the CFA Institute's Automation Scorecard. This will enable him to identify high-value, feasible, and low-risk automation opportunities, ensuring strategic alignment and maximizing return on investment. The goal is to build a clear, prioritized automation roadmap that resonates with various stakeholders across portfolio management, compliance, and IT.

This notebook will guide Alex through a step-by-step workflow, from defining key evaluation dimensions to classifying projects and estimating their potential ROI, all within a practical, hands-on environment.

---

### 1. Setup and Data Loading

Alex begins by setting up his analytical environment and preparing the initial dataset of potential automation tasks. This step ensures all necessary tools are available and the raw data for evaluation is structured correctly.

#### 1.1 Install Required Libraries

Before starting, Alex needs to install the necessary Python libraries for data manipulation, numerical computations, and visualizations.

```python
!pip install pandas numpy matplotlib seaborn
```

#### 1.2 Import Required Dependencies

Next, Alex imports the installed libraries. These will be used throughout the notebook for data handling, calculations, and generating insightful plots.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io # For handling string data as a file
from math import pi # For radar charts
```

---

### 2. Defining the Automation Scorecard Dimensions

To systematically evaluate automation projects, Alex needs to clearly define the key dimensions that will be used. These dimensions are crucial for consistent scoring and strategic alignment, based on the CFA Institute's Automation Scorecard framework.

Alex understands that a robust framework requires clear definitions for each evaluation criterion. This helps ensure consistency in scoring across different projects and reduces subjectivity in the evaluation process.

The weighted suitability score will be calculated using the Multi-Criteria Decision Analysis (MCDA) formula:
$$S_j = \sum_{d=1}^{D} W_d \cdot X_{jd}$$
Where:
- $S_j$ is the suitability score for project $j$.
- $W_d$ is the weight assigned to dimension $d$. A positive weight means a higher score on this dimension contributes positively to suitability; a negative weight means a higher score contributes negatively.
- $X_{jd}$ is the raw score of project $j$ on dimension $d$.
- $D$ is the total number of dimensions.

```python
# Markdown Cell - Story + Context + Real-World Relevance

Alex needs to programmatically define the 6 key dimensions for evaluating automation suitability. These definitions guide how each project will be scored, ensuring a standardized approach. He also considers the scoring scale for each dimension.

*   **Task Complexity:** Measures the intricacy and variability of the task. (1: Repetitive, simple; 5: Highly variable, complex)
*   **Output Objectivity:** Assesses how easily the task's output can be objectively verified. (1: Objective/verifiable; 5: Subjective, requires interpretation)
*   **Data Structure:** Describes the nature of the data involved. (1: Structured/tabular; 5: Unstructured/textual, contextual)
*   **Risk Level:** Evaluates the potential impact of an automation error. (1: Low risk of failure; 5: High risk, critical impact)
*   **Human Oversight Requirement:** Indicates whether human review or sign-off is mandatory. (0: No review needed; 1: Sign-off required)
*   **Impact on Efficiency:** Quantifies the potential time or cost savings. (1: Minimal savings; 5: Massive savings)

```python
# Code cell (function definition + function execution)

# Define the scorecard dimensions and their descriptions
SCORECARD_DIMENSIONS = {
    'task_complexity': 'Repetitive (1) to highly variable (5)',
    'output_objectivity': 'Objective/verifiable (1) to subjective (5)',
    'data_structure': 'Structured/tabular (1) to unstructured/text (5)',
    'risk_level': 'Low risk of failure (1) to high risk (5)',
    'human_oversight': 'No review needed (0) to sign-off required (1)',
    'efficiency_impact': 'Minimal savings (1) to massive savings (5)'
}

# Print the dimensions for review
print("CFA Institute Automation Scorecard Dimensions:")
print("=" * 60)
for dim, desc in SCORECARD_DIMENSIONS.items():
    print(f"- {dim:20s}: {desc}")

# Define the columns that represent scores (excluding 'ID' and 'Description')
SCORE_COLUMNS = list(SCORECARD_DIMENSIONS.keys())
```

```markdown
**Explanation of Execution:**

By explicitly defining each dimension and its scoring scale, Alex establishes a common language for discussing automation suitability within PGAM. This clarity is vital for gathering consistent inputs from different department heads and ensuring that all stakeholders evaluate projects using the same criteria. This structured definition forms the foundation for objective analysis.
```

---

### 3. Scoring Candidate Automation Projects

Alex identifies several potential automation projects within PGAM's investment workflows. He then scores each project against the defined dimensions, using a synthetic dataset that represents typical tasks. This hands-on scoring exercise is crucial for applying the framework to real-world scenarios.

```markdown
# Markdown Cell - Story + Context + Real-World Relevance

Alex has compiled a list of tasks that his team performs regularly. To apply the scorecard, he needs to assign numerical scores to each task across the 6 dimensions. This initial scoring forms the raw data for the Multi-Criteria Decision Analysis (MCDA).

He will start with a pre-defined set of 8 tasks and their scores, representing common investment operations. Later, he'll add a few custom tasks to simulate how he would evaluate new ideas.

**Important Note:** As highlighted by the CFA Institute, scores are subjective inputs. Different analysts or departments might score the same task differently based on their specific context or risk tolerance. The value of this scorecard lies in structuring the conversation and surfacing these disagreements early, *before* committing significant resources.

```python
# Code cell (function definition + function execution)

# Synthetic dataset of 8 candidate automation tasks with their scores
data = {
    'T1: Data Ingestion':        [1, 1, 1, 1, 0, 4],
    'T2: 10-K Risk Extract':     [3, 3, 5, 2, 1, 5],
    'T3: Earnings Summary':      [3, 3, 5, 2, 1, 5],
    'T4: Performance Report':    [4, 4, 2, 2, 1, 4],
    'T5: Compliance Check':      [2, 1, 1, 5, 1, 3],
    'T6: Credit Recalib':        [4, 2, 2, 4, 1, 3],
    'T7: Rebalancing':           [2, 1, 1, 4, 1, 4],
    'T8: Thesis Drafting':       [5, 5, 5, 3, 1, 4]
}

# Create a DataFrame from the synthetic data
scores_df = pd.DataFrame(data, index=SCORE_COLUMNS).T
scores_df.index.name = 'Project_ID'
scores_df.columns.name = 'Dimension'

print("Initial Scores for Candidate Automation Projects:")
print("=" * 60)
print(scores_df)

# Function to add a new custom task (user input)
def add_custom_task(df):
    print("\n--- Add a New Custom Automation Task ---")
    task_id = input("Enter a unique Project ID (e.g., 'T9: My Custom Task'): ")
    if task_id in df.index:
        print(f"Error: Project ID '{task_id}' already exists. Please choose a different one.")
        return df

    task_description = input(f"Enter a brief description for '{task_id}': ")
    scores = {}
    print(f"Enter scores for '{task_id}' (1-5, or 0/1 for human_oversight):")
    for dim, desc in SCORECARD_DIMENSIONS.items():
        while True:
            try:
                score = int(input(f"  {dim} ({desc}): "))
                if dim == 'human_oversight' and score not in [0, 1]:
                    print("Human Oversight must be 0 or 1. Please try again.")
                elif dim != 'human_oversight' and not (1 <= score <= 5):
                    print("Score must be between 1 and 5. Please try again.")
                else:
                    scores[dim] = score
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    new_task = pd.DataFrame([scores], index=[task_id], columns=SCORE_COLUMNS)
    new_task['Description'] = task_description # Add description for potential future use
    
    # Check if 'Description' column exists in original df, if not add it.
    # For now, let's keep scores_df purely numerical as intended for calculations
    # and manage descriptions separately if needed, or add them as an initial column.
    # For this lab, scores_df will remain scores only.
    
    return pd.concat([df, new_task])

# Alex decides to add 2 custom tasks from his team's workflow
print("\nAlex is adding 2 custom tasks:")
scores_df = add_custom_task(scores_df)
scores_df = add_custom_task(scores_df)

print("\nUpdated Scores for Candidate Automation Projects (including custom tasks):")
print("=" * 60)
print(scores_df)
```

```markdown
**Explanation of Execution:**

Alex has now populated his scorecard with initial ratings for various tasks. The inclusion of custom tasks demonstrates the framework's flexibility to incorporate new ideas directly from his team's experience. This scored matrix serves as the foundation for quantitative analysis, enabling him to move towards calculating overall suitability. The raw scores immediately highlight tasks that might be complex, high-risk, or have significant efficiency potential, providing initial insights even before weighted calculations.
```

---

### 4. Prioritizing with Weighted Suitability Scores (MCDA) and Sensitivity Analysis

Understanding that different strategic priorities exist within PGAM, Alex needs to evaluate projects based on varying criteria. He will use Multi-Criteria Decision Analysis (MCDA) with different weight profiles to compute suitability scores and perform a sensitivity analysis. This allows him to see how project rankings change based on PGAM's strategic focus (e.g., 'Efficiency-First' versus 'Risk-First').

```markdown
# Markdown Cell - Story + Context + Real-World Relevance

Alex knows that PGAM's priorities can shift. Sometimes, the firm needs to prioritize quick wins and efficiency gains. At other times, minimizing risk is paramount. To account for these strategic nuances, he defines two distinct weight profiles for the scorecard dimensions: 'Efficiency-First' and 'Risk-First'.

The formula for calculating the weighted suitability score, $S_j = \sum_{d=1}^{D} W_d \cdot X_{jd}$, allows for flexibility. A higher positive weight on 'efficiency_impact', for instance, means tasks with high efficiency potential will rank higher under an 'Efficiency-First' profile. Conversely, a higher negative weight on 'risk_level' means high-risk tasks will be penalized more heavily under a 'Risk-First' profile.

This sensitivity analysis is a powerful tool to understand the robustness of recommendations and to facilitate discussions among stakeholders about trade-offs.

```python
# Code cell (function definition + function execution)

# Define two distinct weight profiles for sensitivity analysis
# Weights are chosen such that positive values indicate desirability, negative values undesirability.
# The sum of absolute weights can be normalized or left as is for relative scaling.
# Here, we keep relative scaling as in the example.

# Profile A: Efficiency-First (prioritizes tasks with high efficiency impact, less complexity/risk)
weights_efficiency_first = {
    'task_complexity': -0.10,          # Penalize complexity
    'output_objectivity': -0.10,       # Penalize subjectivity
    'data_structure': 0.00,            # Neutral on data structure for this profile
    'risk_level': -0.15,               # Penalize risk moderately
    'human_oversight': -0.15,          # Penalize required oversight moderately
    'efficiency_impact': 0.50          # Heavily reward high efficiency impact
}

# Profile B: Risk-First (heavily penalizes risk and subjectivity)
weights_risk_first = {
    'task_complexity': -0.10,          # Penalize complexity
    'output_objectivity': -0.15,       # Penalize subjectivity more
    'data_structure': -0.05,           # Slightly penalize unstructured data
    'risk_level': -0.35,               # Heavily penalize risk
    'human_oversight': -0.15,          # Penalize required oversight moderately
    'efficiency_impact': 0.20          # Moderately reward efficiency impact
}

# Function to calculate weighted suitability score for a given row (task) and weights
def calculate_weighted_score(row, weights):
    score = sum(row[dim] * weight for dim, weight in weights.items())
    return score

# Apply the weighting profiles to the scores DataFrame
scores_df['suit_efficiency'] = scores_df.apply(calculate_weighted_score, axis=1, weights=weights_efficiency_first)
scores_df['suit_risk'] = scores_df.apply(calculate_weighted_score, axis=1, weights=weights_risk_first)

# Display the suitability scores and ranking by 'suit_efficiency'
ranking_efficiency = scores_df[['suit_efficiency', 'suit_risk']].sort_values(
    'suit_efficiency', ascending=False
)

print("\nWeighted Suitability Scores (Efficiency-First Ranking):")
print("=" * 60)
print(ranking_efficiency.round(2))

# Also display ranking by 'suit_risk' for comparison
ranking_risk = scores_df[['suit_efficiency', 'suit_risk']].sort_values(
    'suit_risk', ascending=False
)
print("\nWeighted Suitability Scores (Risk-First Ranking):")
print("=" * 60)
print(ranking_risk.round(2))

# Add 'Description' column for display purposes, assuming original tasks had descriptions
# For simplicity in this lab, we'll embed descriptions into the project ID for now (e.g., 'T1: Data Ingestion')
# If a separate 'Description' column were needed, we'd ensure it's propagated from original data or user input.
```

```markdown
**Explanation of Execution:**

By applying two different weight profiles, Alex can observe how project priorities shift based on strategic focus. For example, a task that ranks high under 'Efficiency-First' might drop significantly under 'Risk-First' if it involves high inherent risk. This sensitivity analysis is critical for Alex to present a nuanced view to PGAM's leadership, facilitating discussions on strategic trade-offs and ensuring that automation efforts align with current firm objectives. The numerical scores now provide a quantitative basis for comparing and ranking projects.
```

---

### 5. Classifying Projects into Automation Tiers

Not all automation is created equal. Alex needs to categorize each project into the most appropriate automation tier to guide technology choices and implementation strategies. This involves applying rule-based logic to the project scores.

```markdown
# Markdown Cell - Story + Context + Real-World Relevance

After calculating suitability scores, Alex's next step is to classify each project into one of four automation tiers:
1.  **Traditional Automation:** Rules-based, structured, objective tasks.
2.  **GenAI / LLM Automation:** Tasks involving unstructured data, context, and generation.
3.  **Human Intervention Required:** High-risk or highly subjective tasks requiring human oversight.
4.  **Hybrid (Traditional + GenAI):** Tasks that combine elements of both structured processing and generative capabilities.

This classification is crucial for determining the right technological approach and resource allocation. The classification logic is based on a decision tree derived from the CFA Institute framework:

*   **Traditional Automation** if `Task_Complexity <= 2` AND `Data_Structure <= 2` AND `Output_Objectivity <= 2`.
*   **Human Intervention Required** if `Risk_Level >= 4` OR (`Output_Objectivity >= 5` AND `Task_Complexity >= 5`).
*   **GenAI / LLM Automation** if `Data_Structure >= 4` OR `Task_Complexity >= 3`.
*   **Hybrid (Traditional + GenAI)** for all other cases.

This ordered logic ensures that high-risk tasks requiring human judgment are identified first, followed by clear cases for traditional or GenAI, with remaining tasks falling into a hybrid category.

```python
# Code cell (function definition + function execution)

# Function to classify a project into an automation tier
def classify_tier(row):
    # Rule 1: Human Intervention Required (Highest priority for risk management)
    # If Risk_Level is high (>=4) OR if output is very subjective (>=5) AND task is very complex (>=5)
    if row['risk_level'] >= 4 or (row['output_objectivity'] >= 5 and row['task_complexity'] >= 5):
        return 'Human Intervention Required'
    
    # Rule 2: Traditional Automation (Simple, structured, objective)
    # If task is simple (<=2), data is structured (<=2), and output is objective (<=2)
    elif row['task_complexity'] <= 2 and row['data_structure'] <= 2 and row['output_objectivity'] <= 2:
        return 'Traditional Automation'
    
    # Rule 3: GenAI / LLM Automation (Unstructured data or higher complexity)
    # If data is unstructured (>=4) OR task is complex (>=3)
    elif row['data_structure'] >= 4 or row['task_complexity'] >= 3:
        return 'GenAI / LLM Automation'
    
    # Rule 4: Hybrid (Default for remaining cases)
    else:
        return 'Hybrid (Traditional + GenAI)'

# Apply the classification function to each project
scores_df['tier'] = scores_df.apply(classify_tier, axis=1)

# Display projects with their assigned tiers and suitability scores, sorted by efficiency suitability
print("\nProjects Classified into Automation Tiers (ranked by Efficiency Suitability):")
print("=" * 70)
print(scores_df[['tier', 'suit_efficiency', 'suit_risk']].sort_values('suit_efficiency', ascending=False).round(2))

print("\nTier Distribution:")
print("=" * 20)
print(scores_df['tier'].value_counts())
```

```markdown
**Explanation of Execution:**

Alex now has a clear classification for each potential automation project, indicating the most appropriate technological approach. This direct categorization helps him in initial resource planning and in guiding discussions with IT and engineering teams about the required infrastructure and expertise. For instance, tasks classified as 'Human Intervention Required' will prompt further investigation into necessary controls and human-in-the-loop processes, aligning with PGAM's ethical guidelines and risk management policies.
```

---

### 6. Visualizing the Automation Project Landscape

To effectively communicate his findings to PGAM's leadership and other stakeholders, Alex needs compelling visualizations. These plots will summarize the project evaluations, highlight key trade-offs, and present the prioritized roadmap clearly.

```markdown
# Markdown Cell - Story + Context + Real-World Relevance

Alex understands that visual representations are much more impactful than raw tables for strategic decision-making. He plans to generate several plots to illustrate:
1.  **Scorecard Heatmap:** A visual overview of how each task scores across all dimensions, quickly identifying patterns.
2.  **Impact vs. Risk Quadrant Plot:** To map projects based on their efficiency potential and risk level, categorizing them into strategic quadrants (e.g., Quick Wins, Caution Areas).
3.  **Suitability Ranking Bar Chart:** A direct comparison of projects based on their weighted suitability, color-coded by automation tier.
4.  **Radar Charts:** Detailed profiles for a few selected tasks, allowing for a deeper comparison across dimensions.

These visualizations will help stakeholders quickly grasp the key insights from the scorecard analysis and facilitate discussions on project prioritization.

```python
# Code cell (function definition + function execution)

# Define a color map for automation tiers for consistent visualization
tier_colors_map = {
    'Traditional Automation': 'green',
    'GenAI / LLM Automation': 'blue',
    'Hybrid (Traditional + GenAI)': 'orange',
    'Human Intervention Required': 'red'
}
scores_df['color'] = scores_df['tier'].map(tier_colors_map)

# --- Visualization 1: Scorecard Heatmap ---
plt.figure(figsize=(14, 8))
# Select only the score columns for the heatmap
heatmap_cols = [dim for dim in SCORECARD_DIMENSIONS.keys()]
sns.heatmap(scores_df[heatmap_cols], annot=True, fmt='.0f', cmap='viridis', linewidths=.5, linecolor='lightgray')
plt.title('Automation Scorecard Heatmap: Task Scores Across Dimensions', fontsize=16)
plt.xlabel('Score Dimensions', fontsize=12)
plt.ylabel('Candidate Projects', fontsize=12)
plt.tight_layout()
plt.show()

# --- Visualization 2: Impact vs. Risk Quadrant Plot ---
plt.figure(figsize=(12, 10))
for task_id, row in scores_df.iterrows():
    plt.scatter(row['efficiency_impact'], row['risk_level'],
                color=row['color'], s=300, edgecolors='black', alpha=0.8, label=row['tier'] if row['tier'] not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.annotate(task_id.split(':')[0], # Display short ID
                 (row['efficiency_impact'] + 0.05, row['risk_level'] + 0.1),
                 fontsize=9, weight='bold')

# Quadrant lines and labels
plt.axhline(y=2.5, color='gray', linestyle='--', alpha=0.6) # Mid-point for Risk
plt.axvline(x=3.0, color='gray', linestyle='--', alpha=0.6) # Mid-point for Efficiency Impact

plt.text(4.5, 1.2, 'Quick Wins', fontsize=12, color='green', fontweight='bold', ha='center', va='center')
plt.text(1.5, 1.2, 'Low Impact / Low Risk', fontsize=12, color='gray', fontweight='bold', ha='center', va='center')
plt.text(1.5, 4.0, 'CAUTION (High Risk)', fontsize=12, color='red', fontweight='bold', ha='center', va='center')
plt.text(4.5, 4.0, 'Strategic / High Risk', fontsize=12, color='purple', fontweight='bold', ha='center', va='center')

plt.xlabel('Efficiency Impact (1-5)', fontsize=12)
plt.ylabel('Risk Level (1-5)', fontsize=12)
plt.title('Automation Priority: Impact vs. Risk Quadrant Plot', fontsize=16)
plt.xticks(np.arange(1, 6, 1))
plt.yticks(np.arange(1, 6, 1))
plt.grid(True, linestyle=':', alpha=0.7)

# Create a custom legend to show unique tier labels and colors
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = list(dict.fromkeys(labels)) # Get unique labels in order of appearance
unique_handles = [handles[labels.index(ul)] for ul in unique_labels]
plt.legend(unique_handles, unique_labels, title="Automation Tier", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent legend overlap
plt.show()

# --- Visualization 3: Suitability Ranking Bar Chart (Efficiency-First) ---
plt.figure(figsize=(12, 8))
# Sort by 'suit_efficiency' for ranking
ranked_projects_eff = scores_df.sort_values('suit_efficiency', ascending=True) # Ascending for horizontal bar plot
sns.barplot(x='suit_efficiency', y=ranked_projects_eff.index, data=ranked_projects_eff,
            palette=ranked_projects_eff['color'].tolist(), hue='tier', dodge=False)

plt.xlabel('Weighted Suitability Score (Efficiency-First)', fontsize=12)
plt.ylabel('Candidate Projects', fontsize=12)
plt.title('Suitability Ranking: Projects by Efficiency-First Score', fontsize=16)
plt.legend(title="Automation Tier", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# --- Visualization 4: Radar Charts for 2-3 Selected Tasks ---
# Select a few diverse tasks for radar charts
selected_tasks = ['T1: Data Ingestion', 'T2: 10-K Risk Extract', 'T8: Thesis Drafting']
# Ensure selected_tasks are in the scores_df index
selected_tasks = [task for task in selected_tasks if task in scores_df.index]

if len(selected_tasks) > 0:
    num_vars = len(heatmap_cols)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1] # Complete the circle

    fig, axes = plt.subplots(1, len(selected_tasks), figsize=(5 * len(selected_tasks), 6), subplot_kw=dict(polar=True))
    if len(selected_tasks) == 1: # Handle single subplot case
        axes = [axes]

    for i, task_id in enumerate(selected_tasks):
        values = scores_df.loc[task_id, heatmap_cols].tolist()
        values += values[:1] # Complete the circle for plotting

        ax = axes[i]
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_yticklabels([]) # Hide radial labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(heatmap_cols, fontsize=9, rotation=45, ha='right')

        # Set max value for radial axis (5 for most, 1 for human_oversight)
        # Assuming all dimensions are roughly 1-5 scale, adjust if human_oversight needs separate scaling
        # For simplicity, we'll plot human_oversight on 1-5 scale as well for visual consistency, or rescale.
        # Let's scale human_oversight (0/1) to (1/5) for radar chart if needed, or set radial limits.
        # For this example, let's keep original scale and set radial limits to max_score
        max_score = 5
        ax.set_rlim(0, max_score) # Radial limits up to 5

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=task_id, color=scores_df.loc[task_id, 'color'])
        ax.fill(angles, values, color=scores_df.loc[task_id, 'color'], alpha=0.25)
        ax.set_title(f"Profile: {task_id.split(':')[0]} ({scores_df.loc[task_id, 'tier']})", va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Add placeholders for Roadmap Timeline (conceptual Gantt-style chart) - will be implemented in Section 8
# (See Section 8 for its implementation in the roadmap table)
```

```markdown
**Explanation of Execution:**

These visualizations provide Alex and PGAM's leadership with a comprehensive and intuitive overview of the automation landscape. The **Scorecard Heatmap** quickly reveals which tasks are complex or structured. The **Impact vs. Risk Quadrant Plot** immediately highlights potential "Quick Wins" (high impact, low risk) and "Caution" areas (high risk), enabling strategic resource allocation. The **Suitability Ranking Bar Chart** offers a clear prioritization list, while the **Radar Charts** allow for detailed task-level analysis, showing the unique profile of each project across the evaluation dimensions. This visual storytelling helps to foster better understanding and consensus among diverse stakeholders.
```

---

### 7. Deep Dive: Hybrid Routing & ROI for the Top Project

For complex tasks, Alex knows that pure automation might not be feasible or desirable. He needs to consider a "Hybrid Routing" approach, breaking down complex tasks into sub-tasks suitable for different automation tiers. Additionally, he must justify the top-ranked project with a solid business case, including ROI estimation.

```markdown
# Markdown Cell - Story + Context + Real-World Relevance

Alex recognizes that many investment workflows are not monolithic; they consist of multiple sub-tasks. For tasks classified as 'Hybrid' or even 'GenAI/LLM Automation', a sophisticated approach might involve decomposing them into sub-tasks that can be handled by Traditional Automation, GenAI/LLM, or Human Intervention. This "Hybrid Routing" maximizes automation potential while maintaining human oversight where critical.

Simultaneously, to secure budget and buy-in for the top-ranked project, Alex must present a compelling business case, including a clear Return on Investment (ROI) estimation. The ROI calculation is a critical metric for financial professionals, directly linking automation efforts to financial value.

The ROI formula for the top candidate project is:
$$ROI = \frac{\text{Annual Hours Saved} \cdot \text{Cost_per_Hour}}{\text{Development_Cost} + \text{Annual_Operating_Cost}} - 1$$
Where:
*   **Annual Hours Saved:** The total hours saved per year by automating the task.
*   **Cost_per_Hour:** The average cost of human labor per hour for the task.
*   **Development_Cost:** One-time costs to develop and implement the automation.
*   **Annual_Operating_Cost:** Recurring costs for maintaining the automation (e.g., infrastructure, licenses).

```python
# Code cell (function definition + function execution)

# --- Hybrid Routing Table for a Complex Task (e.g., 'T4: Monthly Performance Report') ---
# This demonstrates conceptually how a complex task can be broken down.
# Let's assume 'T4: Monthly Performance Report' was classified as 'Hybrid' or 'GenAI/LLM'
# from our previous step (it was 'Hybrid (Traditional + GenAI)').

# Define sub-tasks for 'T4: Monthly Performance Report' and their recommended tiers/times
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
        'GenAI / LLM Automation', # Could also be Hybrid depending on customization level
        'Human Intervention Required'
    ],
    'Time (Manual, min)': [15, 30, 20, 45, 20, 15], # Estimated manual time for each sub-task
    'Time (Auto, min)':   [0.17, 0.08, 0.17, 0.5, 0.25, 10] # Estimated automated/human intervention time in minutes
                                                            # 10s = 0.17 min, 5s = 0.08 min, 30s = 0.5 min, 15s = 0.25 min
}
hybrid_routing_df = pd.DataFrame(hybrid_routing_data)

print("\nConceptual Hybrid Routing Table for 'T4: Monthly Performance Report':")
print("=" * 80)
print(hybrid_routing_df.to_string(index=False))

total_manual_time = hybrid_routing_df['Time (Manual, min)'].sum()
total_hybrid_time = hybrid_routing_df['Time (Auto, min)'].sum()
time_reduction_percent = (1 - (total_hybrid_time / total_manual_time)) * 100

print(f"\nTotal manual time for T4: {total_manual_time:.2f} minutes")
print(f"Total hybrid-routed time for T4: {total_hybrid_time:.2f} minutes")
print(f"Estimated time reduction: {time_reduction_percent:.2f}%")


# --- ROI Estimation for the Top-Ranked Project (based on 'Efficiency-First') ---
# Select the top project based on efficiency suitability
top_project_id = scores_df.sort_values('suit_efficiency', ascending=False).index[0]
top_project_data = scores_df.loc[top_project_id]

# Conceptual values for ROI calculation (example based on T1: Data Ingestion from PDF)
# These would be gathered from operations and finance teams in a real scenario
annual_hours_saved = 96  # Example: 8 hours/month * 12 months = 96 hours
cost_per_hour = 150      # Example: Average burdened cost of an analyst/associate
development_cost = 1200  # Example: One-time development cost for automation script/tool
annual_operating_cost = 100 # Example: Annual maintenance, monitoring, infra costs

# Calculate ROI using the formula
roi = (annual_hours_saved * cost_per_hour) / (development_cost + annual_operating_cost) - 1

print(f"\nBusiness Case Summary for Top Ranked Project: '{top_project_id}'")
print("=" * 80)
print(f"Automation Tier: {top_project_data['tier']}")
print(f"Efficiency-First Suitability Score: {top_project_data['suit_efficiency']:.2f}")
print("\nEstimated Financial Impact:")
print(f"  Annual Hours Saved: {annual_hours_saved} hours")
print(f"  Cost per Hour: ${cost_per_hour:.2f}")
print(f"  Estimated Annual Savings: ${annual_hours_saved * cost_per_hour:.2f}")
print(f"  Development Cost: ${development_cost:.2f}")
print(f"  Annual Operating Cost: ${annual_operating_cost:.2f}")
print(f"  Calculated ROI: {roi:.2%} (or {roi*100:.2f}x return)")

# Conceptual timeline and risk mitigation
print("\nConceptual Timeline & Risk Mitigation:")
print("  Timeline: Estimated 4-6 weeks for development and initial rollout.")
print("  Risk Mitigation: Implement robust data validation checks, phased rollout, and human-in-the-loop review for exceptions.")
```

```markdown
**Explanation of Execution:**

The **Hybrid Routing Table** clearly demonstrates to PGAM's technical teams how a seemingly complex task like a performance report can be efficiently broken down, leveraging different AI/ML capabilities while retaining human judgment for critical review. This optimizes automation efforts and ensures scalability.

The **ROI Estimation** provides a clear financial justification for pursuing the top-ranked project. By quantifying the financial benefits against costs, Alex can make a compelling case to the firm's management, highlighting the tangible value of strategic automation. This directly supports resource allocation decisions and demonstrates the practical application of his analysis in driving business value.
```

---

### 8. Prioritized Automation Roadmap and Conclusion

Alex consolidates all findings into a comprehensive automation roadmap, categorizing projects into 'Quick Wins', 'Medium-Term', and 'Strategic' initiatives. This final output provides PGAM with a clear, actionable plan for its AI/ML automation journey.

```markdown
# Markdown Cell - Story + Context + Real-World Relevance

Alex's ultimate goal is to present a clear, actionable automation roadmap to PGAM's executive committee. This roadmap summarizes all the analysis performed: the scores, suitability rankings, automation tiers, and a conceptual timeline. Categorizing projects by effort/timeline (Quick Wins, Medium-Term, Strategic) helps manage expectations and sequence implementation, ensuring a phased approach to AI adoption.

This roadmap serves as a strategic document, guiding the firm's investment in AI/ML and fostering a common understanding across different departments. It transforms raw data and complex analysis into a digestible, decision-driving output.

```python
# Code cell (function definition + function execution)

# Add conceptual 'hours_saved_month' and 'effort' categories for roadmap timeline
# These values are illustrative and would be estimated in a real-world scenario.
# Align with the PDF example values where possible.
hours_saved_month_map = {
    'T1: Data Ingestion': 8,
    'T2: 10-K Risk Extract': 12,
    'T3: Earnings Summary': 10,
    'T4: Performance Report': 6,
    'T5: Compliance Check': 4,
    'T6: Credit Recalib': 3,
    'T7: Rebalancing': 5,
    'T8: Thesis Drafting': 8
}
# Extend for custom tasks (assigning default/plausible values)
for task_id in scores_df.index:
    if task_id not in hours_saved_month_map:
        hours_saved_month_map[task_id] = 7 # Default for custom tasks

effort_map = {
    'T1: Data Ingestion': 'Low',
    'T2: 10-K Risk Extract': 'Med',
    'T3: Earnings Summary': 'Med',
    'T4: Performance Report': 'High', # Adjusted from PDF 'Med' to 'High' for more diversity
    'T5: Compliance Check': 'Low', # Adjusted from PDF 'Med' to 'Low' for more diversity
    'T6: Credit Recalib': 'High',
    'T7: Rebalancing': 'Med',
    'T8: Thesis Drafting': 'High'
}
# Extend for custom tasks (assigning default/plausible values)
for task_id in scores_df.index:
    if task_id not in effort_map:
        effort_map[task_id] = 'Med' # Default for custom tasks

scores_df['hours_saved_month'] = scores_df.index.map(hours_saved_month_map)
scores_df['effort'] = scores_df.index.map(effort_map)

# Function to assign timeline categories based on effort
def categorize_timeline(row):
    if row['effort'] == 'Low':
        return 'Quick Win (<1 week)'
    elif row['effort'] == 'Med':
        return 'Medium-Term (1-4 weeks)'
    else: # 'High' effort
        return 'Strategic (1-3 months)'

scores_df['timeline'] = scores_df.apply(categorize_timeline, axis=1)

# Sort the roadmap by efficiency suitability
final_roadmap = scores_df.sort_values('suit_efficiency', ascending=False)

print("\n--- Prioritized Automation Roadmap ---")
print("=" * 100)
# Display a formatted table with key information
print(f"{'Timeline':<25s} | {'Project':<30s} | {'Hours Saved/Month':<20s} | {'Automation Tier':<25s}")
print("-" * 100)
for task, row in final_roadmap.iterrows():
    print(f"{row['timeline']:<25s} | {task:<30s} | {row['hours_saved_month']:<20} | {row['tier']:<25s}")
print("=" * 100)

print("\n--- Final Recommendations ---")
print("Alex has successfully applied the CFA Institute Automation Scorecard framework to prioritize AI/ML automation projects at Prosperity Global Asset Management.")
print("The analysis provides:")
print("1. A clear ranking of projects based on weighted suitability, allowing for strategic alignment.")
print("2. Classification into appropriate automation tiers, guiding technology selection.")
print("3. Visualizations that facilitate stakeholder communication and highlight key trade-offs.")
print("4. A conceptual ROI for top projects, enabling data-driven investment decisions.")
print("\nThis structured approach moves PGAM beyond ad-hoc automation, ensuring that future AI/ML investments target high-value, feasible, and risk-managed opportunities.")

```
```markdown
**Explanation of Execution:**

This final roadmap is Alex's central deliverable. It provides PGAM's leadership with a clear, concise, and actionable plan. Projects are categorized to manage expectations regarding development timelines and required effort, fostering a realistic approach to AI adoption. The roadmap ensures that PGAM's automation efforts are strategically aligned, financially justifiable, and systematically managed, moving the firm closer to its goals of operational efficiency and innovation while mitigating risks. This completes Alex's objective of moving from ad-hoc decisions to a structured, data-driven framework for AI/ML project prioritization.
```
