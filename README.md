# QuLab: Lab 12: Automation Project Evaluation

## Automation Project Evaluation: A CFA Institute Scorecard Approach for Investment Professionals

### Case Study: Systematically Prioritizing AI/ML Projects at Prosperity Global Asset Management

This Streamlit application provides a comprehensive framework for evaluating and prioritizing potential AI/ML automation initiatives within an investment firm, inspired by the CFA Institute's Automation Scorecard. It guides users through a structured workflow to identify high-value, feasible, and low-risk automation opportunities, ensuring strategic alignment and maximizing return on investment.

The application addresses a common challenge faced by investment operations managers like Alex Chen, CFA, at Prosperity Global Asset Management (PGAM), who needs to move beyond ad-hoc automation efforts and implement a systematic approach. The goal is to build a clear, prioritized automation roadmap that resonates with various stakeholders across portfolio management, compliance, and IT.

## ✨ Features

This application offers a guided, step-by-step workflow for automation project evaluation:

1.  **Define Dimensions**: Clearly define the 6 key dimensions of the Automation Scorecard (e.g., Efficiency Impact, Data Structure, Risk Level) and their scoring scales.
2.  **Score Projects**: Input and edit scores for candidate automation tasks across the defined dimensions using an interactive data editor. Users can add new custom tasks dynamically. A heatmap visualization provides an immediate overview of scores.
3.  **Prioritize with Weights**: Configure different strategic weighting profiles (e.g., "Efficiency-First" vs. "Risk-First") to compute weighted suitability scores using Multi-Criteria Decision Analysis (MCDA). This allows for sensitivity analysis and understanding how project rankings shift based on strategic focus.
4.  **Classify Tiers**: Automatically classify projects into appropriate automation tiers (Traditional Automation, GenAI / LLM Automation, Human Intervention Required, Hybrid) based on rule-based logic derived from project scores.
5.  **Visualize Landscape**: Generate compelling visualizations to summarize findings:
    *   **Impact vs. Risk Quadrant Plot**: Maps projects based on efficiency potential and risk, identifying "Quick Wins" and "Caution Areas".
    *   **Suitability Ranking Bar Chart**: Ranks projects by their weighted suitability score, color-coded by automation tier.
    *   **Radar Charts**: Provides detailed score profiles for selected projects, allowing for comparative analysis across all dimensions.
    *   *(Conceptual Sensitivity Tornado)*: A conceptual discussion on perturbing individual dimension weights.
6.  **Deep Dive & ROI**: Explore advanced concepts like "Hybrid Routing" (breaking down complex tasks into sub-tasks suitable for different automation types) and perform a Return on Investment (ROI) estimation for the top-ranked project, allowing adjustment of financial parameters.
7.  **Automation Roadmap**: Consolidate all analysis into a prioritized automation roadmap, categorizing projects by conceptual timeline and presenting final recommendations for actionable implementation.

## 🚀 Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository** (if this project is hosted on GitHub/GitLab):
    ```bash
    git clone https://github.com/your-username/quslab-automation-evaluation.git
    cd quslab-automation-evaluation
    ```
    *(If not a repository, create a folder and place `app.py` and `source.py` inside it.)*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages**:
    Create a `requirements.txt` file in your project directory with the following content:
    ```
    streamlit==1.x.x # Use a compatible version, e.g., 1.30.0
    pandas==2.x.x
    numpy==1.x.x
    matplotlib==3.x.x
    seaborn==0.x.x
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♀️ Usage

1.  **Ensure `app.py` (containing the provided Streamlit code) and `source.py` are in the same directory.**

2.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

3.  Your default web browser should automatically open a new tab displaying the application. If not, open your browser and navigate to `http://localhost:8501`.

4.  **Navigate and Interact**:
    *   Use the **"Automation Scorecard Navigator"** in the sidebar to switch between different sections of the evaluation workflow.
    *   Interact with the `st.data_editor` to modify project scores, add new projects using the form, adjust weighting sliders, and experiment with ROI parameters.
    *   Observe how the visualizations and suitability rankings update in real-time based on your inputs.

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application script
├── source.py               # Contains data, helper functions, and initial configurations
├── requirements.txt        # List of Python dependencies
└── README.md               # This file
```

*   `app.py`: This is the main script that runs the Streamlit application. It handles page configuration, session state management, UI layout, user interactions, and calls functions from `source.py`.
*   `source.py`: This file acts as a backend for the application. It defines initial data (e.g., project scores, weights, dimension descriptions), helper functions for calculations (e.g., `calculate_weighted_score`, `classify_tier`, `categorize_timeline`), and static configurations.

## 💻 Technology Stack

*   **Framework**: [Streamlit](https://streamlit.io/)
*   **Programming Language**: Python 3
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: A `LICENSE` file would need to be created in the project root.)*

## 📧 Contact

For any questions or inquiries, please contact:

*   **Quant University**
*   **Website**: [www.quantuniversity.com](https://www.quantuniversity.com/)
*   **Email**: info@quantuniversity.com

---
*Created with ❤️ by Quant University for educational purposes.*


## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
