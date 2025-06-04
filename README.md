📊 Clinical Trial Statistical Analysis App
Automated statistical analysis for clinical trials - no statistics knowledge required!
Transform your raw clinical trial data into professional statistical reports with just a few clicks. This Streamlit application automatically detects study designs, runs appropriate statistical tests, and generates publication-ready results.
Show Image
Show Image
Show Image
🎯 Features
🔍 Smart Study Design Detection

Automatic RCT Detection: Identifies randomized controlled trials
Cohort Study Support: Handles longitudinal observational studies
Cross-sectional Analysis: Processes survey and questionnaire data
Case-Control Studies: Manages matched case-control designs

📊 Professional Statistical Analysis

Baseline Comparisons: Chi-square tests for categorical, t-tests for continuous variables
Primary Efficacy Analysis: ANCOVA with baseline adjustment, independent t-tests
Secondary Analyses: Response rate comparisons, effect size calculations
Missing Data Handling: Automatic detection and reporting

📚 Smart Excel File Handling

Multi-Sheet Support: Automatically processes complex Excel workbooks
Context Extraction: Uses study descriptions and variable definitions
Enhanced Analysis: Leverages metadata for smarter variable classification
Flexible Input: Supports both single-sheet and multi-sheet formats

🤖 AI-Enhanced Analysis (Optional)

Claude AI Integration: Optional enhanced analysis with Anthropic's Claude
Intelligent Variable Classification: Context-aware outcome detection
Smart Recommendations: AI-guided analysis approach selection

📈 Rich Visualizations

Interactive Plots: Plotly-powered charts and graphs
Baseline Distributions: Compare characteristics between groups
Outcome Visualizations: Box plots, histograms, and trend analysis
Professional Quality: Publication-ready figures

📄 Comprehensive Reporting

Statistical Reports: Downloadable markdown reports
JSON Export: Machine-readable results for further analysis
Professional Format: Following clinical trial reporting standards
Variable Descriptions: Enhanced context in results

🚀 Quick Start
Installation

Clone the repository:

bashgit clone https://github.com/yourusername/clinical-trial-analysis.git
cd clinical-trial-analysis

Install required packages:

bashpip install streamlit pandas numpy plotly scipy statsmodels matplotlib openpyxl xlrd

Run the application:

bashstreamlit run clinical_stats_app.py

Open your browser to http://localhost:8501

Basic Usage

Upload your data file (.xlsx recommended, .xls and .csv supported)
Configure analysis options in the sidebar
Review automated results including study design detection
Download reports and visualizations

📁 Supported File Formats
✅ Recommended: Excel (.xlsx) Files
Single Sheet Format:

Row 1: Variable headers
Rows 2+: Patient data
Clear variable naming (e.g., 'group', 'age', 'outcome_baseline')

Multi-Sheet Format (Best Results):

Sheet 1: Study description and background information
Sheet 2: Variable descriptions and codebook
Sheet 3: Actual patient data (named "Data")

⚠️ Also Supported:

Excel (.xls): Older format, requires xlrd package
CSV (.csv): Simple comma-separated format

📋 Example Data Structure
csvid,age,sex,group,pk1_severity,pk2_severity,response
1,45,1,0,8.5,6.2,1
2,52,0,1,7.2,4.1,1
3,38,1,0,9.1,8.9,0
4,61,0,1,6.8,3.5,1
🔧 Analysis Capabilities
Study Types Automatically Detected:

Randomized Controlled Trials (RCTs)
Cohort Studies
Cross-sectional Studies
Case-Control Studies

Statistical Tests Performed:

Baseline Comparisons: Chi-square, t-tests, Mann-Whitney U
Primary Analysis: ANCOVA, independent t-tests
Effect Sizes: Cohen's d, treatment effects
Response Rates: Contingency table analysis

Variable Types Recognized:

Demographics: age, sex, education, etc.
Treatment Groups: group, treatment, arm, allocation
Outcomes: severity scores, quality of life measures
Time Variables: baseline, follow-up, longitudinal measures

🤖 AI Enhancement Setup (Optional)
To enable AI-enhanced analysis with Claude:

Get an API key from Anthropic
Enable AI Enhancement in the sidebar
Enter your API key in the secure field
Enjoy smarter analysis with enhanced variable detection and insights

Note: AI features require an active Anthropic API subscription
📊 Example Output
The app generates comprehensive statistical reports including:

Study Overview: Detected design, sample size, confidence scores
Baseline Characteristics: Group comparisons with statistical tests
Primary Analysis Results: Treatment effects, p-values, confidence intervals
Visualizations: Interactive charts showing group differences
Professional Report: Downloadable statistical summary

🛠️ Technical Requirements
Python Version

Python 3.8 or higher

Key Dependencies
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.0.0
scipy>=1.9.0
statsmodels>=0.13.0
matplotlib>=3.5.0
openpyxl>=3.0.0
xlrd>=2.0.1
System Requirements

4GB RAM minimum (8GB recommended for large datasets)
Modern web browser (Chrome, Firefox, Safari, Edge)

🎯 Use Cases
Perfect For:

Clinical Researchers: Analyze trial data without statistical software expertise
Medical Students: Learn statistical analysis through automated examples
Regulatory Submissions: Generate professional statistical reports
Pilot Studies: Quick analysis of preliminary data
Academic Research: Publication-ready statistical analysis

Typical Datasets:

Drug Trials: Efficacy and safety comparisons
Medical Device Studies: Before/after comparisons
Quality of Life Studies: Patient-reported outcome measures
Behavioral Interventions: Treatment vs. control comparisons
Diagnostic Studies: Sensitivity and specificity analysis

🔒 Data Privacy

Local Processing: All analysis runs locally on your machine
No Data Storage: Files are processed in memory only
Secure: No data transmitted except optional AI API calls
HIPAA Considerate: Designed with healthcare data privacy in mind

🐛 Troubleshooting
Common Issues:
"Missing dependency 'xlrd'" Error:
bashpip install xlrd
Excel file not reading properly:

Save as .xlsx format for best compatibility
Ensure data is in a sheet named "Data" for multi-sheet files
Check that headers are in the first row

No group variable detected:

Ensure treatment group column is named clearly (e.g., 'group', 'treatment')
Check that group variable has exactly 2 unique values
Verify data types are correct (numeric for continuous, text for categorical)

Low confidence warnings:

Review variable names for clarity
Add variable descriptions in a separate sheet
Ensure standard clinical trial data structure
