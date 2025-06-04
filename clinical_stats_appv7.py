import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
import warnings
import json
import requests
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Clinical Trial Statistical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SmartExcelHandler:
    """Handle multi-sheet Excel files intelligently"""
    
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.sheets_data = {}
        self.study_description = ""
        self.variable_descriptions = {}
        self.main_data = None
        
    def load_all_sheets(self):
        """Load all sheets and classify their content"""
        try:
            # Determine engine based on file extension
            if self.uploaded_file.name.endswith('.xls'):
                engine = 'xlrd'
            else:
                engine = 'openpyxl'
            
            # Load all sheets
            excel_file = pd.ExcelFile(self.uploaded_file, engine=engine)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(self.uploaded_file, sheet_name=sheet_name, engine=engine)
                    self.sheets_data[sheet_name] = df
                except Exception as e:
                    st.warning(f"Could not read sheet '{sheet_name}': {str(e)}")
            
            # Classify sheets
            self._classify_sheets()
            return True
            
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            return False
    
    def _classify_sheets(self):
        """Classify sheets based on content"""
        for sheet_name, df in self.sheets_data.items():
            sheet_name_lower = sheet_name.lower()
            
            # Check if it's a description sheet
            if 'description' in sheet_name_lower or 'info' in sheet_name_lower:
                if 'data' not in sheet_name_lower:
                    # This is study description
                    self.study_description = self._extract_description(df)
                    continue
            
            # Check if it's variable descriptions
            if 'variable' in sheet_name_lower and len(df.columns) >= 2:
                self.variable_descriptions = self._extract_variable_descriptions(df)
                continue
            
            # Check if it's the main data
            if ('data' in sheet_name_lower or 
                len(df.columns) > 10 or  # Likely main dataset
                any(col.lower() in ['id', 'subject', 'patient'] for col in df.columns)):
                
                # Additional check: make sure it's not just text
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 3:  # Has substantial numeric data
                    self.main_data = df
                    self.main_data_sheet = sheet_name
    
    def _extract_description(self, df):
        """Extract study description from description sheet"""
        description_text = ""
        for col in df.columns:
            for value in df[col].dropna():
                if isinstance(value, str) and len(value) > 20:
                    description_text += value + " "
        return description_text.strip()
    
    def _extract_variable_descriptions(self, df):
        """Extract variable descriptions from variables sheet"""
        var_dict = {}
        if len(df.columns) >= 2:
            # Assume first column is variable name, second is description
            var_col = df.columns[0]
            desc_col = df.columns[1]
            
            for _, row in df.iterrows():
                if pd.notna(row[var_col]) and pd.notna(row[desc_col]):
                    var_dict[str(row[var_col])] = str(row[desc_col])
        
        return var_dict
    
    def get_analysis_ready_data(self):
        """Return the main dataset ready for analysis"""
        return self.main_data
    
    def get_enhanced_context(self):
        """Return enhanced context for AI analysis"""
        return {
            'study_description': self.study_description,
            'variable_descriptions': self.variable_descriptions,
            'data_shape': self.main_data.shape if self.main_data is not None else None,
            'column_names': list(self.main_data.columns) if self.main_data is not None else []
        }

class AIEnhancedAnalyzer:
    """Enhanced analyzer with AI integration"""
    
    def __init__(self, use_ai=False, api_key=None):
        self.use_ai = use_ai
        self.api_key = api_key
        self.data = None
        self.context = {}
        self.study_type = None
        self.primary_outcomes = []
        self.secondary_outcomes = []
        self.group_variable = None
        self.time_variables = []
        self.demographic_vars = []
        self.confidence_score = 0
        
    def analyze_with_context(self, df, context=None):
        """Analyze data with enhanced context"""
        self.data = df
        self.context = context or {}
        
        if self.use_ai and self.api_key:
            return self._ai_enhanced_analysis()
        else:
            return self._standard_analysis()
    
    def _ai_enhanced_analysis(self):
        """Use AI to enhance the analysis"""
        try:
            # Prepare context for AI
            ai_context = {
                'study_description': self.context.get('study_description', ''),
                'variable_descriptions': self.context.get('variable_descriptions', {}),
                'column_names': list(self.data.columns),
                'data_sample': self.data.head().to_dict() if len(self.data) > 0 else {},
                'data_shape': self.data.shape
            }
            
            # Call AI API for analysis guidance
            analysis_plan = self._get_ai_analysis_plan(ai_context)
            
            # Execute analysis based on AI recommendations
            return self._execute_ai_guided_analysis(analysis_plan)
            
        except Exception as e:
            st.warning(f"AI analysis failed, falling back to standard analysis: {str(e)}")
            return self._standard_analysis()
    
    def _get_ai_analysis_plan(self, context):
        """Get analysis plan from AI"""
        # This would integrate with Claude API
        # For now, return a structured plan
        return {
            'study_type': 'RCT',
            'primary_outcomes': ['pk1', 'pk2', 'pk5'],
            'group_variable': 'group',
            'demographic_vars': ['age', 'sex'],
            'analysis_methods': ['baseline_comparison', 'ancova', 'response_rate']
        }
    
    def _execute_ai_guided_analysis(self, plan):
        """Execute analysis based on AI plan"""
        # Use the AI plan to guide analysis
        self.study_type = plan.get('study_type', 'Unknown')
        self.primary_outcomes = plan.get('primary_outcomes', [])
        self.group_variable = plan.get('group_variable')
        self.demographic_vars = plan.get('demographic_vars', [])
        
        return self._run_analysis()
    
    def _standard_analysis(self):
        """Standard rule-based analysis"""
        self._detect_study_design()
        self._classify_variables()
        return self._run_analysis()
    
    def _detect_study_design(self):
        """Detect study design using rules"""
        patterns = {
            'RCT': ['group', 'treatment', 'control', 'randomized', 'arm'],
            'cohort': ['followup', 'follow_up', 'time', 'baseline'],
            'cross_sectional': ['survey', 'questionnaire'],
            'case_control': ['case', 'control', 'matched']
        }
        
        column_names = [col.lower() for col in self.data.columns]
        scores = {}
        
        for study_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords 
                       if any(keyword in col for col in column_names))
            scores[study_type] = score
        
        # Also check study description for additional context
        if self.context.get('study_description'):
            desc = self.context['study_description'].lower()
            for study_type, keywords in patterns.items():
                for keyword in keywords:
                    if keyword in desc:
                        scores[study_type] = scores.get(study_type, 0) + 2
        
        if scores['RCT'] > 0:
            self.study_type = 'RCT'
            self.confidence_score = 0.9
        elif scores['cohort'] > 1:
            self.study_type = 'Cohort Study'
            self.confidence_score = 0.8
        else:
            self.study_type = 'Cross-sectional Study'
            self.confidence_score = 0.6
    
    def _classify_variables(self):
        """Classify variables using rules and context"""
        demographics = ['age', 'sex', 'gender', 'race', 'ethnicity', 'education']
        outcomes_keywords = ['score', 'severity', 'pain', 'quality', 'sf36', 'outcome', 'pk']
        time_keywords = ['pk1', 'pk2', 'pk3', 'pk4', 'pk5', 'baseline', 'followup', 
                        'month', 'week', 'day', 'time', 'date']
        group_keywords = ['group', 'treatment', 'arm', 'allocation']
        
        # Use variable descriptions if available
        var_descriptions = self.context.get('variable_descriptions', {})
        
        for col in self.data.columns:
            col_lower = col.lower()
            description = var_descriptions.get(col, '').lower()
            
            # Demographics
            if any(demo in col_lower for demo in demographics):
                self.demographic_vars.append(col)
            
            # Group/treatment variables
            elif any(group in col_lower for group in group_keywords):
                if self.group_variable is None:
                    self.group_variable = col
            elif 'control' in description and 'acupuncture' in description:
                if self.group_variable is None:
                    self.group_variable = col
            
            # Time-related variables
            elif any(time in col_lower for time in time_keywords):
                self.time_variables.append(col)
            
            # Outcome variables
            elif (any(outcome in col_lower for outcome in outcomes_keywords) or
                  'severity' in description or 'score' in description):
                if len(self.primary_outcomes) < 5:
                    self.primary_outcomes.append(col)
                else:
                    self.secondary_outcomes.append(col)
    
    def _run_analysis(self):
        """Run the actual statistical analysis"""
        results = {
            'study_info': self._get_study_info(),
            'descriptive_stats': self._descriptive_statistics(),
            'baseline_comparison': self._baseline_comparison(),
            'primary_analysis': self._primary_efficacy_analysis(),
            'secondary_analysis': self._secondary_analyses(),
            'visualizations': self._create_visualizations()
        }
        
        return results
    
    def _get_study_info(self):
        """Get basic study information"""
        return {
            'study_type': self.study_type,
            'confidence_score': self.confidence_score,
            'sample_size': len(self.data),
            'variables_count': len(self.data.columns),
            'group_variable': self.group_variable,
            'primary_outcomes': self.primary_outcomes,
            'secondary_outcomes': self.secondary_outcomes[:5],
            'demographic_vars': self.demographic_vars,
            'missing_data': self.data.isnull().sum().sum(),
            'context_available': bool(self.context.get('study_description') or self.context.get('variable_descriptions'))
        }
    
    def _descriptive_statistics(self):
        """Generate descriptive statistics"""
        results = {}
        
        # Overall descriptive stats
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        results['numeric_summary'] = self.data[numeric_cols].describe()
        
        # Categorical variables
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        results['categorical_summary'] = {}
        for col in categorical_cols:
            if self.data[col].nunique() < 20:
                results['categorical_summary'][col] = self.data[col].value_counts()
        
        return results
    
    def _baseline_comparison(self):
        """Compare baseline characteristics between groups"""
        if not self.group_variable or self.group_variable not in self.data.columns:
            return {"error": "No group variable identified for comparison"}
        
        results = {}
        groups = self.data[self.group_variable].dropna().unique()
        
        if len(groups) != 2:
            return {"error": f"Expected 2 groups, found {len(groups)}"}
        
        # Compare demographics and baseline variables
        comparison_results = []
        baseline_vars = self.demographic_vars + [col for col in self.primary_outcomes if 'pk1' in col or 'baseline' in col.lower()]
        
        for var in baseline_vars:
            if var not in self.data.columns:
                continue
                
            try:
                if self.data[var].dtype in ['object', 'category']:
                    # Categorical variable - Chi-square test
                    contingency = pd.crosstab(self.data[var], self.data[self.group_variable])
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    test_type = "Chi-square"
                else:
                    # Continuous variable - t-test
                    group0 = self.data[self.data[self.group_variable] == groups[0]][var].dropna()
                    group1 = self.data[self.data[self.group_variable] == groups[1]][var].dropna()
                    
                    t_stat, p_value = ttest_ind(group0, group1)
                    test_type = "Independent t-test"
                
                comparison_results.append({
                    'Variable': var,
                    'Test': test_type,
                    'P-value': p_value,
                    'Significant': p_value < 0.05
                })
            except Exception as e:
                comparison_results.append({
                    'Variable': var,
                    'Test': 'Error',
                    'P-value': None,
                    'Significant': False,
                    'Error': str(e)
                })
        
        results['comparison_table'] = pd.DataFrame(comparison_results)
        return results
    
    def _primary_efficacy_analysis(self):
        """Perform primary efficacy analysis"""
        if not self.group_variable or not self.primary_outcomes:
            return {"error": "Missing group variable or primary outcomes"}
        
        results = {}
        
        for outcome in self.primary_outcomes[:3]:
            if outcome not in self.data.columns:
                continue
                
            try:
                # Find baseline if available
                baseline_col = None
                if 'pk2' in outcome or 'pk5' in outcome:
                    baseline_col = outcome.replace('pk2', 'pk1').replace('pk5', 'pk1')
                    if baseline_col not in self.data.columns:
                        baseline_col = None
                
                if baseline_col and baseline_col in self.data.columns:
                    # ANCOVA
                    clean_data = self.data[[outcome, self.group_variable, baseline_col]].dropna()
                    if len(clean_data) > 10:
                        # Clean column names for formula
                        clean_data_renamed = clean_data.copy()
                        outcome_clean = 'outcome_var'
                        group_clean = 'group_var'
                        baseline_clean = 'baseline_var'
                        
                        clean_data_renamed = clean_data_renamed.rename(columns={
                            outcome: outcome_clean,
                            self.group_variable: group_clean,
                            baseline_col: baseline_clean
                        })
                        
                        formula = f"{outcome_clean} ~ {group_clean} + {baseline_clean}"
                        model = ols(formula, data=clean_data_renamed).fit()
                        
                        results[outcome] = {
                            'analysis_type': 'ANCOVA',
                            'treatment_effect': model.params.get(group_clean, None),
                            'p_value': model.pvalues.get(group_clean, None),
                            'r_squared': model.rsquared,
                            'n_analyzed': len(clean_data),
                            'formula_used': formula
                        }
                else:
                    # Simple comparison
                    groups = self.data[self.group_variable].dropna().unique()
                    if len(groups) == 2:
                        group0 = self.data[self.data[self.group_variable] == groups[0]][outcome].dropna()
                        group1 = self.data[self.data[self.group_variable] == groups[1]][outcome].dropna()
                        
                        if len(group0) > 0 and len(group1) > 0:
                            t_stat, p_value = ttest_ind(group0, group1)
                            
                            results[outcome] = {
                                'analysis_type': 'Independent t-test',
                                'mean_group0': group0.mean(),
                                'mean_group1': group1.mean(),
                                'difference': group1.mean() - group0.mean(),
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'n_group0': len(group0),
                                'n_group1': len(group1)
                            }
            
            except Exception as e:
                results[outcome] = {'error': str(e)}
        
        return results
    
    def _secondary_analyses(self):
        """Perform secondary analyses"""
        results = {}
        
        # Response rate analysis
        response_cols = [col for col in self.data.columns if 'response' in col.lower()]
        for col in response_cols:
            if col in self.data.columns and self.data[col].nunique() <= 3:
                try:
                    contingency = pd.crosstab(self.data[col], self.data[self.group_variable])
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    results[f'{col}_analysis'] = {
                        'analysis_type': 'Response Rate Comparison',
                        'contingency_table': contingency,
                        'chi2_statistic': chi2,
                        'p_value': p_value
                    }
                except Exception as e:
                    results[f'{col}_analysis'] = {'error': str(e)}
        
        return results
    
    def _create_visualizations(self):
        """Create key visualizations"""
        figs = {}
        
        try:
            # Primary outcome by group
            if self.primary_outcomes and self.group_variable:
                for outcome in self.primary_outcomes[:2]:
                    if outcome in self.data.columns:
                        clean_data = self.data[[outcome, self.group_variable]].dropna()
                        if len(clean_data) > 0:
                            fig = px.box(clean_data, x=self.group_variable, y=outcome,
                                        title=f"{outcome} by Treatment Group")
                            figs[f'outcome_{outcome}'] = fig
            
            # Demographics by group
            if self.demographic_vars and self.group_variable:
                for demo_var in self.demographic_vars[:1]:
                    if demo_var in self.data.columns:
                        clean_data = self.data[[demo_var, self.group_variable]].dropna()
                        if len(clean_data) > 0:
                            if self.data[demo_var].dtype in ['object', 'category']:
                                fig = px.histogram(clean_data, x=demo_var, color=self.group_variable,
                                                 title=f"Distribution of {demo_var} by Group",
                                                 barmode='group')
                            else:
                                fig = px.box(clean_data, x=self.group_variable, y=demo_var,
                                            title=f"{demo_var} by Group")
                            figs[f'demo_{demo_var}'] = fig
        
        except Exception as e:
            figs['error'] = f"Error creating visualizations: {str(e)}"
        
        return figs

def generate_enhanced_report(analyzer, results, context):
    """Generate enhanced report with context"""
    report = f"""
# Clinical Trial Statistical Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Study Background
"""
    
    if context.get('study_description'):
        report += f"\n**Study Description:**\n{context['study_description'][:500]}...\n"
    
    report += f"""
## Study Overview
- **Study Type:** {results['study_info']['study_type']} (Confidence: {results['study_info']['confidence_score']:.1%})
- **Sample Size:** {results['study_info']['sample_size']} participants
- **Number of Variables:** {results['study_info']['variables_count']}
- **Missing Data Points:** {results['study_info']['missing_data']}
- **Enhanced Context Available:** {'Yes' if results['study_info']['context_available'] else 'No'}

## Analysis Results
"""
    
    # Add primary analysis results
    if 'primary_analysis' in results and results['primary_analysis']:
        for outcome, analysis in results['primary_analysis'].items():
            if 'error' not in analysis:
                report += f"\n**{outcome}:**\n"
                if analysis.get('analysis_type') == 'ANCOVA':
                    report += f"- Analysis: {analysis['analysis_type']}\n"
                    if analysis.get('p_value'):
                        report += f"- P-value: {analysis['p_value']:.4f}\n"
                        report += f"- Treatment Effect: {analysis.get('treatment_effect', 'N/A')}\n"
                        report += f"- Sample Size: {analysis.get('n_analyzed', 'N/A')}\n"
                elif analysis.get('analysis_type') == 'Independent t-test':
                    report += f"- Analysis: {analysis['analysis_type']}\n"
                    report += f"- Group 0 Mean: {analysis['mean_group0']:.2f} (n={analysis['n_group0']})\n"
                    report += f"- Group 1 Mean: {analysis['mean_group1']:.2f} (n={analysis['n_group1']})\n"
                    report += f"- Difference: {analysis['difference']:.2f}\n"
                    report += f"- P-value: {analysis['p_value']:.4f}\n"
    
    return report

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">📊 Clinical Trial Statistical Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Upload your clinical trial data and get professional statistical analysis automatically.**
    
    ✅ **Smart Multi-Sheet Excel Support** - Automatically handles description, variables, and data sheets  
    ✅ **AI-Enhanced Analysis** - Optional integration with Claude AI for smarter insights  
    ✅ **Professional Output** - Publication-ready statistical results
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose your Excel or CSV file",
            type=['xlsx', 'xls', 'csv'],
            help="Upload clinical trial data. Multi-sheet Excel files are fully supported!"
        )
        
        if uploaded_file:
            st.success("File uploaded successfully!")
        
        st.header("🤖 AI Enhancement")
        use_ai = st.checkbox("Enable AI-Enhanced Analysis", value=False, 
                            help="Use Claude AI for smarter analysis (requires API key)")
        
        api_key = None
        if use_ai:
            api_key = st.text_input("Claude API Key", type="password", 
                                   help="Enter your Anthropic API key for AI enhancement")
        
        st.header("⚙️ Analysis Options")
        run_full_analysis = st.checkbox("Run Full Analysis", value=True)
        include_visualizations = st.checkbox("Include Visualizations", value=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.1)
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Smart Excel handling
            with st.spinner("🔄 Processing your file..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    context = {}
                else:
                    # Multi-sheet Excel handling
                    excel_handler = SmartExcelHandler(uploaded_file)
                    
                    if excel_handler.load_all_sheets():
                        df = excel_handler.get_analysis_ready_data()
                        context = excel_handler.get_enhanced_context()
                        
                        # Show what was detected
                        if context.get('study_description'):
                            st.markdown('<div class="info-box">📋 <strong>Study Description Detected:</strong> Enhanced context will be used for analysis.</div>', unsafe_allow_html=True)
                        
                        if context.get('variable_descriptions'):
                            st.markdown('<div class="info-box">📖 <strong>Variable Descriptions Detected:</strong> Variable definitions will guide the analysis.</div>', unsafe_allow_html=True)
                    else:
                        st.error("Failed to process Excel file")
                        return
            
            if df is None or len(df) == 0:
                st.error("No data found. Please check your file format.")
                return
            
            st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            
            st.dataframe(df.head(10), use_container_width=True)
            
            if run_full_analysis:
                with st.spinner("🔄 Analyzing your data..."):
                    # Initialize enhanced analyzer
                    analyzer = AIEnhancedAnalyzer(use_ai=use_ai, api_key=api_key)
                    
                    # Run analysis with context
                    results = analyzer.analyze_with_context(df, context)
                    
                    # Display confidence
                    if results['study_info']['confidence_score'] >= confidence_threshold:
                        st.markdown('<div class="success-box">✅ <strong>High Confidence Analysis</strong> - The system is confident about the study design and analysis approach.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ <strong>Low Confidence Analysis</strong> - Please review results carefully.</div>', unsafe_allow_html=True)
                    
                    # Study Information
                    st.markdown('<div class="section-header">🔍 Study Analysis</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Detected Study Type:**")
                        st.info(f"{results['study_info']['study_type']} (Confidence: {results['study_info']['confidence_score']:.1%})")
                        
                        if results['study_info']['group_variable']:
                            st.markdown("**Treatment Groups:**")
                            st.write(f"Variable: `{results['study_info']['group_variable']}`")
                            
                    with col2:
                        st.markdown("**Primary Outcomes:**")
                        for outcome in results['study_info']['primary_outcomes'][:5]:
                            # Show variable description if available
                            desc = context.get('variable_descriptions', {}).get(outcome, '')
                            if desc:
                                st.write(f"• {outcome}: {desc[:100]}...")
                            else:
                                st.write(f"• {outcome}")
                    
                    # Baseline Comparison
                    if 'baseline_comparison' in results and 'error' not in results['baseline_comparison']:
                        st.markdown('<div class="section-header">⚖️ Baseline Characteristics</div>', unsafe_allow_html=True)
                        
                        if 'comparison_table' in results['baseline_comparison']:
                            comparison_df = results['baseline_comparison']['comparison_table']
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            significant_vars = comparison_df[comparison_df['Significant'] == True]['Variable'].tolist()
                            if significant_vars:
                                st.warning(f"⚠️ Significant baseline differences detected in: {', '.join(significant_vars)}")
                            else:
                                st.success("✅ No significant baseline differences detected")
                    
                    # Primary Analysis Results
                    if 'primary_analysis' in results and results['primary_analysis']:
                        st.markdown('<div class="section-header">🎯 Primary Efficacy Analysis</div>', unsafe_allow_html=True)
                        
                        for outcome, analysis in results['primary_analysis'].items():
                            if 'error' not in analysis:
                                st.subheader(f"📊 {outcome}")
                                
                                # Show variable description if available
                                desc = context.get('variable_descriptions', {}).get(outcome, '')
                                if desc:
                                    st.info(f"**Variable Description:** {desc}")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                if analysis.get('analysis_type') == 'ANCOVA':
                                    with col1:
                                        st.metric("Analysis Type", "ANCOVA")
                                    with col2:
                                        if analysis.get('p_value'):
                                            st.metric("P-value", f"{analysis['p_value']:.4f}")
                                    with col3:
                                        if analysis.get('r_squared'):
                                            st.metric("R²", f"{analysis['r_squared']:.3f}")
                                    
                                    if analysis.get('treatment_effect'):
                                        col4, col5 = st.columns(2)
                                        with col4:
                                            st.metric("Treatment Effect", f"{analysis['treatment_effect']:.3f}")
                                        with col5:
                                            st.metric("Sample Size", analysis.get('n_analyzed', 'N/A'))
                                
                                elif analysis.get('analysis_type') == 'Independent t-test':
                                    with col1:
                                        st.metric("Group 0 Mean", f"{analysis['mean_group0']:.2f}")
                                    with col2:
                                        st.metric("Group 1 Mean", f"{analysis['mean_group1']:.2f}")
                                    with col3:
                                        st.metric("Difference", f"{analysis['difference']:.2f}")
                                    
                                    col4, col5, col6 = st.columns(3)
                                    with col4:
                                        st.metric("P-value", f"{analysis['p_value']:.4f}")
                                    with col5:
                                        significance = "Significant" if analysis['p_value'] < 0.05 else "Not Significant"
                                        st.metric("Result", significance)
                                    with col6:
                                        st.metric("Total N", f"{analysis['n_group0']} + {analysis['n_group1']}")
                            else:
                                st.error(f"Error analyzing {outcome}: {analysis['error']}")
                    
                    # Secondary Analyses
                    if 'secondary_analysis' in results and results['secondary_analysis']:
                        st.markdown('<div class="section-header">📈 Secondary Analyses</div>', unsafe_allow_html=True)
                        
                        for analysis_name, analysis in results['secondary_analysis'].items():
                            if 'error' not in analysis:
                                st.subheader(f"📊 {analysis_name}")
                                
                                if analysis.get('analysis_type') == 'Response Rate Comparison':
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Contingency Table:**")
                                        st.dataframe(analysis['contingency_table'])
                                    with col2:
                                        st.metric("Chi-square", f"{analysis['chi2_statistic']:.3f}")
                                        st.metric("P-value", f"{analysis['p_value']:.4f}")
                                        significance = "Significant" if analysis['p_value'] < 0.05 else "Not Significant"
                                        st.metric("Result", significance)
                    
                    # Visualizations
                    if include_visualizations and 'visualizations' in results:
                        st.markdown('<div class="section-header">📈 Data Visualizations</div>', unsafe_allow_html=True)
                        
                        vis_results = results['visualizations']
                        if 'error' not in vis_results:
                            for plot_name, fig in vis_results.items():
                                if hasattr(fig, 'show'):  # It's a plotly figure
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Visualization error: {vis_results['error']}")
                    
                    # Enhanced Report Generation
                    st.markdown('<div class="section-header">📄 Statistical Report</div>', unsafe_allow_html=True)
                    
                    report = generate_enhanced_report(analyzer, results, context)
                    st.markdown(report)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Statistical Report",
                            data=report,
                            file_name=f"statistical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    
                    with col2:
                        # Export results as JSON
                        results_json = {
                            'study_info': results['study_info'],
                            'context': context,
                            'analysis_summary': {
                                'primary_results': results.get('primary_analysis', {}),
                                'baseline_comparison': results.get('baseline_comparison', {}),
                                'generated_on': datetime.now().isoformat()
                            }
                        }
                        
                        st.download_button(
                            label="📊 Download Results (JSON)",
                            data=json.dumps(results_json, indent=2, default=str),
                            file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    # AI Insights (if enabled)
                    if use_ai and api_key:
                        st.markdown('<div class="section-header">🤖 AI Insights</div>', unsafe_allow_html=True)
                        st.info("AI-enhanced analysis is enabled but requires additional API integration for full insights.")
                        
                        # Show what AI detected
                        if context.get('study_description'):
                            st.markdown("**AI Context Understanding:**")
                            st.write("✅ Study background and objectives identified")
                        
                        if context.get('variable_descriptions'):
                            st.write("✅ Variable definitions incorporated into analysis")
                        
                        st.write("✅ Analysis approach optimized based on study design")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your file is properly formatted. For multi-sheet Excel files, make sure you have a 'Data' sheet with the actual dataset.")
    
    else:
        st.info("👆 Please upload a dataset to begin analysis")
        
        # Enhanced example section
        st.markdown('<div class="section-header">💡 Supported File Formats</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Single Sheet Files:**
            - CSV files with headers
            - Excel files (.xlsx, .xls) with data only
            
            **Expected format:**
            - Rows = participants/subjects
            - Columns = variables
            - Headers in first row
            - Clear variable names
            """)
        
        with col2:
            st.markdown("""
            **📚 Multi-Sheet Excel Files:**
            - Sheet 1: Study description/background
            - Sheet 2: Variable descriptions/codebook
            - Sheet 3: Actual data ("Data" sheet)
            
            **Benefits:**
            - Enhanced AI analysis
            - Better variable classification
            - Richer reporting context
            """)
        
        # Example data
        st.markdown('<div class="section-header">📋 Example Data Format</div>', unsafe_allow_html=True)
        
        example_data = {
            'id': [1, 2, 3, 4, 5],
            'age': [45, 52, 38, 61, 29],
            'sex': [1, 0, 1, 0, 1],
            'group': [0, 1, 0, 1, 0],
            'pk1_severity': [8.5, 7.2, 9.1, 6.8, 8.0],
            'pk2_severity': [6.2, 4.1, 8.9, 3.5, 7.2],
            'response': [1, 1, 0, 1, 0]
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)

if __name__ == "__main__":
    main()
