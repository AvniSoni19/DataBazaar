import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from openai import OpenAI
from datetime import datetime
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Conversational Data Analysis",
    layout="wide"
)

def initialize_openai_client():
    """Initialize OpenAI client with API key from environment variables."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("OpenAI API key not found. Please check your environment variables.")
        return False
    try:
        st.session_state.client = OpenAI(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return False

def analyze_data_quality(df):
    """Analyze data quality and return issues in conversational format."""
    issues = []
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        missing_cols = ', '.join(missing_values[missing_values > 0].index)
        issues.append(f"I noticed some missing information in these columns: {missing_cols}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"I found {duplicates} duplicate entries in the data")
    
    # Check for potential outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            issues.append(f"In the {col} column, I spotted {outliers} values that seem unusual")
    
    return issues

def refine_data(df):
    """Refine the data and return both the refined DataFrame and a conversational summary."""
    df_refined = df.copy()
    refinement_messages = []
    
    # Handle duplicates
    initial_rows = len(df_refined)
    df_refined = df_refined.drop_duplicates()
    duplicates_removed = initial_rows - len(df_refined)
    if duplicates_removed > 0:
        refinement_messages.append(f"I removed {duplicates_removed} duplicate entries to clean up the data.")
    
    # Handle missing values
    for column in df_refined.columns:
        missing_count = df_refined[column].isnull().sum()
        if missing_count > 0:
            if np.issubdtype(df_refined[column].dtype, np.number):
                # For numeric columns, fill with median
                median_value = df_refined[column].median()
                df_refined[column] = df_refined[column].fillna(median_value)
                refinement_messages.append(f"I filled {missing_count} missing values in the numeric column {column} with the median value.")
            else:
                # For non-numeric columns, fill with 'Unknown'
                df_refined[column] = df_refined[column].fillna('Unknown')
                refinement_messages.append(f"I marked {missing_count} missing values in the {column} column as 'Unknown'.")
    
    # Handle outliers in numeric columns
    numeric_columns = df_refined.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        Q1 = df_refined[column].quantile(0.25)
        Q3 = df_refined[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_refined[(df_refined[column] < lower_bound) | 
                            (df_refined[column] > upper_bound)]
        if len(outliers) > 0:
            median_value = df_refined[column].median()
            df_refined.loc[(df_refined[column] < lower_bound) | 
                          (df_refined[column] > upper_bound), column] = median_value
            refinement_messages.append(f"I replaced {len(outliers)} outlier values in the {column} column with the median value.")
    
    refinement_summary = "I've cleaned up the data. " + " ".join(refinement_messages)
    return df_refined, refinement_summary

def reset_session_state():
    """Reset session state when a new file is uploaded"""
    st.session_state.df = None
    st.session_state.df_original = None
    st.session_state.is_refined = False
    st.session_state.conversation_history = []
    st.session_state.current_dataset = 'original'

def display_result(result, context=""):
    """Display analysis result in appropriate format"""
    if isinstance(result, pd.DataFrame):
        if result is None or len(result) == 0:
            st.write("I couldn't find any matching data.")
        else:
            st.write(f"I found {len(result)} relevant data points:")
            st.dataframe(result)
            
    elif isinstance(result, pd.Series):
        if result is None or result.empty:
            st.write("I couldn't find any relevant data.")
        else:
            # Convert series to DataFrame for better display
            df_result = pd.DataFrame(result).T
            st.dataframe(df_result)
            
    else:
        st.write(format_result_as_sentence(result, context))

def format_result_as_sentence(result, context=""):
    """Convert analysis result into a conversational response for non-tabular data"""
    if isinstance(result, (int, float)):
        if result is None or pd.isna(result):
            return "I'm not able to calculate that value with the available data."
                
        if any(word in context.lower() for word in ['average', 'mean']):
            return f"The average comes out to {result:.2f}"
        elif 'sum' in context.lower():
            return f"Adding those up, I get {result:.2f}"
        elif any(word in context.lower() for word in ['count', 'how many']):
            return f"I counted {result:,} instances"
        else:
            return f"The value is {result}"
    elif result is None:
        return "I wasn't able to find any results for that query."
    else:
        if pd.isna(result):
            return "That information isn't available in the data."
        return f"Here's what I found: {str(result)}"

def get_current_dataframe():
    """Get the currently selected dataframe"""
    if st.session_state.current_dataset == 'refined' and st.session_state.is_refined:
        return st.session_state.df
    return st.session_state.df_original

def interpret_comparison(text):
    """Interpret comparison operators from text."""
    operators = {
        'greater than': '>',
        'more than': '>',
        'above': '>',
        'over': '>',
        'higher than': '>',
        'less than': '<',
        'lower than': '<',
        'below': '<',
        'under': '<',
        'equal to': '==',
        'equals': '==',
        'is': '=='
    }
    
    for phrase, operator in operators.items():
        if phrase in text.lower():
            return operator
    return None

def extract_numeric_value(text):
    """Extract numeric value from text."""
    import re
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return float(numbers[0]) if numbers else None

def get_column_type(df, column):
    """Get the data type category of a column."""
    if np.issubdtype(df[column].dtype, np.number):
        return 'numeric'
    elif df[column].dtype == 'bool':
        return 'boolean'
    elif df[column].dtype == 'datetime64[ns]':
        return 'datetime'
    else:
        return 'categorical'

def dynamic_filter(df, query_text):
    """
    Dynamically filter DataFrame based on natural language query.
    Returns filtered DataFrame and explanation message.
    """
    try:
        # Lowercase all column names for easier matching
        df.columns = df.columns.str.lower()
        
        # Find which column the query might be about
        potential_columns = [col for col in df.columns if col in query_text.lower()]
        
        if not potential_columns:
            return df, "I couldn't identify which column you wanted to filter by. Could you specify the column name?"
        
        target_column = potential_columns[0]  # Use the first matched column
        column_type = get_column_type(df, target_column)
        
        # Handle different types of columns differently
        if column_type == 'numeric':
            # First, try to extract direct comparison from query
            if '>' in query_text:
                parts = query_text.split('>')
                try:
                    value = float(parts[1].strip())
                    result = df[df[target_column] > value].copy()
                    return result, f"Found {len(result)} items where {target_column} is greater than {value}"
                except (ValueError, IndexError):
                    pass
            
            # If direct comparison fails, try natural language interpretation
            operator = interpret_comparison(query_text)
            value = extract_numeric_value(query_text)
            
            if operator and value is not None:
                if operator == '>':
                    result = df[df[target_column] > value].copy()
                    description = f"greater than {value}"
                elif operator == '<':
                    result = df[df[target_column] < value].copy()
                    description = f"less than {value}"
                elif operator == '==':
                    result = df[df[target_column] == value].copy()
                    description = f"equal to {value}"
                
                return result, f"Found {len(result)} items where {target_column} is {description}"
            
            return df, f"I couldn't determine how to filter the numeric column {target_column}. Please specify a comparison (e.g., '> 1500')."
            
        elif column_type == 'categorical':
            # For categorical, we'll look for exact matches or 'contains' logic
            if 'contains' in query_text.lower():
                search_term = query_text.lower().split('contains')[-1].strip()
                result = df.loc[df[target_column].str.contains(search_term, case=False, na=False)].copy()
                return result, f"Found {len(result)} items where {target_column} contains '{search_term}'"
            else:
                # Try to find any word in the query that matches a value in the column
                unique_values = df[target_column].dropna().unique()
                for value in unique_values:
                    if str(value).lower() in query_text.lower():
                        result = df.loc[df[target_column].str.lower() == str(value).lower()].copy()
                        return result, f"Found {len(result)} items where {target_column} is '{value}'"
        
        return df, f"I couldn't determine how you wanted to filter the {target_column} column. Could you be more specific?"
    
    except Exception as e:
        return df, f"I encountered an error while filtering the data: {str(e)}"
        

def process_data_query(df, query_text):
    """
    Process a data query and return results in the format expected by the chat interface.
    """
    try:
        filtered_df, message = dynamic_filter(df, query_text)
        
        if len(filtered_df) == 0:
            return {
                "type": "text",
                "content": "I couldn't find any data matching your criteria."
            }
        elif len(filtered_df) == len(df):
            return {
                "type": "text",
                "content": "The filter didn't change the dataset. Could you be more specific?"
            }
        else:
            return {
                "type": "data",
                "content": filtered_df,
                "context": message
            }
    except Exception as e:
        return {
            "type": "text",
            "content": f"I encountered an error while processing your query: {str(e)}"
        }

def get_ai_response(user_message, df, conversation_history):
    """Get AI response for user message with integrated data refinement and filtering"""
    if not st.session_state.client:
        if not initialize_openai_client():
            return {"type": "text", "content": "I need an API key to help analyze your data. Could you check the configuration?"}
    
    try:
        # Handle data refinement requests
        if any(word in user_message.lower() for word in ['clean', 'refine', 'fix', 'improve']):
            issues = analyze_data_quality(df)
            if issues:
                refined_df, summary = refine_data(df)
                st.session_state.df = refined_df
                st.session_state.is_refined = True
                return {"type": "text", "content": f"I found some issues with the data: {'. '.join(issues)}. {summary} Would you like to know anything specific about the cleaned data?"}
            else:
                return {"type": "text", "content": "I've looked at the data, and it actually appears to be quite clean! Is there anything specific you'd like to know about it?"}
        
        # Handle filtering/querying requests
        if any(word in user_message.lower() for word in ['show', 'find', 'get', 'where', 'which']):
            return process_data_query(df, user_message)
        
        # For other types of queries, use the OpenAI API to generate code
        system_prompt = f"""You are a friendly data analyst helping analyze a dataset with these columns: {', '.join(df.columns)}

        Generate Python code to analyze the data based on the user's question. The code should:
        1. Always use the provided 'df' variable which contains the DataFrame
        2. Store the final result in a variable named 'result'
        3. Use pandas operations appropriate for the question
        4. Include only the necessary code without any imports or print statements

        Return only the Python code without any explanation or markdown formatting.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate code to: {user_message}"}
        ]
        
        response = st.session_state.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        
        generated_code = response.choices[0].message.content.strip()
        
        # Execute the generated code
        local_dict = {'df': df, 'pd': pd, 'np': np, 'result': None}
        exec(generated_code, {"__builtins__": {}}, local_dict)
        
        result = local_dict['result']
        if isinstance(result, pd.DataFrame):
            if len(result) == 0:
                return {"type": "text", "content": "I couldn't find any data matching your criteria."}
            else:
                return {"type": "data", "content": result, "context": user_message}
        else:
            return {"type": "text", "content": format_result_as_sentence(result, user_message)}
                
    except Exception as e:
        return {"type": "text", "content": f"I encountered an error while processing your request: {str(e)}"}



# Initialize the app state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'client' not in st.session_state:
    st.session_state.client = None
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = 'original'
if 'is_refined' not in st.session_state:
    st.session_state.is_refined = False

# Main app
st.title("💬 Smart Data Assistant")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'], on_change=reset_session_state)

if uploaded_file is not None:
    # Read the CSV file if it's not already in session state
    if st.session_state.df_original is None:
        st.session_state.df_original = pd.read_csv(uploaded_file)
        st.session_state.df = st.session_state.df_original.copy()
    
    # Dataset selection
    st.sidebar.write("### 📊 Dataset Selection")
    dataset_options = ['original', 'refined'] if st.session_state.is_refined else ['original']
    st.session_state.current_dataset = st.sidebar.radio(
        "Choose dataset:",
        options=dataset_options,
        index=dataset_options.index(st.session_state.current_dataset)
    )
    
    current_df = get_current_dataframe()
    
    # Display basic information
    st.write("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", current_df.shape[0])
    with col2:
        st.metric("Columns", current_df.shape[1])
    with col3:
        missing_cells = current_df.isnull().sum().sum()
        st.metric("Missing Values", missing_cells)
    with col4:
        duplicates = current_df.duplicated().sum()
        st.metric("Duplicates", duplicates)
    
    # Data Quality Section
    st.write("### 🔍 Data Quality Check")
    issues = analyze_data_quality(current_df)
    
    if issues:
        st.warning("I found some potential issues in your data:")
        for issue in issues:
            st.write(f"- {issue}")
            
        # Add refinement options
        if not st.session_state.is_refined:
            if st.button("📊 Refine Data"):
                with st.spinner("Refining your data..."):
                    refined_df, summary = refine_data(current_df)
                    st.session_state.df = refined_df
                    st.session_state.is_refined = True
                    st.success(summary)
                    st.rerun()
    else:
        st.success("✨ Your data looks clean! I didn't find any major quality issues.")
    
    # Chat interface
    st.write("### 💭 Chat with Your Data")
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message("user" if message["is_user"] else "assistant"):
            if message["is_user"]:
                st.write(message["message"])
            else:
                if message["type"] == "text":
                    st.write(message["content"])
                elif message["type"] == "data":
                    display_result(message["content"], message["context"])
    
    # Chat input
    user_message = st.chat_input("What would you like to know about your data?")
    
    if user_message:
        # Display user message
        with st.chat_message("user"):
            st.write(user_message)
        
        # Add to conversation history
        st.session_state.conversation_history.append({
            "message": user_message,
            "is_user": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Get and display AI response
        with st.chat_message("assistant"):
            response = get_ai_response(
                user_message, 
                current_df,
                st.session_state.conversation_history
            )
            
            if response["type"] == "text":
                st.write(response["content"])
            elif response["type"] == "data":
                display_result(response["content"], response["context"])
        
        # Add AI response to conversation history
        st.session_state.conversation_history.append({
            **response,
            "is_user": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Sidebar options
    with st.sidebar:
        st.write("### ⚙️ Options")
        
        # Add option to clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Add option to download current dataset
        if st.download_button(
            label="📥 Download Current Dataset",
            data=BytesIO(current_df.to_csv(index=False).encode()).getvalue(),
            file_name="processed_data.csv",
            mime="text/csv"
        ):
            st.success("Dataset downloaded!")
        
        # Show data preview
        with st.expander("👁️ Preview Data"):
            st.dataframe(current_df.head())


# import streamlit as st
# import pandas as pd
# import numpy as np
# from io import BytesIO
# from openai import OpenAI
# from datetime import datetime
# import json
# import os

# # Set page configuration
# st.set_page_config(
#     page_title="Conversational Data Analysis",
#     layout="wide"
# )

# def format_result_as_sentence(result, context=""):
#     """Convert analysis result into a conversational response."""
#     try:
#         if isinstance(result, pd.DataFrame):
#             if result is None or len(result) == 0:
#                 return "I looked, but I couldn't find any matching data."
            
#             # Replace None/NaN values with "N/A" for better readability
#             formatted_result = result.fillna("N/A")
            
#             if len(result) == 1:
#                 return f"I found one data point that's relevant to your question. Here it is:\n{formatted_result.to_string()}"
#             else:
#                 return f"I found {len(result)} relevant data points. Here's what I found:\n{formatted_result.to_string()}"
        
#         elif isinstance(result, pd.Series):
#             if result is None or result.empty:
#                 return "I looked into that, but I couldn't find any relevant data."
            
#             formatted_result = result.fillna("N/A")
#             return f"Here's what I found: {formatted_result.to_string()}"
        
#         elif isinstance(result, (int, float)):
#             if result is None or pd.isna(result):
#                 return "I'm not able to calculate that value with the available data."
                
#             # Handle common aggregation contexts
#             if any(word in context.lower() for word in ['average', 'mean']):
#                 return f"The average comes out to {result:.2f}"
#             elif 'sum' in context.lower():
#                 return f"Adding those up, I get {result:.2f}"
#             elif any(word in context.lower() for word in ['count', 'how many']):
#                 return f"I counted {result:,} instances"
#             else:
#                 return f"The value is {result}"
                
#         elif result is None:
#             return "I wasn't able to find any results for that query."
            
#         else:
#             if pd.isna(result):
#                 return "That information isn't available in the data."
#             return f"Here's what I found: {str(result)}"
            
#     except Exception as e:
#         return f"I found something, but I'm having trouble explaining it clearly: {str(result)}"

# def analyze_data_quality(df):
#     """Analyze data quality and return issues in conversational format."""
#     issues = []
    
#     # Check for missing values
#     missing_values = df.isnull().sum()
#     if missing_values.any():
#         missing_cols = ', '.join(missing_values[missing_values > 0].index)
#         issues.append(f"I noticed some missing information in these columns: {missing_cols}")
    
#     # Check for duplicates
#     duplicates = df.duplicated().sum()
#     if duplicates > 0:
#         issues.append(f"I found {duplicates} duplicate entries in the data")
    
#     # Check for potential outliers
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     for col in numeric_cols:
#         z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
#         outliers = (z_scores > 3).sum()
#         if outliers > 0:
#             issues.append(f"In the {col} column, I spotted {outliers} values that seem unusual")
    
#     return issues

# def refine_data(df):
#     """Refine the data and return both the refined DataFrame and a conversational summary."""
#     df_refined = df.copy()
#     refinement_messages = []
    
#     # Handle duplicates
#     initial_rows = len(df_refined)
#     df_refined = df_refined.drop_duplicates()
#     duplicates_removed = initial_rows - len(df_refined)
#     if duplicates_removed > 0:
#         refinement_messages.append(f"I removed {duplicates_removed} duplicate entries to clean up the data.")
    
#     # Handle missing values
#     for column in df_refined.columns:
#         missing_count = df_refined[column].isnull().sum()
#         if missing_count > 0:
#             df_refined[column] = df_refined[column].fillna("Unknown")
#             refinement_messages.append(f"I marked {missing_count} missing values in the {column} column as 'Unknown'.")
    
#     # Handle outliers in numeric columns
#     numeric_columns = df_refined.select_dtypes(include=[np.number]).columns
#     for column in numeric_columns:
#         if not (df_refined[column] == "Unknown").any():
#             Q1 = df_refined[column].quantile(0.25)
#             Q3 = df_refined[column].quantile(0.75)
#             IQR = Q3 - Q1
#             outliers = df_refined[(df_refined[column] < Q1 - 1.5 * IQR) | 
#                                 (df_refined[column] > Q3 + 1.5 * IQR)].index.size
#             if outliers > 0:
#                 refinement_messages.append(f"I identified {outliers} unusual values in the {column} column and marked them as 'Unknown'.")
    
#     refinement_summary = "I've cleaned up the data. " + " ".join(refinement_messages)
#     return df_refined, refinement_summary

# def get_ai_response(user_message, df, conversation_history):
#     """Get AI response for user message with integrated data refinement and formatting."""
#     if not st.session_state.client:
#         if not initialize_openai_client():
#             return "I need an API key to help analyze your data. Could you check the configuration?"
    
#     try:
#         # Handle data refinement requests
#         if any(word in user_message.lower() for word in ['clean', 'refine', 'fix', 'improve']):
#             issues = analyze_data_quality(df)
#             if issues:
#                 refined_df, summary = refine_data(df)
#                 st.session_state.df = refined_df
#                 return f"I found some issues with the data: {'. '.join(issues)}. {summary} Would you like to know anything specific about the cleaned data?"
#             else:
#                 return "I've looked at the data, and it actually appears to be quite clean! Is there anything specific you'd like to know about it?"
        
#         # Construct the system prompt with data analysis instructions
#         system_prompt = f"""You are a friendly data analyst helping analyze a dataset with these columns: {', '.join(df.columns)}

#         Generate Python code to analyze the data based on the user's question. The code should:
#         1. Always store the final result in a variable named 'result'
#         2. Use pandas operations appropriate for the question
#         3. Include proper error handling
#         4. Work with the available columns: {', '.join(df.columns)}

#         Return only the Python code wrapped in ```python tags.
        
#         Example response format:
#         ```python
#         try:
#             # Analysis code here
#             result = df[...] # Store final result
#         except Exception as e:
#             result = f"Error: {{str(e)}}"
#         ```
#         """

#         # Prepare conversation messages
#         messages = [{"role": "system", "content": system_prompt}]
#         messages.append({"role": "user", "content": f"Generate Python code to answer this question about the data: {user_message}"})
        
#         # Get response from OpenAI
#         response = st.session_state.client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             temperature=0.7
#         )
        
#         ai_message = response.choices[0].message.content
        
#         # Extract and execute the code
#         if "```python" in ai_message:
#             try:
#                 # Extract code from the response
#                 code = ai_message[ai_message.find("```python")+10:ai_message.rfind("```")]
                
#                 # Create a safe local environment
#                 local_dict = {
#                     'df': df,
#                     'pd': pd,
#                     'np': np,
#                     'result': None
#                 }
                
#                 # Execute the generated code
#                 exec(code.strip(), {"__builtins__": {}}, local_dict)
                
#                 # Format and return the result
#                 if 'result' in local_dict:
#                     result = local_dict['result']
#                     if isinstance(result, pd.Series):
#                         # For a single row result
#                         return format_result_as_sentence(result, user_message)
#                     elif isinstance(result, pd.DataFrame):
#                         # For multiple row results
#                         return format_result_as_sentence(result, user_message)
#                     else:
#                         # For scalar results or strings
#                         return str(result)
#                 else:
#                     return "I wasn't able to find the information you're looking for. Could you rephrase your question?"
                    
#             except Exception as exec_error:
#                 return f"I encountered an error while analyzing the data: {str(exec_error)}"
#         else:
#             return "I couldn't generate appropriate code to answer your question. Could you rephrase it?"
        
#     except Exception as api_error:
#         return f"I ran into an issue while processing your request: {str(api_error)}"
    

# # Initialize the app state
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'client' not in st.session_state:
#     st.session_state.client = None

# def initialize_openai_client():
#     """Initialize OpenAI client with API key from environment variables."""
#     api_key = os.getenv('OPENAI_API_KEY')
#     if not api_key:
#         st.error("OpenAI API key not found. Please check your environment variables.")
#         return False
#     try:
#         st.session_state.client = OpenAI(api_key=api_key)
#         return True
#     except Exception as e:
#         st.error(f"Error initializing OpenAI client: {str(e)}")
#         return False

# # Initialize OpenAI client
# initialize_openai_client()

# # Main app
# st.title("💬 Smart Data Assistant")

# # File upload
# uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# if uploaded_file is not None:
#     # Read the CSV file if it's not already in session state
#     if 'df' not in st.session_state or st.session_state.df is None:
#         st.session_state.df = pd.read_csv(uploaded_file)
#         st.session_state.df_original = st.session_state.df.copy()  # Keep original version
#         st.session_state.is_refined = False
    
#     # Display basic information
#     st.write("### Dataset Overview")
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Rows", st.session_state.df.shape[0])
#     with col2:
#         st.metric("Columns", st.session_state.df.shape[1])
#     with col3:
#         missing_cells = st.session_state.df.isnull().sum().sum()
#         st.metric("Missing Values", missing_cells)
#     with col4:
#         duplicates = st.session_state.df.duplicated().sum()
#         st.metric("Duplicates", duplicates)
    
#     # Data Quality Section
#     st.write("### 🔍 Data Quality Check")
    
#     # Analyze and display data quality issues
#     issues = analyze_data_quality(st.session_state.df)
    
#     if issues:
#         st.warning("I found some potential issues in your data:")
#         for issue in issues:
#             st.write(f"- {issue}")
            
#         # Add refinement options
#         col1, col2 = st.columns(2)
#         with col1:
#             if not st.session_state.is_refined:
#                 if st.button("📊 Refine Data"):
#                     with st.spinner("Refining your data..."):
#                         refined_df, summary = refine_data(st.session_state.df)
#                         st.session_state.df = refined_df
#                         st.session_state.is_refined = True
#                         st.success(summary)
#                         st.rerun()
#         with col2:
#             if st.session_state.is_refined:
#                 if st.button("↩️ Revert to Original"):
#                     st.session_state.df = st.session_state.df_original.copy()
#                     st.session_state.is_refined = False
#                     st.info("Reverted to original data.")
#                     st.rerun()
#     else:
#         st.success("✨ Your data looks clean! I didn't find any major quality issues.")
    
#     # Show current data status
#     st.info(f"Currently using: {'Refined' if st.session_state.is_refined else 'Original'} dataset")
 
    
#     # Chat interface
#     st.write("### 💭 Chat with Your Data")
#     # st.write("""
#     # I can help you:
#     # - Explore and understand your data
#     # - Clean and refine the dataset
#     # - Analyze trends and patterns
#     # - Answer specific questions
#     # - Provide insights and recommendations
    
#     # Just start chatting below!
#     # """)
    
#     # Display conversation history
#     for message in st.session_state.conversation_history:
#         with st.chat_message("user" if message["is_user"] else "assistant"):
#             st.write(message["message"])
    
#     # Chat input
#     user_message = st.chat_input("What would you like to know about your data?")
    
#     if user_message:
#         # Display user message
#         with st.chat_message("user"):
#             st.write(user_message)
        
#         # Add to conversation history
#         st.session_state.conversation_history.append({
#             "message": user_message,
#             "is_user": True,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         })
        
#         # Get and display AI response
#         with st.chat_message("assistant"):
#             ai_response = get_ai_response(
#                 user_message, 
#                 st.session_state.df,
#                 st.session_state.conversation_history
#             )
#             st.write(ai_response)
        
#         # Add AI response to conversation history
#         st.session_state.conversation_history.append({
#             "message": ai_response,
#             "is_user": False,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         })
    
#     # Sidebar options
#     with st.sidebar:
#         st.write("### ⚙️ Options")
        
#         # Add option to clear chat history
#         if st.button("🗑️ Clear Chat History"):
#             st.session_state.conversation_history = []
#             st.rerun()
        
#         # Add option to download current dataset
#         if st.download_button(
#             label="📥 Download Current Dataset",
#             data=BytesIO(st.session_state.df.to_csv(index=False).encode()).getvalue(),
#             file_name="processed_data.csv",
#             mime="text/csv"
#         ):
#             st.success("Dataset downloaded!")
        
#         # Show data preview
#         with st.expander("👁️ Preview Data"):
#             st.dataframe(st.session_state.df.head())
            
#         # Show refinement history if data was refined
#         if st.session_state.is_refined:
#             with st.expander("📋 Refinement Summary"):
#                 if hasattr(st.session_state.df, 'attrs') and 'quality_report' in st.session_state.df.attrs:
#                     report = st.session_state.df.attrs['quality_report']
#                     st.write("Changes made:")
#                     for step in report['refinement_steps']:
#                         st.write(f"- {step['message']}")

