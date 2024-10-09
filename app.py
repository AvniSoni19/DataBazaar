"""
File: app.py
Description: This script provides a Streamlit-based web application for data analysis and visualization. 
             It allows users to upload CSV files, refine data, perform analysis, and generate visualizations.
Author: Avni Soni
Date: 2023-10-05
"""

__author__ = "Avni Soni"
__copyright__ = "Copyright 2023, Avni Soni"
__license__ = "MIT"


import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
import warnings
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_experimental.agents.agent_toolkits.pandas.base')


# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_agent' not in st.session_state:
    st.session_state.df_agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None
if 'imputed_columns' not in st.session_state:
    st.session_state.imputed_columns = {}
if 'selected_actions' not in st.session_state:
    st.session_state.selected_actions = {
        'missing_values': None,
        'remove_duplicates': False
    }

class InsightWise:
    """
    InsightWise is a Streamlit-based application for data analysis and visualization.
    It allows users to upload CSV files, refine data, perform analysis, and generate visualizations.
    The class handles the initialization of the application, data processing, and interaction with OpenAI's API for chat-based data analysis.
    """

    def __init__(self):
        """
        Initializes the class instance.

        This method retrieves the OpenAI API key from the environment variables
        and stores it in the Streamlit session state if available.

        Attributes:
            api_key (str): The OpenAI API key retrieved from environment variables.
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            st.session_state.openai_api_key = api_key
        
    def initialize_app(self):
        """
        Initializes the Streamlit application with a title, API key input, data upload section, 
        and a tabbed interface for data refinement, analysis, and visualizations.
        Method:
        initialize_app(self):
            Sets up the Streamlit application interface, including:
            - Displaying the application title.
            - Prompting the user for an OpenAI API key if not already set.
            - Providing a file uploader for CSV files.
            - Displaying a tabbed interface for data refinement, analysis, and visualizations if data is loaded.
        Returns:
            None
        """
        st.title("INSIGHTWISE")
        
        # # API Key Input if not already set
        # if not st.session_state.openai_api_key:
        #     st.session_state.openai_api_key = st.text_input(
        #         "Enter your OpenAI API key:",
        #         type="password",
        #         help="You can find your API key at https://platform.openai.com/account/api-keys"
        #     )
        #     if not st.session_state.openai_api_key:
        #         st.warning("Please enter your OpenAI API key to enable chat functionality.")
        #         return
        
        # Data upload section
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            self.load_and_process_data(uploaded_file)
            if st.session_state.openai_api_key is not None:
                self.initialize_chat_agent(st.session_state.df)
            
        # Display tabbed interface if data is loaded
        if st.session_state.df is not None:
            tab1, tab2, tab3 = st.tabs(["🛠️ Data Refinement", "💡 Analysis", "📊 Visualizations"])
            
            with tab1:
                self.display_refinement_tab()
            
            with tab2:
                self.display_analysis_tab()
            
            with tab3:
                self.display_visualization_tab()

    def display_refinement_tab(self):
        """
        Display the Data Refinement tab in the Streamlit application.
        This method sets up the header for the Data Refinement tab and generates a data report
        based on the DataFrame stored in the Streamlit session state.
        Args:
            None
        Returns:
            None
        """
        st.header("Data Refinement")
        
        self.generate_data_report(st.session_state.df)

    def display_analysis_tab(self):
        """
        Displays the analysis tab in the Streamlit application, allowing users to chat with their data.
        This method handles the following:
        - Displays a header for the chat functionality.
        - Checks if the OpenAI API key is available in the session state.
        - Prompts the user to enter their OpenAI API key if it is not available.
        - Verifies and initializes the chat agent using the provided API key.
        - Provides information on how to obtain an OpenAI API key.
        - Initializes the chat agent if the API key is available but the agent is not yet initialized.
        - Displays the interactive chat interface if everything is properly initialized.
        Returns:
            None
        """
        st.header("Chat with Your Data")
        
        # Check if OpenAI API key is available
        if not st.session_state.openai_api_key:
            st.warning("⚠️ OpenAI API Key Required")
            
            # Create columns for better layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                api_key = st.text_input(
                    "Please enter your OpenAI API key to enable chat functionality:",
                    type="password",
                    help="You can find your API key at https://platform.openai.com/account/api-keys"
                )
            
            with col2:
                if st.button("Submit", key="submit_api_key"):
                    if api_key:
                        try:
                            # Verify the API key by attempting to initialize the chat agent
                            st.session_state.openai_api_key = api_key
                            with st.spinner("🔄 Verifying API key..."):
                                self.initialize_chat_agent(st.session_state.df)
                            st.success("✅ API Key verified and chat agent initialized successfully!")
                            st.rerun()  # Rerun to refresh the interface
                        except Exception as e:
                            st.error(f"❌ Invalid API key or initialization error: {str(e)}")
                            st.session_state.openai_api_key = None  # Reset the invalid key
                    else:
                        st.error("Please enter an API key")
            
            # Show information about getting an API key
            with st.expander("ℹ️ How to get an OpenAI API key"):
                st.markdown("""
                1. Go to [OpenAI's website](https://platform.openai.com/signup)
                2. Create an account or sign in
                3. Navigate to API keys section
                4. Click on 'Create new secret key'
                5. Copy the key and paste it above
                
                Note: Keep your API key secure and never share it publicly.
                """)
            
            return  # Exit the method if no API key is available
        
        # If API key is available but chat agent isn't initialized, initialize it
        if st.session_state.df_agent is None:
            try:
                with st.spinner("🔄 Initializing chat agent..."):
                    self.initialize_chat_agent(st.session_state.df)
                st.success("✅ Chat agent initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing chat agent: {str(e)}")
                st.session_state.openai_api_key = None  # Reset the API key if initialization fails
                st.rerun()
                return
        
        # If everything is properly initialized, display the chat interface
        self.display_interactive_interface()


    # def display_analysis_tab(self):
    #     """
    #     Method to display the analysis tab in the application.
    #     This method sets up the header for the analysis tab and calls another method to display the interactive interface.
    #     Args:
    #         self: Instance of the class.
    #     Returns:
    #         None
    #     """
    #     st.header("Chat with Your Data")
        
    #     self.display_interactive_interface()

    def display_visualization_tab(self):
        """
        Method to display various data visualizations in a Streamlit app.
        This method provides an interface for users to select and display different types of data visualizations 
        including Distribution Plot, Scatter Plot, Bar Chart, and Correlation Heatmap. It uses Plotly for 
        interactive visualizations and Seaborn for the heatmap.
        Parameters:
        None
        Returns:
        None
        Visualization Types:
        - Distribution Plot: Displays a histogram for a selected numeric column.
        - Scatter Plot: Displays a scatter plot for two selected numeric columns.
        - Bar Chart: Displays a bar chart for a selected categorical column.
        - Correlation Heatmap: Displays a heatmap of the correlation matrix for numeric columns.
        Notes:
        - The method assumes that the DataFrame is stored in the Streamlit session state as `st.session_state.df`.
        - The method handles cases where there are no numeric or categorical columns available for the selected visualization type.
        """
        st.header("Data Visualizations")
        df = st.session_state.df

        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Distribution Plot", "Scatter Plot", "Bar Chart", "Correlation Heatmap"]
        )

        if viz_type == "Distribution Plot":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty:
                col = st.selectbox("Select column for distribution plot", numeric_cols)
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No numeric columns available for distribution plot")

        elif viz_type == "Scatter Plot":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) >= 2:
                col_x = st.selectbox("Select X-axis column", numeric_cols)
                col_y = st.selectbox("Select Y-axis column", [col for col in numeric_cols if col != col_x])
                fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_x} vs {col_y}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Need at least two numeric columns for scatter plot")

        elif viz_type == "Bar Chart":
            categorical_cols = df.select_dtypes(include=['object']).columns
            if not categorical_cols.empty:
                col = st.selectbox("Select categorical column", categorical_cols)
                
                # Fix for the bar chart
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = ['Category', 'Count']  # Rename columns
                
                fig = px.bar(
                    value_counts, 
                    x='Category', 
                    y='Count', 
                    title=f"Count of {col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No categorical columns available for bar chart")

        elif viz_type == "Correlation Heatmap":
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else:
                st.write("No numeric columns available for correlation heatmap") 
            
    def load_and_process_data(self, uploaded_file):
        """
        Method to load and process data from an uploaded CSV file.
        This method reads a CSV file into a pandas DataFrame, stores it in the session state,
        and initializes a chat agent with the DataFrame. If an error occurs during the file
        loading process, an error message is displayed.
        Args:
            uploaded_file (UploadedFile): The uploaded CSV file to be processed.
        Raises:
            Exception: If there is an error loading the file, an error message is displayed.
        """
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # # Generate data report
            # self.generate_data_report(df)
            
            # Initialize chat agent
            # self.initialize_chat_agent(df)
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    def apply_all_refinements(self, df, missing_strategy, remove_duplicates):
        """
        Apply all selected refinements to the dataset.
        This method performs data cleaning operations on the provided DataFrame based on the specified strategies for handling missing values and removing duplicates.
        Parameters:
        df (pandas.DataFrame): The input DataFrame to be refined.
        missing_strategy (str): The strategy to handle missing values. Options are 'Select strategy', 'Fill with mean', or 'Fill with median'.
        remove_duplicates (bool): A flag indicating whether to remove duplicate rows from the DataFrame.
        Returns:
        pandas.DataFrame: The refined DataFrame after applying the selected refinements.
        str: A success message indicating the completion of the data refinement process.
        """
        """Apply all selected refinements to the dataset"""
        original_rows = len(df)
        
        # Handle missing values
        if missing_strategy != 'Select strategy':
            # Handle numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    value = df[col].mean() if missing_strategy == 'Fill with mean' else df[col].median()
                    df[col].fillna(value, inplace=True)
            
            # Handle non-numeric columns
            non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
            for col in non_numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna('NaN', inplace=True)

        # Remove duplicates
        if remove_duplicates:
            df = df.drop_duplicates()

        # Return refined dataframe and single success message
        success_message = "✅ Data refinement completed successfully!"
        return df, success_message

    def generate_data_report(self, df):
        """
        Generates a comprehensive data analysis report using Streamlit.
        This method provides an interactive data analysis report that includes basic information,
        data preview, and options for handling missing values and duplicates. Users can apply
        refinements and download the refined data.
        Args:
            df (pd.DataFrame): The DataFrame containing the data to be analyzed and refined.
        Returns:
            None
        """
        with st.expander("📊 Data Analysis Report", expanded=True):
            # Basic info
            st.markdown("### 📋 Basic Information")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", len(df))
            col2.metric("Columns", len(df.columns))
            col3.metric("Size (MB)", round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Missing values and duplicates section
            st.markdown("### 🛠️ Data Refinement Options")
            
            # Missing values handling
            missing_values = df.isnull().sum()
            missing_cols = missing_values[missing_values > 0]
            if not missing_cols.empty:
                st.markdown("#### 🔍 Missing Values")
                for col, count in missing_cols.items():
                    percentage = (count / len(df)) * 100
                    st.write(f"- {col}: {count} missing values ({percentage:.2f}%)")
                
                st.session_state.selected_actions['missing_values'] = st.selectbox(
                    "Handle missing values:",
                    ['Select strategy', 'Fill with mean', 'Fill with median', 'Drop rows']
                )
            
            # Duplicates handling
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                st.markdown("#### 🔄 Duplicates")
                st.write(f"Found {duplicates} duplicate rows ({(duplicates/len(df))*100:.2f}%)")
                st.session_state.selected_actions['remove_duplicates'] = st.checkbox("Remove duplicates")

            # Apply All button
            if st.button("Apply All Refinements"):
                refined_df, message = self.apply_all_refinements(
                    df.copy(),
                    st.session_state.selected_actions['missing_values'],
                    st.session_state.selected_actions['remove_duplicates']
                )
                
                # Update the session state with refined DataFrame
                st.session_state.df = refined_df
                
                # Show success message with changes
                st.success(message)
                
                # Offer download option
                csv = refined_df.to_csv(index=False)
                st.download_button(
                    label="Download Refined Data",
                    data=csv,
                    file_name="refined_data.csv",
                    mime="text/csv"
                )
                
                # Reinitialize chat agent with refined data
                # self.initialize_chat_agent(refined_df)


    def format_and_display_response(self, response):
        """
        Method to format and display the response in a structured format based on its content.
        This method attempts to interpret the response content and display it in an appropriate format
        using Streamlit components. It handles various types of response formats such as CSV-like strings,
        key-value pairs, and bullet points.
        Args:
            response (str): The response content to be formatted and displayed.
        Returns:
            None
        """
       
        try:
            # If response is a string containing a table-like structure with commas
            if isinstance(response, str) and ',' in response and '\n' in response:
                try:
                    # Try to create a DataFrame from the string
                    df = pd.read_csv(StringIO(response))
                    st.dataframe(df)
                    return
                except:
                    pass
            
            # If response contains key-value pairs (common in analysis)
            if isinstance(response, str) and ':' in response and '\n' in response:
                lines = [line.strip() for line in response.split('\n') if line.strip() and ':' in line]
                if lines:
                    # Create two columns for key-value display
                    data = {}
                    for line in lines:
                        key, value = line.split(':', 1)
                        data[key.strip()] = value.strip()
                    
                    # Display in a clean table format
                    df = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
                    st.table(df)
                    return

            # If response is a list or bullet points
            if isinstance(response, str) and any(line.strip().startswith(('-', '•', '*')) for line in response.split('\n')):
                lines = [line.strip().lstrip('-').lstrip('•').lstrip('*').strip() 
                        for line in response.split('\n') 
                        if line.strip()]
                for line in lines:
                    st.markdown(f"• {line}")
                return

            # Default: display as regular text
            st.write(response)

        except Exception as e:
            st.write(response)  # Fallback to simple display


    def display_structured_data(self, data):
        """
        Display structured JSON data in a Streamlit app.

        This method takes structured JSON data, which can be either a dictionary or a list, 
        and displays it in a Streamlit app. If the data is a dictionary, it displays the 
        key-value pairs using Streamlit's metric component. If the data is a list, it 
        displays the items as a table.

        Args:
            data (dict or list): The structured JSON data to be displayed. It can be either 
                                 a dictionary of key-value pairs or a list of items.

        """
        
        if isinstance(data, dict):
            # If data is a dictionary of key-value pairs
            cols = st.columns(min(len(data), 3))
            for idx, (key, value) in enumerate(data.items()):
                with cols[idx % 3]:
                    st.metric(label=key, value=value)
        elif isinstance(data, list):
            # If data is a list of items
            self.display_as_table(data)

    def display_key_value_pairs(self, lines):
        """
        Display key-value pairs from a list of strings in a Streamlit app.
        Args:
            lines (list of str): A list of strings where each string contains a key and a value separated by a colon.
        Returns:
            None: This method displays the key-value pairs in the Streamlit app using the metric component.
        """
        
        # Split lines into keys and values
        pairs = [line.split(':', 1) for line in lines]
        
        # Create a clean dictionary
        data = {k.strip(): v.strip() for k, v in pairs}
        
        # Display in columns (3 columns max)
        cols = st.columns(min(len(data), 3))
        for idx, (key, value) in enumerate(data.items()):
            with cols[idx % 3]:
                st.metric(label=key, value=value)

    def display_as_table(self, data):
        """
        Display data as an interactive table or write it directly.
        This method takes a data input and displays it as an interactive table using Streamlit's `st.dataframe` if the data is in a list format. If the data is not a list, it writes the data directly using `st.write`.
        Parameters:
        data (list or any): The data to be displayed. If the data is a list of dictionaries, it will be converted to a DataFrame. If the data is a list of strings that resemble CSV content, it will be parsed and converted to a DataFrame. Otherwise, the data will be displayed directly.
        Returns:
        None
        """

        if isinstance(data, list):
            if isinstance(data[0], dict):
                # If list of dictionaries, convert to DataFrame
                df = pd.DataFrame(data)
            else:
                # If list of strings, try to parse CSV-like content
                try:
                    if isinstance(data[0], str) and ',' in data[0]:
                        # Assume first row is header
                        header = data[0].split(',')
                        rows = [row.split(',') for row in data[1:]]
                        df = pd.DataFrame(rows, columns=header)
                    else:
                        df = pd.DataFrame(data, columns=['Value'])
                except:
                    df = pd.DataFrame(data, columns=['Value'])
            
            # Display as an interactive table
            st.dataframe(df, use_container_width=True)
        else:
            st.write(data)

    def display_as_list(self, lines):
        """
        Method to display a list of lines as a numbered markdown list.
        Args:
            lines (list of str): The list of lines to be displayed.
        Returns:
            None
        """
       
        # Clean up the lines
        cleaned_lines = [line.lstrip('- *•').strip() for line in lines]
        
        # Create a numbered list using markdown
        for idx, line in enumerate(cleaned_lines, 1):
            st.markdown(f"{idx}. {line}")

    def initialize_chat_agent(self, df):
        """Method to initialize a chat agent for analyzing a dataset using OpenAI's GPT-3.5-turbo model.
        Args:
            df (pd.DataFrame): The pandas DataFrame containing the dataset to be analyzed.
        Raises:
            Exception: If there is an error during the initialization of the chat agent, an error message is displayed using Streamlit's st.error.
        This method sets up a chat agent with a custom prompt to encourage structured responses. The prompt specifies formats for single responses,
          multiple statistics or metrics, lists, and tabular data. The chat agent is created using the `create_pandas_dataframe_agent` function and is 
          stored in the Streamlit session state for later use."""
        
        try:

            if not st.session_state.openai_api_key:
                raise ValueError("OpenAI API key is not set")

            chat_model = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                openai_api_key=st.session_state.openai_api_key
            )
            
            # Custom prompt to encourage structured responses
            prefix = """You are an AI assistant analyzing a dataset. When responding:
            1. For single response, use plain text
            2. For multiple statistics or metrics, use "key: value" format, one per line
            3. For lists, use bullet points starting with "-"
            4. For tabular data, use comma-separated values with headers
            
            Example formats:
            
            For metrics:
            Total Count: 100
            Average Value: 45.2
            Maximum: 90
            
            For lists:
            - First insight
            - Second insight
            - Third insight
            
            For tables:
            Category,Count,Percentage
            A,50,25%
            B,75,37.5%
            C,75,37.5%
            """

            agent = create_pandas_dataframe_agent(
                llm=chat_model,
                df=df,
                verbose=True,
                prefix=prefix,
                allow_dangerous_code=True 
            )

            st.session_state.df_agent = agent

        except Exception as e:
            st.error(f"Failed to initialize chat agent: {str(e)}")

    def display_interactive_interface(self):
        """
        Displays an interactive chat interface for users to ask questions about their data.
        This method sets up a Streamlit interface with the following features:
        - An expander with example queries to guide users.
        - A button to clear the chat history.
        - A chat container to display the conversation history.
        - An input container with a form for users to submit their questions.
        - Processes user questions and displays responses from a data analysis agent.
        The chat history is maintained in the Streamlit session state and is updated with each user query and response.
        Raises:
            Exception: If there is an error processing the user's question.
        """

        if 'chat_input' not in st.session_state:
            st.session_state.chat_input = ''
        
        # Example queries to help users
        with st.expander("📝 Not a Data Analyst? Try these Queries!"):
            st.markdown("""
            
            - "Give me the basic statistics of the numerical columns"
            - "What are the key metrics for gender column?"
            - "Show me the distribution of service type in a table format"
            - "Give me a breakdown of device used by category"

            """)
        
        # Clear chat history button
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Chat container
        chat_container = st.container()
        
        # Input container
        input_container = st.container()
        
        with input_container:
            with st.form(key='chat_form', clear_on_submit=True):
                user_question = st.text_input("Ask anything about your data:", key="chat_input")
                submit_button = st.form_submit_button("Send")
        
        # Display chat history
        with chat_container:
            for question, answer in st.session_state.chat_history:
                message(question, is_user=True)
                with st.container():
                    self.format_and_display_response(answer)
        
        # Process the question when form is submitted
        if submit_button and user_question:
            try:
                with st.spinner("🤔 Analyzing..."):
                    response = st.session_state.df_agent.run(user_question)
                
                # Append to chat history
                st.session_state.chat_history.append((user_question, response))
                
                # Rerun the app to update the chat display
                st.rerun()
                
            except Exception as e:
                st.error("I couldn't process that question. Could you try rephrasing it?")


def message(text, is_user=False):
    """
    Display a message in a styled HTML div using Streamlit's markdown function.

    Parameters:
    text (str): The message text to be displayed.
    is_user (bool): A flag indicating whether the message is from the user. 
                    If True, the message is right-aligned with a specific style.
                    If False, the message is left-aligned with a different style.
    """
    if is_user:
        st.markdown(f'<div style="display: flex; justify-content: flex-end;"><div style="background-color: #e6f3ff; padding: 10px; border-radius: 10px; margin: 5px; max-width: 70%;">{text}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="display: flex; justify-content: flex-start;"><div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin: 5px; max-width: 70%;">{text}</div></div>', unsafe_allow_html=True)

def main():
    """
    Main entry point for the application.
    This function configures warnings to be ignored for specific modules and categories,
    then initializes and starts the InsightWise application.
    Warnings Configured:
    - UserWarning for 'langchain_experimental.agents.agent_toolkits.pandas.base' module
    - DeprecationWarning for 'langchain' module
    Initializes:
    - InsightWise application instance and calls its initialize_app method.
    """

    st.set_page_config(
        page_title="insightWise",
        page_icon="🧮",
        # layout="wide",  # Makes the app take up the full width of the browser
        initial_sidebar_state="expanded"
    )

    # Configure warnings at the start of the application
    warnings.filterwarnings('ignore', category=UserWarning, module='langchain_experimental.agents.agent_toolkits.pandas.base')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='langchain')
    

    app = InsightWise()
    app.initialize_app()

if __name__ == "__main__":
    main()



