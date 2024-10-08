# insightWise
An intelligent data platform that allows users to interact with product data through natural language.

## Introduction
This application is designed to process and analyze data using various tools and libraries. It integrates with OpenAI's API to provide intelligent responses and data manipulation capabilities.

## Setup
1. **Environment Setup**:
   - Ensure you have Python installed.
   - Install required libraries using `pip install -r requirements.txt`.
   - Create a `.env` file with your OpenAI API key.

2. **Required Libraries**:
   - `langchain_community`
   - `langchain_experimental`
   - `dotenv`
   - `plotly`
   - `matplotlib`
   - `seaborn`
   - `pandas`
   - `numpy`

## Workflow

1. Importing Libraries

2. Filtering Warnings
Specific warnings are filtered out to avoid cluttering the output:

3. Initializing Session State
The session state is initialized to store various states and dataframes:

4. Handling Data
Data is read and processed, with null values being replaced by np.nan:

5. User Interaction
Users interact with the application through a web interface, where they can upload files, input queries, and receive responses. The application uses OpenAI's API to process these queries and provide intelligent responses.

6. Key Functions and Modules
ChatOpenAI: Used for interacting with OpenAI's API.
create_pandas_dataframe_agent: Creates an agent for manipulating pandas dataframes.
plotly, matplotlib, seaborn: Used for data visualization.

7. Handling Numerical and Categorical Data
The application handles numerical and categorical data separately:

Numerical Data: Null values are replaced with np.nan, and the column is converted to a numeric type for comparisons.
Categorical Data: Null values are handled appropriately, and the application looks for exact matches or 'contains' logic in the queries.

8. Example Usage
Here is an example of how to use the application:

-Upload a CSV file.
-Enter a query to filter or analyze the data.
-View the results and visualizations.

Conclusion
This application leverages powerful libraries and APIs to provide a robust data processing and analysis tool. By following the setup instructions and understanding the workflow, users can effectively utilize the application for their data needs.
