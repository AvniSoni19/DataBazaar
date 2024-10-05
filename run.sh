#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with your OpenAI API key:"
    echo "OPENAI_API_KEY=your-api-key-here"
    exit 1
fi

# Load environment variables
export $(cat .env | xargs)

# Run the Streamlit app
streamlit run app.py