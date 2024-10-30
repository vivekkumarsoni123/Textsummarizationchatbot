# Use official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install required libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Expose the port for Streamlit
EXPOSE 8501

# Define the entry point for Streamlit (or your custom app script)
CMD ["streamlit", "run", "main.py"]
