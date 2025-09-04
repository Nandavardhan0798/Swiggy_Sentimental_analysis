# Use Python 3.9 image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port (default 8501)
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]