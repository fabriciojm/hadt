# Use a lightweight Python image
FROM python:3.10-slim

# Create a writable cache directory
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# Set environment variable for cache
ENV CACHE_DIR=/app/cache

# Set the working directory in the container
WORKDIR /app

# Copy the API code and dependencies
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port
EXPOSE 8000

# Run the FastAPI server with Uvicorn
CMD ["uvicorn", "fast:app", "--host", "0.0.0.0", "--port", "8000"]