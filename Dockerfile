# 1. Start with a lightweight Python operating system
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install necessary system tools (needed for some ML packages)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files into the container
COPY . .

# 6. Hugging Face Spaces strictly requires port 7860
EXPOSE 7860

# 7. Start the FastAPI server
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860"]