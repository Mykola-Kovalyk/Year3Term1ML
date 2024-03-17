# Use an official Python runtime as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

COPY requirements.web.txt .
RUN pip install --no-cache-dir -r requirements.web.txt

COPY mnist_classifier.pth .
COPY run.py .
COPY mnist_classifier.py .

# Expose port 80 for the Flask web server
EXPOSE 80

# Set the entrypoint command to start the Flask web server
CMD ["python", "run.py"]