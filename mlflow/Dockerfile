FROM python:3.8

# Set the working directory in the container
WORKDIR /app

COPY requirements.ci.txt .
RUN pip install --no-cache-dir -r requirements.ci.txt


# Expose port 80 for the Flask web server
EXPOSE 5000

# Set the entrypoint command to start the Flask web server
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0"]