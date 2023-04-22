# Use the official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any additional packages needed
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# COPY . .

# Expose a port for serving predictions (optional)
# EXPOSE 5000

# Set the entrypoint for the container (optional)
# This can be a script that runs training, prediction, or both
# ENTRYPOINT ["python", "your_script.py"]


# command
# docker build -t heartcheck .
# docker run -it --rm -p -v "/mnt/c/Users/91961/Documents/Learn/AIML/MLOps/MiniProject/MiniProj2/Project/heartcheck":/app  5000:5000 heartcheck
