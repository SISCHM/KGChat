# Use the miniconda3 base image
FROM mambaorg/micromamba:latest

# Set the working directory in the container
WORKDIR /app

# Copy the environment.yml file into the container at /app
COPY environment.yml .

# Create the conda environment using micromamba
RUN micromamba create -f environment.yml -n g-docker

# Activate the environment and set the shell
SHELL ["micromamba", "run", "-n", "g-docker", "/bin/bash", "-c"]

# Install PyTorch with CUDA support
RUN micromamba run -n g-docker pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the rest of the application code into the container at /app
COPY src /app/src
COPY static /app/static
COPY templates /app/templates
COPY app.py /app/app.py
COPY config.txt /app/config.txt

# Create the chats directory and set permissions
RUN mkdir -p /app/chats && chmod -R 777 /app/chats

# Ensure the environment is activated when the container starts
ENTRYPOINT ["micromamba", "run", "-n", "g-docker", "python", "app.py"]

# Expose the port the app runs on (if needed)
EXPOSE 5000
