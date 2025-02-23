FROM python:3.10

# Set the working directory
WORKDIR /app

ENV PYTHONPATH=/app

# Copy the current directory contents into the container at /app
COPY . /app

# Install packages specified in pyproject.toml
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install --no-root

ENTRYPOINT ["poetry", "run", "python", "main.py"]