# Use the Python base image
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/devcontainers/python:0-${VARIANT}

# Define the version of Poetry to install
ARG POETRY_VERSION=1.8.3
ENV POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=true

# Create a Python virtual environment for Poetry and install it
RUN pipx install poetry==${POETRY_VERSION}

# Setup for bash
RUN poetry completions bash >> /home/vscode/.bash_completion && \
    echo "export PATH=.:$PATH" >> ~/.bashrc
