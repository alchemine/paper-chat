#!/bin/bash

sudo chown -R vscode:vscode /app
sudo chown -R vscode:vscode /home/vscode/.ssh
poetry install --no-root