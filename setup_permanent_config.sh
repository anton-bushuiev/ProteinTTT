#!/bin/bash
# Setup permanent bash configuration

echo "Setting up permanent bash configuration..."

# Backup existing bashrc
cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)

# Add configurations to bashrc
cat .bashrc_additions >> ~/.bashrc

echo "Configuration added to ~/.bashrc"
echo "Please run 'source ~/.bashrc' or start a new terminal session"
