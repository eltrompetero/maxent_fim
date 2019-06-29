#!/bin/bash
# Author: Eddie Lee, edl56@cornell.edu

# Run the following commands in your shell prompt to create custom Python environment.
# conda create --name <env_name>
# conda activate <env_name>

read -p 'Are you sure want to install new packages into your Python environment? [Yn]' yn
    case $yn in
        [Yy]* ) ./install.sh;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
