#!/bin/bash
# Author: Eddie Lee, edl56@cornell.edu

# Run the following commands in your shell prompt to create custom Python environment.
# conda create --name <env_name>
# conda activate <env_name>

echo 'We recommend creating a custom Python environment first. Open setup.sh for instructions.'
read -p 'Are you sure want to install new packages into your current Python environment? [Yn]' yn
    case $yn in
        [Yy]* ) ./install.sh;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
