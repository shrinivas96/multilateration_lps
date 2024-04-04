#!/bin/bash

# where are we right now?
CURRENT_DIR="$PWD"

# relative to cwd, some locations
VENV_NAME="lps"
VENV="$CURRENT_DIR/$VENV_NAME"
REQUIREMENTS_FILE="$CURRENT_DIR/requirements.txt"
MAIN_FILE="$CURRENT_DIR/main.py"

# create a venv, if not already present
create_venv() {
    if [[ ! -d "$VENV" ]]; then
        python3 -m venv "$VENV"
        echo "Virtual environment created at: $VENV."
    fi
}

# activate venv
activate_venv() {
    if [[ -d "$VENV" ]]; then
        source "$VENV/bin/activate"
        echo "$VENV is active."
    else
        echo "Could not activate the virtual environment."
        exit 1
    fi
}

# install dependencies; needs a check to make sure it is only done once
install_dependencies() {
    python -m pip install -r "$REQUIREMENTS_FILE"
    echo "Dependencies installed from $REQUIREMENTS_FILE."
}

# run em all
echo "Setting up the environment..."

create_venv
activate_venv
install_dependencies

echo "Done setting up. Running $MAIN_FILE"
python "$MAIN_FILE"