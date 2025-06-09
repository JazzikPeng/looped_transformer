#!/bin/bash

SESSION_NAME="tf_loop"
COMMAND="bash exec/script_loop.sh"
ENV_NAME="loop_tf"

# Create the session if it doesn't exist
tmux has-session -t "$SESSION_NAME" 2>/dev/null
if [ $? != 0 ]; then
  tmux new-session -d -s "$SESSION_NAME"
fi

# Activate the conda environment in session
tmux send-keys -t "$SESSION_NAME" "conda activate $ENV_NAME" C-m

# Send the command to the session
tmux send-keys -t "$SESSION_NAME" "$COMMAND" C-m