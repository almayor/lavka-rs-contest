#!/bin/bash

# Directory paths - IMPORTANT: Update this to your actual local directory path
LOCAL_DIR="$PWD"  # Current directory, change this if needed
REMOTE_USER="majoroval"
REMOTE_HOST="imladris"
REMOTE_DIR="/data/majoroval/jupyter/RS-25/homework/week02"

# Exclusions
EXCLUDES="--exclude=.DS_Store --exclude=._.DS_Store --exclude=*.tmp --exclude=__pycache__/ --exclude=.ipynb_checkpoints/ --exclude=.git/ --exclude=.gitignore --exclude=.gitattributes --exclude=wandb/ --exclude=lightning_logs/ --exclude=*.git/*
    --exclude=*.zip"

# Size thresholds
MAX_SIZE_MB=100
MAX_FILES=1000

# Function to check large files
check_large_files() {
    local SOURCE=$1
    local DEST=$2
    local ACTION=$3
    
    echo "Analyzing largest files to be $ACTION..."
    if [[ $SOURCE == *":"* ]]; then
        # Remote source
        ssh ${SOURCE%%:*} "find ${SOURCE#*:} -type f -exec du -m {} \; | sort -nr | head -5" | awk '{printf "%8.2f MB  %s \n", $1, $2}'
    else
        # Local source
        find "$SOURCE" -type f -exec du -m {} \; | sort -nr | head -5 | awk '{printf "%8.2f MB  %s \n", $1, $2}'
    fi
}

# Check remote to local transfer (pull)
echo "Analyzing files to be pulled from remote..."
PULL_DRY_RUN=$(rsync -avzn --stats $EXCLUDES "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/")
PULL_FILES=$(echo "$PULL_DRY_RUN" | grep "Number of files transferred" | awk '{print $5}')
PULL_SIZE=$(echo "$PULL_DRY_RUN" | grep "Total transferred file size" | awk '{print $5}')

# Convert to MB if necessary
if [[ "$PULL_SIZE" == *"bytes"* ]]; then
    PULL_SIZE_MB=$(echo "scale=2; ${PULL_SIZE%% *}/1048576" | bc)
elif [[ "$PULL_SIZE" == *"KB"* ]]; then
    PULL_SIZE_MB=$(echo "scale=2; ${PULL_SIZE%% *}/1024" | bc)
elif [[ "$PULL_SIZE" == *"MB"* ]]; then
    PULL_SIZE_MB=${PULL_SIZE%% *}
elif [[ "$PULL_SIZE" == *"GB"* ]]; then
    PULL_SIZE_MB=$(echo "scale=2; ${PULL_SIZE%% *}*1024" | bc)
else
    PULL_SIZE_MB=$(echo "scale=2; $PULL_SIZE/1048576" | bc)
fi

echo "Files to pull: $PULL_FILES"
echo "Size to transfer: ${PULL_SIZE_MB}MB"

# Check if thresholds are exceeded for pull
if (( $(echo "$PULL_SIZE_MB > $MAX_SIZE_MB" | bc -l) )); then
    echo "Large file size detected in pull operation."
    check_large_files "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/" "pulled"
    read -p "Pull exceeds size threshold ($MAX_SIZE_MB MB). Continue? (y/n): " CONFIRM
    if [[ $CONFIRM != "y" ]]; then
        echo "Pull canceled."
        exit 1
    fi
fi

# Check local to remote transfer (push)
echo "Analyzing files to be pushed to remote..."
PUSH_DRY_RUN=$(rsync -avzn --stats $EXCLUDES "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/")
PUSH_FILES=$(echo "$PUSH_DRY_RUN" | grep "Number of files transferred" | awk '{print $5}')
PUSH_SIZE=$(echo "$PUSH_DRY_RUN" | grep "Total transferred file size" | awk '{print $5}')

# Convert to MB if necessary
if [[ "$PUSH_SIZE" == *"bytes"* ]]; then
    PUSH_SIZE_MB=$(echo "scale=2; ${PUSH_SIZE%% *}/1048576" | bc)
elif [[ "$PUSH_SIZE" == *"KB"* ]]; then
    PUSH_SIZE_MB=$(echo "scale=2; ${PUSH_SIZE%% *}/1024" | bc)
elif [[ "$PUSH_SIZE" == *"MB"* ]]; then
    PUSH_SIZE_MB=${PUSH_SIZE%% *}
elif [[ "$PUSH_SIZE" == *"GB"* ]]; then
    PUSH_SIZE_MB=$(echo "scale=2; ${PUSH_SIZE%% *}*1024" | bc)
else
    PUSH_SIZE_MB=$(echo "scale=2; $PUSH_SIZE/1048576" | bc)
fi

echo "Files to push: $PUSH_FILES"
echo "Size to transfer: ${PUSH_SIZE_MB}MB"

# Check if thresholds are exceeded for push
if (( $(echo "$PUSH_SIZE_MB > $MAX_SIZE_MB" | bc -l) )); then
    echo "Large file size detected in push operation."
    check_large_files "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "pushed"
    read -p "Push exceeds size threshold ($MAX_SIZE_MB MB). Continue? (y/n): " CONFIRM
    if [[ $CONFIRM != "y" ]]; then
        echo "Push canceled."
        exit 1
    fi
fi

# Perform the sync
echo "Starting two-way sync..."

# Pull - Get newest files from remote
echo "Pulling from remote..."
rsync -avz --progress --update $EXCLUDES "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"

# Push - Send newest files to remote
echo "Pushing to remote..."
rsync -avz --progress --update $EXCLUDES "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

echo "Two-way sync completed!"