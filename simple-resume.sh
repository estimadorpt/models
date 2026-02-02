#!/bin/bash

# Check if a prompt was provided
if [ -z "$1" ]; then
    echo "Usage: ./simple-resume.sh \"Your prompt here\""
    exit 1
fi

PROMPT="$1"

echo "🔄 Starting auto-resume loop for: '$PROMPT'"
echo "---------------------------------------------------"

while true; do
    # Get current timestamp for logs
    TIMESTAMP=$(date "+%H:%M:%S")
    
    # Try to execute the command directly. 
    # We use 'gtimeout' to prevent hanging, and '2>&1' to capture errors.
    # Note: We ALWAYS use -c to ensure we stay in the current conversation context.
    echo "[$TIMESTAMP] Attempting to run command..."
    OUTPUT=$(gtimeout 300s claude -c --dangerously-skip-permissions -p "$PROMPT" 2>&1)
    EXIT_CODE=$?

    # Check for usage limit (Case Insensitive flag -i is crucial here)
    if echo "$OUTPUT" | grep -iq "limit reached"; then
        echo "⚠️  USAGE LIMIT HIT"
        
        # Try to extract the reset time for the user's benefit
        RESET_TIME=$(echo "$OUTPUT" | grep -o "resets [0-9]*[ap]m")
        if [ ! -z "$RESET_TIME" ]; then
             echo "   (Claude says: $RESET_TIME)"
        fi
        
        echo "   Sleeping for 5 minutes before retrying..."
        sleep 300
    
    # If exit code is 0, it worked!
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "✅ SUCCESS!"
        echo "---------------------------------------------------"
        echo "$OUTPUT"
        break
        
    # If it failed for a reason OTHER than limits (e.g. network error)
    else
        echo "❌ Command failed with error (Code: $EXIT_CODE)"
        echo "Output:"
        echo "$OUTPUT"
        
        # Optional: Wait a bit before retrying network errors, or just exit?
        # Let's exit to be safe so we don't spam errors.
        exit 1
    fi
done
