#!/usr/bin/env python3
"""
Query tool for BBQ and StereoSet evaluation runs.

This script provides a simple interface to query and filter
evaluation runs from the tracking system.

Examples:
    # Show 5 latest BBQ runs
    python query_runs.py bbq --latest 5
    
    # Show all StereoSet runs with a specific model
    python query_runs.py stereoset --model "gpt-4"
    
    # Show details for a specific run
    python query_runs.py bbq --run-id "bbq_20250511_161948_mock_model"
"""

from common.tracking import main

if __name__ == "__main__":
    main()
