import os
import json
import time
import random
import atexit
from glob import glob
from datetime import datetime
from ollama_utils import (
    install_and_setup_ollama,
    kill_existing_ollama_service,
    clear_gpu_memory,
    start_ollama_service_windows,
    stop_ollama_service,
    is_windows,
    get_story_response_from_model
)

MODEL_NAME = 'llama3'
MAX_TOKENS = 70
INITIAL_PROMPT = ("Shorten the following scene description to 200 characters or less retaining as much content as you can. "
                  "ONLY respond with the shortened version and nothing else.")
POSE_JSON_FILE = "pose.json"

def ensure_initial_json_structure(file_path):
    initial_structure = {"activity": []}
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump(initial_structure, f, indent=2)
    else:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if "activity" not in data:
                    raise ValueError("Invalid JSON structure")
            except (json.JSONDecodeError, ValueError):
                with open(file_path, 'w') as f:
                    json.dump(initial_structure, f, indent=2)

def get_latest_story_json_file(directory):
    """Get the latest _story.json file in the specified directory."""
    json_files = glob(os.path.join(directory, "*_story.json"))
    latest_file = max(json_files, key=os.path.getmtime)
    return latest_file

def send_line_to_ollama(model_name, line):
    retry_count = 0
    while retry_count < 5:
        try:
            response = get_story_response_from_model(model_name,
                f"{INITIAL_PROMPT} Scene: {line}")
            if response:
                # Return the response text trimmed of any surrounding whitespace.
                return response.strip()
            else:
                retry_count += 1
                print(f"Retrying... ({retry_count})")
                time.sleep(1)
        except Exception as e:
            print(f"Error: {str(e)}")
            retry_count += 1
            time.sleep(1)
    return None

def main():
    start_time = time.time()

    kill_existing_ollama_service()
    clear_gpu_memory()
    install_and_setup_ollama(MODEL_NAME)

    if is_windows():
        start_ollama_service_windows()
        time.sleep(10)

    # Delete any existing pose.json file
    if os.path.exists(POSE_JSON_FILE):
        os.remove(POSE_JSON_FILE)
        print(f"Deleted existing {POSE_JSON_FILE}")

    # Ensure initial JSON structure
    ensure_initial_json_structure(POSE_JSON_FILE)

    # Discover the latest JSON file
    directory = os.getcwd()
    latest_json_file = get_latest_story_json_file(directory)
    print(f"Latest JSON file found: {latest_json_file}")

    # Read initial story
    with open(latest_json_file, 'r') as f:
        data = json.load(f)
        story_chapters = data.get("story_chapters", [])

    for line in story_chapters:
        shortened_description = send_line_to_ollama(MODEL_NAME, line)
        if shortened_description:
            with open(POSE_JSON_FILE, 'r') as f:
                existing_data = json.load(f)
            existing_data["activity"].append(shortened_description)
            with open(POSE_JSON_FILE, 'w') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

    stop_ollama_service()
    clear_gpu_memory()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    atexit.register(stop_ollama_service)
    atexit.register(clear_gpu_memory)
    main()