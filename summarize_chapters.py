import os
import time
import json
import atexit
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

# GLOBAL VARIABLES #
MODEL_NAME = 'llama3'
DIRECTORY_PATH = 'storylines'  # Directory where the JSON file is created (default is current directory)

SUMMARY_REQUEST_TEMPLATE = "Please summarize the following line in 10 words or less: \"{line}\""

def find_latest_non_summarized_json_file(directory_path):
    """Find the latest non-summarized JSON file in the specified directory."""
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json') and "_10_word_chapter_summaries" not in f]
    if not json_files:
        raise FileNotFoundError("No non-summarized JSON files found in the specified directory.")
    latest_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(directory_path, f)))
    return os.path.join(directory_path, latest_file)

def summarize_line(model_name, line):
    """Summarize a single line using the model."""
    summary_prompt = SUMMARY_REQUEST_TEMPLATE.format(line=line)
    summary = get_story_response_from_model(model_name, summary_prompt).strip()
    return summary

def summarize_story_chapters(json_file_path, model_name):
    """Summarize each chapter in the story and save as summaries."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    story_chapters = data.get("story_chapters", [])
    main_character = data.get("main_character", "")
    story_summary = data.get("story_summary", "")
    summarized_chapters = []

    for index, chapter in enumerate(story_chapters):
        print(f"Summarizing chapter {index + 1}/{len(story_chapters)}")
        chapter_summary = summarize_line(model_name, chapter)
        summarized_chapters.append({
            "chapter": chapter,
            "chapter_summary": chapter_summary
        })

    # Create the new JSON structure
    summarized_data = {
        "story_chapters": summarized_chapters,
        "story_summary": story_summary,
        "main_character": main_character
    }

    # Extract the base name of the JSON file and append _10_word_chapter_summaries.json
    base_name = os.path.basename(json_file_path)
    summary_file_name = f"{base_name.split('.')[0]}_10_word_chapter_summaries.json"
    summary_output_path = os.path.join(os.path.dirname(json_file_path), summary_file_name)

    with open(summary_output_path, 'w') as f:
        json.dump(summarized_data, f, indent=2, ensure_ascii=False)

    print(f"Summaries saved to {summary_output_path}")

def main():
    global MODEL_NAME, DIRECTORY_PATH

    start_time = time.time()

    kill_existing_ollama_service()
    clear_gpu_memory()

    install_and_setup_ollama(MODEL_NAME)

    if is_windows():
        start_ollama_service_windows()
        time.sleep(10)

    # Find the latest non-summarized JSON file in the specified directory
    latest_json_file = find_latest_non_summarized_json_file(DIRECTORY_PATH)
    print(f"Processing latest non-summarized JSON file: {latest_json_file}")

    # Summarize the story chapters in the latest JSON file
    summarize_story_chapters(latest_json_file, MODEL_NAME)

    stop_ollama_service()
    clear_gpu_memory()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    atexit.register(stop_ollama_service)
    atexit.register(clear_gpu_memory)
    main()