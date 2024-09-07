import os
import time
import json
import random
import atexit
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
INITIAL_PROMPT = "a beautiful girl..."
LOOPS = 333
MAX_RETRIES = 5
PERSONA_TO_USE = 'Stephen King'
COSINE_SIMILARITY_THRESHOLD = 0.8  # Set the similarity threshold to retry
SUMMARY_COSINE_SIMILARITY_THRESHOLD = 0.6  # Similar threshold for summary updates

CONSTRAINT_REMINDER = "Remember, the response should be only 2 or 3 sentences with a maximum of 100 words in total."

PHASE_INSTRUCTIONS = {
    "beginning": "Establish characters and setting subtly. End with a captivating moment.",
    "middle": "Develop the plot and raise stakes without giving everything away. End with an uneasy anticipation.",
    "end": "Subtly wrap up the narrative while leaving thematic elements open to interpretation."
}

USER_MESSAGE_TEMPLATE = (
    "We are writing a story together in the style of {persona}. "
    "Continue the following story creatively, making bold assumptions about what could happen next. "
    "Address core issues of the storyline and transition smoothly to the next scene. "
    "Maintain an imaginative style fitting {persona}'s narrative while keeping responses to "
    "2 or 3 sentences and a maximum of 100 words. "
    "Each response should imply {ending}. The current story is: {current_story}. "
    "Here is a summary of the story so far: {summary}. "
    "{phase_instructions} "
    + CONSTRAINT_REMINDER
)

SUMMARY_UPDATE_TEMPLATE = (
    "Here is the current summary of the story: \"{current_summary}\". "
    "The latest addition to the story is: \"{latest_addition}\". "
    "Can you enhance the overall summary with it without changing and limiting the overall summary to 4-5 sentences "
    "and 750 characters? Ensure the summary encourages someone to read more. If the new addition adds no value, "
    "don't change it. Only return the revised summary in your response, nothing else."
)

COMPLETE_SYNOPSIS_TEMPLATE = (
    "Please read the following lines from the story: \"{selected_lines}\" and the following summary: \"{summary}\". "
    "Using these, create a captivating synopsis that reads like the cover of a book, enticing someone to read the entire story. "
    "Keep the synopsis limited to 6-7 sentences and less than 900 characters. Return only the synopsis, nothing else."
)

CHARACTER_DESCRIPTION_TEMPLATE = "Create and describe in detail the main character 250 characters or less, reply with ONLY the description. Here is a complete synopsis: {complete_synopsis}."

# Ensure the directory for saving JSON files exists
output_dir = "storylines"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
JSON_FILE = os.path.join(output_dir, f"{timestamp}_story.json")

def get_story_context(current_story, initial_prompt, retry_count):
    """Generate context for the story based on retry count."""
    if retry_count == 0:
        return " ".join([initial_prompt, *current_story[-2:]])
    elif retry_count == 1:
        return " ".join([initial_prompt, *current_story[-3:]])
    elif retry_count == 2:
        return " ".join(current_story[-1:])
    elif retry_count == 3:
        return " ".join([initial_prompt, current_story[-1]])
    return None

def generate_summary(current_story):
    """Generate a summary of the current story in 2-3 sentences."""
    full_text = " ".join(current_story)
    sentences = full_text.split(". ")
    summary = ". ".join(sentences[:3]).strip()
    if not summary.endswith("."):
        summary += "."
    summary = summary if len(summary) <= 750 else summary[:747] + "..."
    return ensure_proper_ending(summary)

def ensure_proper_ending(summary):
    """Ensure the summary ends correctly with common abbreviations and exactly three ellipses."""
    abbreviations = ["Dr.", "Mr.", "Ms.", "Mrs.", "Jr.", "Sr.", "St.", "etc."]
    for abbr in abbreviations:
        if summary.endswith(abbr):
            return summary + "..."
    # Ensure only three ellipses at the end
    return summary.rstrip(".") + "..."

def get_phase(loop_index, total_loops):
    """Determine the phase of the story based on the current loop index."""
    if loop_index < total_loops * 0.25:
        return "beginning"
    elif loop_index < total_loops * 0.9:
        return "middle"
    else:
        return "end"

def calculate_cosine_similarity(text1, text2):
    """Calculate the cosine similarity between two texts."""
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def enhance_summary(current_summary, latest_addition):
    """Enhance the overall summary with the latest story addition."""
    summary_prompt = SUMMARY_UPDATE_TEMPLATE.format(current_summary=current_summary, latest_addition=latest_addition)
    enhanced_summary = get_story_response_from_model(MODEL_NAME, summary_prompt).strip()
    return enhanced_summary

def generate_complete_synopsis(current_story, final_summary):
    """Generate a complete synopsis from selected lines in the story."""
    if len(current_story) < 8:
        selected_lines = current_story
    else:
        selected_lines = []
        selected_lines.extend(current_story[:3])  # First 3 lines
        selected_lines.extend(current_story[-3:])  # Last 3 lines
        remaining_indexes = list(range(3, len(current_story) - 3))
        random.shuffle(remaining_indexes)
        selected_lines.extend([
            current_story[remaining_indexes[0]],
            current_story[remaining_indexes[1]]
        ])  # 2 random lines in the middle

    selected_lines_text = " ".join(selected_lines)
    synopsis_prompt = COMPLETE_SYNOPSIS_TEMPLATE.format(selected_lines=selected_lines_text, summary=final_summary)
    complete_synopsis = get_story_response_from_model(MODEL_NAME, synopsis_prompt).strip()
    
    print("\nSelected lines for final synopsis:\n", json.dumps(selected_lines, indent=2))
    print("\nFinal story summary included in the synopsis:\n", final_summary)
    print("\nGenerated complete synopsis:\n", complete_synopsis)
    
    return complete_synopsis

def write_story_segment(model_name, prompt, loops, json_file):
    """Generate story segments and save them to a file."""
    current_story = [prompt]
    overall_summary = generate_summary(current_story)
    
    initial_data = {
        "story_chapters": current_story,
        "story_summary": overall_summary
    }

    # Initialize the JSON file with the initial data
    with open(json_file, 'w') as f:
        json.dump(initial_data, f, indent=2)

    for loop_index in range(loops):
        with open(json_file, 'r') as f:
            data = json.load(f)
            current_story = data["story_chapters"]
            overall_summary = data["story_summary"]

        retry_count = 0
        phase = get_phase(loop_index, loops)
        
        if phase == "beginning":
            ending = "an intriguing moment"
        elif phase == "middle":
            ending = "an insight into what might unfold"
        else:  # phase == "end"
            ending = "a resolution with a lingering question"
        
        phase_instructions = PHASE_INSTRUCTIONS[phase]
        
        print(f"\n{'!' * 10} Currently in the {phase} phase of the story, loop: {loop_index + 1}/{loops} ({((loop_index + 1) / loops) * 100:.2f}%) {'!' * 10}\n")

        while retry_count <= MAX_RETRIES:
            current_story_text = get_story_context(current_story, prompt, retry_count)
            if current_story_text is None:
                print("Exhausted all retry mechanisms. Stopping...")
                return current_story

            user_message = USER_MESSAGE_TEMPLATE.format(
                current_story=current_story_text, persona=PERSONA_TO_USE,
                summary=overall_summary, ending=ending,
                phase_instructions=phase_instructions
            )

            print(f"\n" + "*" * 40)
            print("**** SENDING IN TO ADD TO THE STORYLINE ****")
            print("*" * 40)
            print(f"{user_message}")
            print("*" * 40 + "\n")

            response = get_story_response_from_model(model_name, user_message)

            if response:
                next_line = response.strip()
                print("\n" + "*" * 40)
                print("*" * 40)
                print(f"****\n\n{next_line}\n\n****")
                print("*" * 40)
                print("*" * 40 + "\n")

                # Check for duplicates using cosine similarity with the last 3 entries
                is_duplicate = False
                for idx, previous_line in enumerate(current_story[-3:], len(current_story) - 3):
                    similarity_score = calculate_cosine_similarity(next_line, previous_line)
                    print(f"Checking similarity between current line and previous line {idx + 1}: {similarity_score}")
                    print(f"Current line: {next_line}")
                    print(f"Previous line {idx + 1}: {previous_line}")
                    if similarity_score > COSINE_SIMILARITY_THRESHOLD:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    current_story.append(next_line)
                    previous_summary = overall_summary
                    overall_summary = enhance_summary(overall_summary, next_line)
                    
                    # Calculate similarity score for summaries
                    similarity_score = calculate_cosine_similarity(previous_summary, overall_summary)
                    print(f"++++++++++++++++++++++++++++++++++++++++")
                    print(f"++++++++++++++++++++++++++++++++++++++++")
                    print(f"++++\n")

                    print(f"The new update to the storyline has a cosine similarity score of {similarity_score:.4f}.\n")
                    print(f"Cosine similarity measures how similar two sequences of text are by representing them as vectors in a high-dimensional space and calculating the cosine of the angle between these vectors.\n")
                    print(f"We have a threshold set of: {SUMMARY_COSINE_SIMILARITY_THRESHOLD}.\n")
                    
                    if similarity_score > SUMMARY_COSINE_SIMILARITY_THRESHOLD:
                        print(f"That means that it was NOT to be changed based on the score.")
                    else:
                        print(f"That means that it was TO be changed based on the score.")
                    
                    print(f"\nHere is the new summary, regardless:\n")
                    print(f"{overall_summary}")
                    print(f"++++")
                    print(f"++++++++++++++++++++++++++++++++++++++++")
                    print(f"++++++++++++++++++++++++++++++++++++++++")

                    # Write the updated story and summary back to the JSON file
                    with open(json_file, 'w') as f:
                        json.dump({"story_chapters": current_story, "story_summary": overall_summary}, f, indent=2, ensure_ascii=False)
                    
                    break
                else:
                    print(f"\n" + "=" * 40)
                    print("=" * 40)
                    print(f"====== Duplicate response detected, retrying... ======")
                    print("=" * 40)
                    print("=" * 40 + "\n")
                    retry_count += 1
                    time.sleep(1)
            else:
                print(f"Failed to get a response from the model.")
                break

    complete_synopsis = generate_complete_synopsis(current_story, overall_summary)

    # Generate the main character description based on the complete synopsis
    main_character_prompt = CHARACTER_DESCRIPTION_TEMPLATE.format(complete_synopsis=complete_synopsis)
    character_description = get_story_response_from_model(model_name, main_character_prompt).strip()
    with open(json_file, 'r') as f:
        data = json.load(f)
    data["complete_synopsis"] = complete_synopsis
    data["main_character"] = character_description

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nOverall synopsis (complete synopsis and final story summary) saved to {json_file}.")
    return current_story

def main():
    global MODEL_NAME, INITIAL_PROMPT, LOOPS, JSON_FILE

    start_time = time.time()

    kill_existing_ollama_service()
    clear_gpu_memory()

    install_and_setup_ollama(MODEL_NAME)

    if is_windows():
        start_ollama_service_windows()
        time.sleep(10)

    write_story_segment(MODEL_NAME, INITIAL_PROMPT, LOOPS, JSON_FILE)

    stop_ollama_service()
    clear_gpu_memory()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    atexit.register(stop_ollama_service)
    atexit.register(clear_gpu_memory)
    main()