import os
import math
import pdb
from MCTS.mcts_node import MCTSNode
import pickle
import numpy as np
import json
from typing import Callable, Any, List, Optional
from android_world.env.representation_utils import UIElement
from html_representation.html_representation import turn_tree_to_html_input_v2
from prompt_template import *
import time
import subprocess
import re
import gzip
from tqdm import tqdm


def disable_key_board(emulator_id="emulator-5554"):
    commands = [
        f"adb -s {emulator_id} shell pm disable-user com.google.android.inputmethod.latin",
        f"adb -s {emulator_id} shell pm disable-user com.google.android.tts"
    ]
    for command in commands:
        try:
            subprocess.run(
                command, shell=True, check=True, text=True, capture_output=True
            )
            print(f"Command succeeded: {command}")
        except subprocess.CalledProcessError:
            print(f"Command failed: {command}")

    return


class StepwiseDataSample:
    def __init__(self, goal, description, historical_actions, action, q_value, family='android_world'):
        """
        Initialize a stepwise data sample.

        :param goal: The goal description for the task.
        :param description: Description of the current step or state.
        :param historical_actions: List of actions taken to reach this step.
        :param action: The action taken at this step.
        :param q_value: Q value (reward or evaluation score) for the action.
        """
        self.goal = goal
        self.description = description
        self.q_value = q_value
        self.family = family

        if historical_actions:
            history = [
                f"Step {i+1}- {action}"
                for i, action in enumerate(historical_actions)
            ]
            self.historical_actions = "\n".join(history)
        else:
            self.historical_actions = "You just started, no action has been performed yet."

        self.action = action

    def to_dict(self):
        """Convert the stepwise data sample to a dictionary for saving or exporting."""
        return {
            "goal": self.goal,
            "description": self.description,
            "historical_actions": self.historical_actions,
            "action": self.action,
            "q_value": self.q_value,
        }

    def to_template(self):
        """Convert the stepwise sample to a formatted string using a template."""
        prompt_prefix =  PROMPT_PREFIX_V2
        return SELF_EVAL_TEMPLATE_VERIFIER_TRAINING_V3.format(
            prompt_prefix=prompt_prefix,
            goal=self.goal,
            history=self.historical_actions,
            before_elements=self.description,
            action=self.action,
        )

    # def extract_prompt(self):
    #     return T3A_TRAINING.format(
    #         goal=self.goal,
    #         history=self.historical_actions,
    #         before_elements=self.description,
    #     )
    
    # def extract_prompt(self):
    #     return T3A_TRAINING.format(
    #         goal=self.goal,
    #         history=self.historical_actions,
    #         before_elements=self.description,
    #     )
    

class PairwiseDataSample:
    def __init__(self, correct_sample: StepwiseDataSample, rejected_sample: StepwiseDataSample):
        """
        Initialize a pairwise data sample with a correct action and a rejected action.

        :param correct_sample: A StepwiseDataSample representing the correct action.
        :param rejected_sample: A StepwiseDataSample representing the rejected action.
        """
        self.correct_sample = correct_sample
        self.rejected_sample = rejected_sample

    def to_dict(self):
        """Convert the pairwise sample to a dictionary for saving or exporting."""
        return {
            "chosen": self.correct_sample.to_dict(),
            "rejected": self.rejected_sample.to_dict(),
        }

    def to_chosen_rejected(self):
        """
        Convert the pairwise sample to a dataset format for training.

        :param template: A string template for formatting.
        """
        chosen = self.correct_sample.to_template()
        rejected = self.rejected_sample.to_template()
        # pdb.set_trace()
        return {"chosen": chosen, "rejected": rejected}
    

    def to_dpo_training(self):
        prompt = self.correct_sample.extract_prompt()
        chosen_action = self.correct_sample.action
        rejected_action = self.rejected_sample.action
        return {"prompt": prompt, "chosen": chosen_action, "rejected": rejected_action}


class DPODataSample:
    def __init__(self, sample: StepwiseDataSample, correctness: bool):
        """
        Initialize a pairwise data sample with a correct action and a rejected action.

        :param sample: A StepwiseDataSample
        :param correctness: A bool var to indicate whether the corresponding action is correct or not
        """
        self.sample = sample
        self.correctness = correctness

    def to_dpo(self):
        prompt = self.sample.to_template()
        chosen = "Yes" if self.correctness else "No"
        rejected = "No" if self.correctness else "Yes"
        
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    

class SelectorDataSample:
    """
    A data sample that provides a list of possible answers (actions) 
    for a particular step (including the correct one) so a selector model 
    can choose the correct answer from the list.
    """

    def __init__(
        self,
        goal: str,
        description: str,
        historical_actions: list[str],
        action_lists: list[str],
        correct_answer: str,
        reason: str = None,
    ):
        """
        :param goal: The goal description for the task.
        :param description: Description of the current step or state (HTML or text).
        :param historical_actions: List of actions taken to reach this step.
        :param action_lists: A list of possible actions (including the correct one).
        :param correct_answer: The correct action among action_lists.
        """
        self.goal = goal
        self.description = description
        self.action_lists = action_lists
        self.correct_answer = correct_answer
        self.reason = reason

        if historical_actions:
            history = [
                f"Step {i+1}- {action}"
                for i, action in enumerate(historical_actions)
            ]
            self.historical_actions = "\n".join(history)
        else:
            self.historical_actions = "You just started, no action has been performed yet."

        action_list_str = [f'{i}: {action}' for i, action in enumerate(action_lists)]
        self.action_lists = "\n".join(action_list_str)

    def to_dict(self):
        """
        Convert this selector data sample into a dictionary format.
        Useful for saving or exporting data.
        """
        return {
            "goal": self.goal,
            "description": self.description,
            "historical_actions": self.historical_actions,
            "action_lists": self.action_lists,
            "correct_answer": self.correct_answer,
        }
    
    # def to_prompt(self):
    #     prompt = SELECTOR_TRAINING.format(
    #         goal=self.goal,
    #         history=self.historical_actions,
    #         before_elements=self.description,
    #         action_lists=self.action_lists,
    #     )

    #     return {"prompt": prompt, "completion": self.correct_answer}
    
    # def to_policy(self):
    #     prompt = POLICY_TRAINING.format(
    #         goal=self.goal,
    #         history=self.historical_actions,
    #         before_elements=self.description,
    #     )
    #     answer = f"Reason: {self.reason}\nAction: {self.correct_answer}"
    #     return {"prompt": prompt, "completion": answer}


def dfs_max_reward(path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
    cur = path[-1]
    if cur.is_terminal:
        return sum([node.reward for node in path[1:]]), path
    if cur.children is None:
        return -math.inf, path
    visited_children = [x for x in cur.children if x.node_info is not None]
    if len(visited_children) == 0:
        return -math.inf, path
    return max((dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])


# Custom Unpickler to replace old references with new ones
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "android_world.agents.sa" and name == "MCTSNode":
            # Redirect to the new class
            return MCTSNode
        return super().find_class(module, name)


class ActionStack:
    """
    A stack that stores actions' reverse operations for potential backtracking.
    """
    def __init__(self):
        # We store the *reverse* actions in a stack (list).
        self.stack: List[str] = []

    def push_reverse(self, action):
        """
        Push an Action (which should be the reverse operation) onto the stack.
        """
        self.stack.append(action)

    def pop_reverse(self):
        """
        Pop the most recent Action (the top of the stack) and execute its reverse_fn.
        """
        if not self.stack:
            print("Reverse stack is empty; nothing to undo.")
            return
        reverse_action = self.stack.pop()
        return reverse_action

    def size(self):
        return len(self.stack)



def entropy_estimation(scores):
    max_score = np.max(scores)
    shifted_scores = [s - max_score for s in scores]
    exp_scores = np.exp(shifted_scores)
    probs = exp_scores / np.sum(exp_scores)
    # Add a small epsilon to avoid log(0)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return entropy

def score_difference(score):
    sorted_scores = np.sort(score)[::-1] 
    score_diff = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    return score_diff



def obtain_reversed_action(action):
    action = json.loads(action)
    action_type = action.get("action_type")
    
    if action_type in ["click", "long_press"]:
        correct_action = {"action_type": "navigate_back"}
    elif action_type == "open_app":
        correct_action = {"action_type": "navigate_home"}
    elif action_type == "scroll":
        direction = action.get("direction")
        if direction == "up":
            correct_action = {"action_type": "scroll", "direction": "down"}
        elif direction == "down":
            correct_action = {"action_type": "scroll", "direction": "up"}
        elif direction == "left":
            correct_action = {"action_type": "scroll", "direction": "right"}
        elif direction == "right":
            correct_action = {"action_type": "scroll", "direction": "left"}
        else:
            correct_action = None
    elif action_type == "input_text":
        correct_action = {"action_type": "clear_text", "index": "<target_index>"}
        correct_action["index"] = action.get("index")
    else:
        correct_action = None
    return json.dumps(correct_action) if correct_action else None


def refresh_env_with_retries(env, max_retries=10, delay=1):
    attempt = 0
    while attempt < max_retries:
        try:
            # Attempt to refresh the environment
            env._controller.refresh_env()
            print("Environment refreshed successfully.")
            return True  # Return True if successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            time.sleep(delay)  # Optionally add a delay between retries

    print(f"Failed to refresh environment after {max_retries} attempts.")
    return False  # Return False if all attempts fail


def load_snapshots(env, family = "android_lab"):
    # command = 'adb -s emulator-5554 emu avd snapshot load "android-lab"'
    if family == "android_lab":
        command = 'docker exec -it mycontainer /root/.android/platform-tools/adb -s emulator-5554 emu avd snapshot load "android-lab"'
    elif family == "android_world":
        command = 'adb -s emulator-5554 emu avd snapshot load "android_world"'
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"Command succeeded: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
    time.sleep(60.0)
    # super().initialize_task(env)
    # contacts_utils.clear_contacts(env.controller)
    
    success = refresh_env_with_retries(env, )
    if success:
        print("Environment refresh was successful.")
    else:
        print("Failed to refresh the environment after multiple attempts.")

    return


def polish_summary(summary):
    # Split the text into sentences
    cleaned_text = re.sub(r'Action History:.*?(?=(?:Reason|$))', '', summary, flags=re.DOTALL)

    sentences = re.split(r'(?<=[.!?]) +', cleaned_text)

    # Use a set to track unique sentences
    unique_sentences = []
    seen = set()

    for sentence in sentences:
        if sentence.endswith(('.', '!', '?')) and sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)

    # Join the unique sentences back into a cleaned-up text
    cleaned_text = " ".join(unique_sentences)

    return cleaned_text

def polish_action(action: str) -> str:
    action.replace("target_index", "index")
    action.replace("target_element_index", "index")
    action.replace("input_field_index", "index")
    action.replace("target_app", "app_name")

    # pattern = r'(\{.*?\})'
    pattern = r'\{.*?\}'
    # Search for the JSON-like pattern and return it, ignoring the rest
    match = re.search(pattern, action)
    if match:
        cleaned_action = match.group(0)
        return cleaned_action
    return action


def polish_reason(reasoning_text):
    # Split the text into sentences
    cleaned_text = re.sub(r'Action History:.*?(?=(?:Reason|$))', '', reasoning_text, flags=re.DOTALL)

    sentences = re.split(r'(?<=[.!?]) +', cleaned_text)

    # Use a set to track unique sentences
    unique_sentences = []
    seen = set()

    for sentence in sentences:
        if sentence.endswith(('.', '!', '?')) and sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)

    # Join the unique sentences back into a cleaned-up text
    cleaned_text = " ".join(unique_sentences)

    cleaned_text = re.split(r'Action:', cleaned_text, maxsplit=1)[0]
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def warm_up(llm):
    warm_up_text = PROMPT_PREFIX_V2 + 'The (overall) user goal/request is: '
    llm.predict_scores_batch([warm_up_text]) 
    return

def get_element_text(elem: UIElement) -> str:
    """
    Extract a 'representative' text from a UI element. 
    You can tailor this to your own usage:
    - If the element has `text`, use that.
    - Otherwise check `content_description`, `resource_name`, etc.
    """
    if elem.text:
        return elem.text.strip()
    elif elem.content_description:
        return elem.content_description.strip()
    elif elem.hint_text:
        return elem.hint_text.strip()
    elif elem.resource_name:
        return elem.resource_name.split('/')[-1].strip()
    return ""

def find_element_by_index(ui_list: List[UIElement], index: int) -> Optional[UIElement]:
    """Fetch the UI element from the list by numeric index, if valid."""
    if 0 <= index < len(ui_list):
        return ui_list[index]
    return None


def generate_step_summary(
    action_dict: dict,
    before_ui_list: List[UIElement],
    after_ui_list: List[UIElement],
) -> str:
    """
    Given:
      - the action just performed (as a dict),
      - a list of UI elements before the action,
      - a list of UI elements after the action,
    produce a short, single-line summary of what happened.
    """
    action_type = action_dict.get("action_type")
    summary_parts = []

    if action_type == "answer":
        text = action_dict.get("text", "")
        summary_parts.append(f"Answered the user's question by stating that {text}")

    elif action_type == "click":
        idx = action_dict.get("index", None)
        if idx is not None:
            elem = find_element_by_index(before_ui_list, idx)
            elem_text = get_element_text(elem) if elem else ""
            if elem_text:
                summary_parts.append(f"Clicked on the \"{elem_text}\" button")
            else:
                summary_parts.append(f"Clicked the \"{idx}\" button")
        else:
            summary_parts.append("Clicked one element")

    elif action_type == "long_press":
        idx = action_dict.get("index", None)
        if idx is not None:
            elem = find_element_by_index(before_ui_list, idx)
            elem_text = get_element_text(elem) if elem else ""
            if elem_text:
                summary_parts.append(f"Long-pressed the \"{elem_text}\" button")
            else:
                summary_parts.append(f"Long-pressed button {idx}")
            summary_parts.append("which brought up further options")
        else:
            summary_parts.append("Long-pressed one button")

    elif action_type == "input_text":
        idx = action_dict.get("index", None)
        input_text = action_dict.get("text", "")
        if idx is not None:
            elem = find_element_by_index(before_ui_list, idx)
            elem_text = get_element_text(elem) if elem else ""
            if elem_text:
                summary_parts.append(f"Typed \"{input_text}\" into \"{elem_text}\" field")
            else:
                summary_parts.append(f"Typed \"{input_text}\" into element at index {idx}")
            summary_parts.append("indicating the input was accepted")
        else:
            summary_parts.append(f"Typed \"{input_text}\"")

    elif action_type == "keyboard_enter":
        summary_parts.append("Pressed Enter key, finalizing the input")

    elif action_type == "navigate_home":
        summary_parts.append("Navigated to home screen")

    elif action_type == "navigate_back":
        summary_parts.append("Navigated back to the previous page")

    elif action_type == "scroll":
        direction = action_dict.get("direction", "")
        idx = action_dict.get("index", None)
        if idx is not None:
            summary_parts.append(f"Scrolled element (index {idx}) {direction} to reveal more content")
        else:
            summary_parts.append(f"Scrolled screen {direction} to reveal more content")

    elif action_type == "open_app":
        app_name = action_dict.get("app_name", "")
        summary_parts.append(f"Opened {app_name} successfully")

    elif action_type == "wait":
        summary_parts.append("Waited for the screen update")
    
    else:
        summary_parts.append("Performed an unrecognized action")

    after_texts = [get_element_text(elem).lower() for elem in after_ui_list if elem]

    if "delete" in after_texts:
        summary_parts.append("a 'Delete' option is now shown")
    if "confirmation" in after_texts or "yes" in after_texts:
        summary_parts.append("a confirmation option is visible")
    
    summary = ", ".join(summary_parts) + "."
    return summary


if __name__ == "__main__":
    data_directory = "saved/sa_gpt_llama31_Round110k/task_info/"
    for file_name in tqdm(os.listdir(data_directory)):
        file_path = os.path.join(data_directory, file_name)
        tasks_name = file_name.split('_')[0]

        # data_path = "saved/sa_gpt_llama31_Round110k/task_info/SystemBrightnessMinVerify_0.pkl.gz"
        with gzip.open(file_path, "rb") as f:
            load_data = pickle.load(f)
        # pdb.set_trace()
        root = load_data[0]['tree']
        cum_reward, correct_trace = dfs_max_reward([root])
        # print(load_data[0]['tree'].children[0].fast_reward)
        # print(correct_trace)
        # print(cum_reward)
        # pdb.set_trace()
        data_samples = []
        whole_action_list_for_trace = []
        whole_rejected_action_list_for_trace = []
        whole_correct_action_list_for_trace = []
        complete_count = 0
        for node in correct_trace[1:]:
            if not node.children:
                continue
            
            node.node_info["ui_elements"]
            action = json.loads(node.action)
            # pdb.set_trace()
            summary = generate_step_summary(action, node.parent.node_info["ui_elements"], node.node_info["ui_elements"], )
            print(summary)


def extract_content_from_text(directory=None):
    with open(directory, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(
        r"Task \(from User\):\s*(?P<task>.*?)\n\n"                # everything up to the blank line before Action History
        r"Action History \(from Policy Agent\):\s*(?P<action_history>.*?)\n\n"
        r"Monitored Information \(from Evaluator Agent while observing each Policy Agent step, newest last\):\s*(?P<monitored_info>.*?)\n\n"
        r"Final UI State \(raw HTML, captured by Evaluator Agent\):\s*(?P<final_ui>.*?)\n\n"
        r"Additional UI Pages \(raw HTML, captured by Evaluator Agent, newest last\):\s*"
        r"(?P<additional_pages>.*?)(?=\n\nRespond in this exact format:)",
        re.DOTALL,
    )

    match = pattern.search(text)
    if not match:
        raise ValueError("Could not match all sections; check that the headers are exactly as expected.")

    task                = match.group("task").strip()
    action_history      = match.group("action_history").strip()
    monitored_info      = match.group("monitored_info").strip()
    final_ui_state      = match.group("final_ui").strip()
    additional_ui_pages = match.group("additional_pages").strip()

    return task, action_history, monitored_info, final_ui_state, additional_ui_pages



def extract_last_action_history(log_path):
    """
    Extract the last block of action history from an app.log file.

    Args:
        log_path (str): path to the app.log file.

    Returns:
        str: the last action history block (steps text), or None if not found.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex: capture the block until next header, UI info section, or EOF
    history_blocks = re.findall(
        r"Here is the history of actions taken:\s*\n(.*?)(?=\nHere is the history of actions taken:|\nHere is the detailed information about the UI|$)",
        content,
        flags=re.DOTALL
    )

    if not history_blocks:
        return None

    return history_blocks[-1].strip()

def load_trajectory(file_path: str):
    with gzip.open(file_path, 'rb') as f:
        trajectory = pickle.load(f)
    return trajectory

def save_trajectory(file_path: str, trajectory):
    with gzip.open(file_path, "wb") as f:
        pickle.dump(trajectory, f)

def no_step_infor(trace):
    for node in trace:
        if "ui_snapshot_note" not in node.node_info:
            return True
    return False

def split_additional_ui_pages(pages_text: str):
    """
    Split concatenated additional_ui_pages string into per-step descriptions.
    Removes empty/blank steps.
    """
    chunks = re.split(r"(Step \d+ - UI Page:)", pages_text)
    steps, current = [], ""
    for chunk in chunks:
        if chunk.startswith("Step "):
            if current.strip():  # save non-empty
                steps.append(current.strip())
            current = chunk
        else:
            current += " " + chunk
    if current.strip():
        steps.append(current.strip())

    return [s for s in steps if s and not s.endswith("UI Page:")]


def extract_content_from_text(directory=None):
    with open(directory, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(
        r"Task \(from User\):\s*(?P<task>.*?)\n\n"                # everything up to the blank line before Action History
        r"Action History \(from Policy Agent\):\s*(?P<action_history>.*?)\n\n"
        r"Monitored Information \(from Evaluator Agent while observing each Policy Agent step, newest last\):\s*(?P<monitored_info>.*?)\n\n"
        r"Final UI State \(raw HTML, captured by Evaluator Agent\):\s*(?P<final_ui>.*?)\n\n"
        r"Additional UI Pages \(raw HTML, captured by Evaluator Agent, newest last\):\s*"
        r"(?P<additional_pages>.*?)(?=\n\nRespond in this exact format:)",
        re.DOTALL,
    )

    match = pattern.search(text)
    if not match:
        raise ValueError("Could not match all sections; check that the headers are exactly as expected.")

    task                = match.group("task").strip()
    action_history      = match.group("action_history").strip()
    monitored_info      = match.group("monitored_info").strip()
    final_ui_state      = match.group("final_ui").strip()
    additional_ui_pages = match.group("additional_pages").strip()

    return task, action_history, monitored_info, final_ui_state, additional_ui_pages


def mask_irrelevant_info(html: str) -> str:
    # Mask time (e.g., 15:40, 1:30 PM)
    html = re.sub(r'\b\d{1,2}:\d{2}(?:\s?[APap][Mm])?\b', '<TIME>', html)

    # Mask percentage (e.g., 100%)
    html = re.sub(r"\b\d+%", '<PERCENTAGE>', html)
    # Mask storage usage (e.g., 45% used - 4.42 GB free)
    html = re.sub(r'\b\d+(\.\d+)?\s?(GB|MB|KB|TB|%)\b', '<STORAGE_VALUE>', html)

    # Mask dynamic status messages (e.g., No internet, Battery charging)
    dynamic_phrases = [
        'Battery charging, 100 percent',
    ]
    for phrase in dynamic_phrases:
        html = html.replace(phrase, '<Battery>')
    
    return html


def extract_status(text):
    # match = re.search(r'Status:\s*(\w+)', text)
    match = re.findall(r'Status:\s*([^\n]+)', text)
    # pdb.set_trace()
    if match:
        return match[-1]
    else:
        return None
    

def mark_frames_to_skip(policy_agent_trace):
    """
    Mark each frame in the trace with a skip flag.
    Skip if:
      1. Consecutive duplicate (same masked HTML as previous), OR
      2. Home/launcher screen (heuristic check).
    
    Returns:
        skip_flags: list[bool]   True if frame should be skipped
        html_descs: list[str]    Masked HTML description for each frame
    """
    skip_flags = []
    html_descs = []

    prev_html_desc = None
    for node in policy_agent_trace:
        if not node.state:
            skip_flags.append(False)
            html_descs.append(None)
            continue
        

        # pdb.set_trace()
        if "raw_ui_state" not in node.state:
            html_input = node.state['html_desc']
        else:
            raw_ui_state = node.state["raw_ui_state"]

            # Handle both dicts and objects
            if isinstance(raw_ui_state, dict):
                forest = raw_ui_state.get("forest", None)
            else:
                forest = getattr(raw_ui_state, "forest", None)

            if isinstance(forest, str):
                # evaluator-style: already plain text
                html_input = forest
            else:
                # policy-style: convert structured tree to HTML
                html_input = turn_tree_to_html_input_v2(forest)

        # raw_ui_state = node.state["raw_ui_state"].forest
        # html_input = turn_tree_to_html_input_v2(raw_ui_state)
        html_desc = mask_irrelevant_info(html_input)

        # --- Duplicate check ---
        is_dup = (html_desc == prev_html_desc)

        # --- Home screen check ---
        is_home = is_home_screen(html_desc)

        # --- Combine into one flag ---
        skip_flags.append(is_dup or is_home)
        html_descs.append(html_desc)

        prev_html_desc = html_desc

    return skip_flags, html_descs

def is_home_screen(html_desc: str) -> bool:
    """
    Detect if a screen is a home/launcher screen.
    Uses heuristic rules based on common launcher UI elements.
    """
    if html_desc is None:
        return False

    lowered = html_desc.lower()

    # Must contain the word 'home' explicitly
    
    # Check for common launcher apps
    launcher_keywords = ["home", "phone", "messages", "chrome", "gmail", "youtube", "photos", "predicted app", "search", "google app", "voice search"]
    found_apps = sum(1 for kw in launcher_keywords if kw in lowered)

    # If at least 6 launcher-like apps are present, treat as home screen
    if found_apps >= 6:
        return True

    return False
