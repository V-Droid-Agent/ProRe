import gzip
import pickle
import os
import re
from PIL import Image
from tqdm import tqdm, trange
import sys
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from android_world.agents import infer
from html_representation.html_representation import turn_tree_to_html_input_v2
from util import dfs_max_reward, extract_last_action_history, split_additional_ui_pages, extract_content_from_text, mark_frames_to_skip, extract_status
from prompt_template import *

class DummyNode:
    def __init__(self, step_id, ui_page_text, screenshot_array):
        self.step_id = step_id
        self.state = {
            "raw_ui_state": {"forest": ui_page_text},  # text UI description
            "screenshot_raw": screenshot_array         # numpy array
        }
        self.node_info = {}


def build_evaluator_trace(screenshot_dir, final_ui_state, additional_ui_pages_text):
    """
    Build evaluator trace nodes from screenshots + UI pages text.
    Loads screenshot into numpy array.
    """
    steps = split_additional_ui_pages(additional_ui_pages_text)
    steps = [final_ui_state] + steps
    evaluator_trace = []
    # pdb.set_trace()
    for idx, desc in enumerate(steps, start=1):
        screenshot_file = os.path.join(screenshot_dir, f"iter_1_step{idx}.jpg")
        if not os.path.exists(screenshot_file):
            continue  # skip if missing
        
        # Load image into numpy array
        screenshot_array = np.array(Image.open(screenshot_file))

        # Extract text after "UI Page:" if present
        if "UI Page:" in desc:
            ui_page_text = desc.split("UI Page:", 1)[1].strip()
        else:
            ui_page_text = desc

        node = DummyNode(step_id=idx, ui_page_text=ui_page_text, screenshot_array=screenshot_array)
        evaluator_trace.append(node)

    return evaluator_trace


def build_simplified_claims_prompt(
    trace,
    intent,
    action_history="",
    role="evaluator",
    min_claims=1,
    max_claims=4,
    use_skip=True,
):
    """
    Build a claims-generation prompt for a single role ('policy' or 'evaluator').
    Returns (prompt_text, images).
    """

    role = role.strip()
    assert role in {"Policy", "Evaluator"}, "role must be 'policy' or 'evaluator'"

    # Collect HTML/text + screenshots
    def _extract_forest(raw_ui_state):
        if isinstance(raw_ui_state, dict):
            return raw_ui_state.get("forest", None)
        return getattr(raw_ui_state, "forest", raw_ui_state)

    def _html_from_node(node):
        if node.state:
            raw_ui_state = node.state.get("raw_ui_state", None)
            forest = _extract_forest(raw_ui_state) if raw_ui_state is not None else None
            if isinstance(forest, str):
                return forest
            if forest is None:
                return "[No HTML available]"
        else:
            return "[No HTML available]"
        return turn_tree_to_html_input_v2(forest)

    html_text, images = [], []
    
    skip_flags, _ = mark_frames_to_skip(trace)

    if not trace:
        html_text.append("[No screens in this trace]")
    else:
        for i, node in enumerate(trace):
            if node:
                if use_skip and skip_flags[i]:
                    html_text.append(f'{role} Agent Step ' + str(i+1) + '-\n' + 'Skipped: duplicated or home screen.')
                    continue
                html_input = _html_from_node(node)
                html_text.append(f'{role} Agent Step ' + str(i+1) + '-\n' + html_input + '\n')
                if node.state:
                    shot = node.state.get("screenshot_raw")
                if shot is not None:
                    images.append(shot)
                

    role_key = f"{role}_claims"
    html_text_block = "\n".join(html_text)

    if role.lower() == "policy":
        guidelines = f"""
**Guidelines for writing claims (Policy Agent):**

### Core Mandate: The Actor's Report
- Think of yourself as the agent actively performing the task. Your claims are a direct report of your own actions and their immediate results.
- Your goal is to narrate your journey through the task, focusing only on the steps you took and the UI states you directly observed or caused.
- Be concise, factual, and strictly focused on the task goal. Avoid speculation or opinions about why something happened.

### Claim Generation Rules:
- Aim for {min_claims}–{max_claims} claims total.
- **For Tasks Involving Information Sources (e.g., "from an image," "using the details in the file"):**
  - **1. Access the Source:** Generate a claim confirming you **accessed and viewed the specified information source**.
    - *Example:* "The agent opened `expenses.jpg` in the gallery to view the expense details."
  - **2. Confirm Data Match:** In the claim about entering the data, explicitly state that the **data entered matches the data from the source**.
    - *Example:* "The two expenses entered, 'Office Supplies for 150' and 'Travel Expenses for 200', match the content of `expenses.jpg`."
- **For Editing, Modifying, or Deletion Tasks:**
  - **1. Capture the 'Before' State:** First, generate a claim that **describes the initial state of the item BEFORE the modification**.
    - *Example:* "Before editing, the contact's phone number was '555-123-4567'."
  - **2. Report the 'After' State:** Then, generate a separate claim describing the **successful modification or deletion**.
    - *Example:* "The contact's phone number was successfully updated to '555-987-6543'."
- **Report All Critical Actions:**
  - Describe your key actions and their direct consequences using state/action phrasing (e.g., “Recording saved and appears in list”).
  - Highlight any mismatches, errors, or unintended actions you performed (e.g., "Opened the wrong menu," "A 'Permission Denied' error appeared").
- **Be Efficient and Relevant:**
  - Merge duplicate claims that describe the same state.
  - Ignore trivial system indicators (battery, clock, signal), home/launcher screens, and redundant repeated actions unless they are evidence of an error or loop.
- Output must be valid JSON following the schema below.
"""
    else:
        guidelines = f"""
**Guidelines for writing claims (Evaluator Agent):**

### Core Mandate: The Detective Analogy
- Think of yourself as a detective arriving at a scene *after* the suspect (the Policy Agent) has left.
- The action history and screenshots you see are your own investigation—using your 'magnifying glass' and 'tools' to inspect the scene.
- Your goal is to make claims about the state of the scene **as the Policy Agent left it**.
- **You must NEVER create a claim about your own investigative actions.** For example, if you tap 'Save' or 'Delete' to check a confirmation dialog, you must not claim "The agent saved the file" or "The agent deleted the item." Your actions are not part of the evaluated task.

## Claim Generation Rules:
- Aim for {min_claims}–{max_claims} claims total.
- **Focus on the evidence:** All claims must describe the final state resulting from the Policy Agent's work, using your observations as proof.
- **Be factual and concise:** Merge duplicates and report on what is present or missing. Avoid speculation.
- **Identify mismatches:** If your investigation reveals that the final state contradicts the task goal (e.g., wrong file type, incorrect note name, content not saved, settings not changed), these are critical claims to include.
- **Ignore trivial states:** Do not report on system indicators (battery, clock), home screens, or app launchers unless directly relevant to the task goal.
- **Phrase claims effectively:** Prefer state/action summaries (e.g., “Recording saved and appears in list”) over simple lists of UI elements.
- Output must be valid JSON following the schema below.
"""

    # Prompt construction
    prompt = f"""You are an expert in evaluating the performance of a mobile GUI agent.
**Workflow overview:**  
1. **User** provides a task intent.  
2. The **Policy Agent** executes UI actions to fulfil that task; its steps are recorded as *Action History*.  
3. The **Evaluator Agent** runs after the Policy Agent has finished, and proactively interact with the environment to gather additional observations.
4. **You** will now produce concise **claims** for the **{role.capitalize()} Agent** only.

You must follow a step-by-step analysis:
1. Read the **Task Goal** and the {role.capitalize()} Agent’s action history (if available).
2. Examine the provided {role.capitalize()} screens (HTML + screenshots are attached in order).
3. Synthesize related observations into claims. Each claim must:
   • List the supporting step indices.
   • Give a brief, evidence-grounded rationale.
   • State a concise, goal-relevant claim.
4. Include any details critical to the final judgment directly in the claims (e.g., specific titles, timestamps, targets, confirmations, error messages).
6. Do **not** judge final success/failure here; only produce claims.

────────────────────────  INPUTS  ────────────────────────
TASK GOAL:
{intent}

ACTION HISTORY ({role.capitalize()} Agent):
{action_history if action_history else "[No action history provided]"}

HTML STATES (TRACE of {role.capitalize()} Agent): 
{html_text_block}

────────────────────── OUTPUT GUIDELINES ──────────────
{guidelines}

────────────────────── OUTPUT SCHEMA  ────────────────────
{{
  "{role_key}": [
    {{
      "steps": [<list of step numbers>],
      "reasoning": "<brief explanation of why this claim is justified>",
      "claim": "<concise, goal-relevant claim>"
    }},
    ...
  ]
}}

Return only the JSON under **Claims:**
"""
    return prompt, images


def extract_claims_from_response(response_text, role="Evaluator"):
    """
    Extract structured claims from an LLM response.
    Expected schema:
    {
      "Evaluator_claims": [
        {"steps": [...], "reasoning": "...", "claim": "..."},
        ...
      ]
    }
    or
    {
      "Policy_claims": [...]
    }

    Args:
        response_text (str): raw LLM output
        role (str): "policy" or "evaluator"

    Returns:
        list of claims (dicts), or [] if parsing fails
    """
    role_key = f"{role}_claims"

    # Find JSON block
    match = re.search(r"\{[\s\S]*\}", response_text)
    if not match:
        print("⚠️ No JSON object found in response.")
        return []

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
    except Exception as e:
        print(f"⚠️ Failed to parse JSON: {e}")
        return []

    claims = data.get(role_key, [])
    if not isinstance(claims, list):
        print(f"⚠️ {role_key} not found or not a list.")
        return []

    return claims


compare_claims_prompt = """You are an expert judge evaluating whether a mobile GUI agent (Policy Agent) has completed the user’s task.  

**Workflow overview:**  
1. **User** provides a task intent.  
2. The **Policy Agent** executes UI actions to fulfil that task; its steps are recorded as *Action History*.  
3. The **Evaluator Agent** runs after the Policy Agent has finished, and proactively probes the resulting states to gather additional observations.
4. Your job is to analyze these claims together, identify their relationships, and determine whether the Policy Agent successfully completed the task.  

You must follow a step-by-step analysis:  
1. **Read the Task Goal** carefully to understand what success means.  
2. **Compare Policy Claims and Evaluator Claims**:  
   • Mark as **confirmed** if an evaluator claim supports a policy claim.  
   • Mark as **contradicted** if an evaluator claim directly disproves a policy claim.  
   • Mark as **complementary** if the evaluator provides additional relevant evidence.  
   • Mark as **unsupported** if no evaluator claim addresses a policy claim.  
3. Highlight any **critical confirmations or contradictions** that directly determine success.  
4. Decide the outcome reward: did the Policy Agent achieve the user’s task goal?  

**Guidelines:**  
- Policy Agent claims may describe progress that is not directly observed by the Evaluator Agent. This is normal and should not be treated as a failure unless the evaluator explicitly disproves the claim.
- Do not mark unobserved progress as contradiction unless explicitly disproved.  
- Consider the correctness of the **target** (e.g., the right file, event, app).  
- For question-answer tasks, the Policy Agent must include an explicit answer claim.  
- Ignore evaluator stray/accidental claims unrelated to the goal.  
- If claims indicate progress but also critical issues (wrong extension, malicious steps), treat as compensated or failure depending on severity.

────────────────────────  INPUTS  ────────────────────────
TASK GOAL:  
{intent}

POLICY CLAIMS:  
{policy_claims}

EVALUATOR CLAIMS:  
{evaluator_claims}

────────────────────── OUTPUT INSTRUCTIONS ──────────────  
Write your reasoning in two sections:  

Analysis:  
- Compare claims explicitly, showing relations (confirmed, contradicted, complementary, unsupported).  
- Explain how these relations support your judgment.  

Status: success or failure  

Return only these two sections, exactly in this format.  
"""

compare_claims_prompt_v2 = """You are an expert judge evaluating whether a mobile GUI agent (Policy Agent) has completed the user’s task.  

**Workflow overview:**  
1. **User** provides a task intent.  
2. The **Policy Agent** executes UI actions to fulfil that task; its steps are recorded as *Action History*.  
3. The **Evaluator Agent** runs after the Policy Agent has finished, and proactively probes the resulting states to gather additional observations.
4. Your job is to analyze these claims together, identify their relationships, and determine whether the Policy Agent successfully completed the task.  

You must follow a two-stage analysis:  

### Stage 1 — Filter Evaluator Claims
- Carefully review the evaluator claims.  
- **Discard any claim that describes actions or outcomes caused by the Evaluator Agent itself** (e.g., accidental saves, unintended edits, stray taps/scrolls).  
- Keep only evaluator claims that serve as **evidence about the Policy Agent’s actual outcome**.  
- If in doubt, prefer to exclude rather than include.  

### Stage 2 — Compare Policy vs. Evaluator Claims
1. **Read the Task Goal** carefully to understand what success means.  
2. **Compare Policy Claims and (filtered) Evaluator Claims**:  
   • Mark as **confirmed** if an evaluator claim supports a policy claim.  
   • Mark as **contradicted** if an evaluator claim directly disproves a policy claim.  
   • Mark as **complementary** if the evaluator provides additional relevant evidence.  
   • Mark as **unsupported** if no evaluator claim addresses a policy claim.  
3. Highlight any **critical confirmations or contradictions** that directly determine success.  
4. Decide the outcome reward: did the Policy Agent achieve the user’s task goal?  

**Guidelines:**  
- Before labeling a contradiction, check if the agents are simply observing different aspects of the same content (e.g., Policy saw page 1, Evaluator scrolled to page 2).
  • If so, their claims are **complementary**. Your job is to **synthesize** them into a single, more complete understanding.
- When claims are in direct conflict, act as a critical arbiter rather than a passive matcher. Evaluate reliability and consistency; do not assume both sides are equally valid.
- Consider the correctness of the **target** (e.g., the right file, event, app).  
- For question-answer tasks, the Policy Agent must include an explicit claim stating the answer it provided, expressed exactly as required by the task.
- Ignore evaluator stray/accidental claims unrelated to the goal.  
- If claims indicate progress but also critical issues (wrong extension, malicious steps), treat as compensated or failure depending on severity.  

────────────────────────  INPUTS  ────────────────────────
TASK GOAL:  
{intent}

POLICY CLAIMS:  
{policy_claims}

EVALUATOR CLAIMS:  
{evaluator_claims}

────────────────────── OUTPUT INSTRUCTIONS ──────────────  
Write your reasoning in two sections:  

Analysis:  
- Stage 1: List which evaluator claims you discarded and why.  
- Stage 2: Compare the remaining evaluator claims against the policy claims, showing relations (confirmed, contradicted, complementary, unsupported).  
- Explain how these relations support your judgment.  

Status: success or failure  

Return only these two sections, exactly in this format.  
"""


def claims_to_text(claims, role="Policy"):
    """
    Convert structured claims into a readable text block for LLM prompts.

    Args:
        claims (list[dict]): list of claims with keys "steps", "reasoning", "claim"
        role (str): "Policy" or "Evaluator"

    Returns:
        str: formatted text block
    """
    
    if not claims:
        return f"[No {role} claims generated]\n"

    lines = []
    
    for i, c in enumerate(claims, 1):
        steps = ", ".join(map(str, c.get("steps", []))) or "N/A"
        reasoning = c.get("reasoning", "").strip()
        claim_text = c.get("claim", "").strip()
        lines.append(
            f"Claim {i} ({role} Agent steps {steps}):\n"
            f"  Reasoning: {reasoning}\n"
            f"  Claim: {claim_text}\n"
        )
    return "\n".join(lines)



def assemble_claims_as_prompt(intent, policy_claims, evaluator_claims):
    """
    Assemble policy and evaluator claims into a single text block
    for use in comparison prompts.
    """
    policy_text = claims_to_text(policy_claims, role="Policy")
    evaluator_text = claims_to_text(evaluator_claims, role="Evaluator")

    prompt = compare_claims_prompt_v2.format(
        intent=intent,
        policy_claims=policy_text,
        evaluator_claims=evaluator_text,
        # action_history=action_history,
    )
    return prompt



def get_token_usage(resp, llm_model):
    try:
        if "gpt" in llm_model:
            return {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "thinking_tokens": 0
            }
        elif "gemini" in llm_model:
            return {
                "prompt_tokens": resp.usage_metadata.prompt_token_count,
                "completion_tokens": resp.usage_metadata.candidates_token_count,
                "thinking_tokens": resp.usage_metadata.thoughts_token_count
            }
    except Exception as e:
        print(f"⚠️ Failed to extract token usage: {e}")
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "thinking_tokens": None
    }


def load_processed_tasks(response_file):
    processed = {}
    with open(response_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                task, val = line.split(":")
                processed[task.strip()] = int(val.strip())
            except:
                continue
    return processed

task_names = [
    "NotesIsTodo",
    "RetroPlayingQueue",
    "SimpleCalendarNextMeetingWithPerson",
    "SportsTrackerActivitiesCountForWeek",
    "TasksDueOnDate",
    "ClockStopWatchPausedVerify",
    "RecipeAddMultipleRecipes",
    "RecipeDeleteDuplicateRecipes",
    "SimpleSmsResend"
]

def load_claims_from_dir(claims_dir, task_name=None):
    """
    Load claims from JSON files in a directory.

    Args:
        claims_dir (str): path to directory containing *_claims.json files
        task_name (str, optional): specific task to load (without "_claims.json")

    Returns:
        dict: {task_name: {"policy_claims": [...], "evaluator_claims": [...], "goal": ...}, ...}
    """
    try:
        with open(claims_dir, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load {claims_dir}: {e}")
        return None

    return data

def query_llm_for_chain_judgement_simplied(llm, traj_path, eval_save_name, response_file, results_file, save_name, read_from_file=False):
    """
    Iterate over all tasks in eval_save_name and run chain-based A3J judgement.
    """
    base_eval_dir = f"saved/{eval_save_name}/record"
    invalid_tasks = []
    prompts_token, completions_token, thinkings_token = [], [], []
    
    try:
        processed_tasks = load_processed_tasks(results_file)
    except:
        print("no processed_tasks")
        processed_tasks = []

    target_tasks = task_names
    for filename in tqdm(sorted(os.listdir(base_eval_dir))):
        if not filename.endswith("_evaluation"):
            continue
        
        if "_0_evaluation" in filename:
            task = filename.replace("_0_evaluation", "")
        else:
            task = filename.replace("_evaluation", "")
        data_path = f"saved/{traj_path}/task_info/{task}_0.pkl.gz"
        log_path = os.path.join(base_eval_dir, filename, "app.log")
        screenshot_dir = os.path.join(base_eval_dir, filename, "screen_shot")

        if task in processed_tasks and processed_tasks[task] != -1:
            print(f"Skipping {task}, already processed.")
            continue
        
        # if task not in target_tasks:
        #     continue

        # --- Load trajectory ---
        try:
            with gzip.open(data_path, "rb") as f:
                load_data = pickle.load(f)
        except FileNotFoundError:
            print(f"Missing trajectory for {task}")
            invalid_tasks.append(task)
            continue
            
        original_goal = load_data[0]['goal']
        root = load_data[0]['tree']
        _, policy_agent_trace = dfs_max_reward([root])

        intent, action_history, final_ui_state, additional_ui_pages = None, None, None, None

        if os.path.exists(log_path):
            try:
                intent, action_history, monitored_infor, final_ui_state, additional_ui_pages = extract_content_from_text(log_path)
                evaluator_agent_history = extract_last_action_history(log_path)
            except Exception as e:
                print(f"Failed to parse log {log_path}: {e}")
                invalid_tasks.append(task)
                continue
        else:
            print(f"Missing log for {task}, skipping evaluator trace")
            invalid_tasks.append(task)
            continue
        
        evaluator_agent_trace = build_evaluator_trace(screenshot_dir, final_ui_state, additional_ui_pages)

        policy_claims_prompt, policy_images = build_simplified_claims_prompt(trace=policy_agent_trace, intent=intent, action_history=action_history, role="Policy")
        response, _, policy_ori_response = llm.predict_mm(policy_claims_prompt, policy_images)
        policy_claims = extract_claims_from_response(response, role="Policy")

        evaluator_claims_prompt, evaluator_images = build_simplified_claims_prompt(trace=evaluator_agent_trace, action_history=evaluator_agent_history, intent=intent, role="Evaluator")
        response, _, evalutor_ori_response = llm.predict_mm(evaluator_claims_prompt, evaluator_images)
        evalutor_claims = extract_claims_from_response(response, role="Evaluator")

        if not policy_claims or not evalutor_claims:
            with open(response_file, "a") as f:
                f.write(f"{task}: {-1}\n")

        prompt = assemble_claims_as_prompt(intent, policy_claims, evalutor_claims)
        
        print("=" * 50)
        print(prompt)
        
        # --- Ask final judge ---
        response_text, _, ori_response = llm.predict(prompt)
        print(response_text)
        print("=" * 50)
        # --- Evaluate result ---
        if not response_text or not response_text.strip():
            # No response from LLM → mark as invalid
            result_value = -1
        else:
            status = extract_status(response_text)
            if status and "success" in status.lower():
                result_value = 1
            elif status and "failure" in status.lower():
                result_value = 0
            else:
                result_value = -1

        claims_save_dir = f"saved/{eval_save_name}/claims_{save_name}"
        os.makedirs(claims_save_dir, exist_ok=True)

        policy_tokens = get_token_usage(policy_ori_response, llm.model_name)
        evaluator_tokens = get_token_usage(evalutor_ori_response, llm.model_name)
        final_tokens = get_token_usage(ori_response, llm.model_name)

        claims_path = os.path.join(claims_save_dir, f"{task}_claims.json")
        with open(claims_path, "w") as f:
            json.dump(
                {
                    "task": task,
                    "goal": intent,
                    "policy_claims_prompt": policy_claims_prompt,
                    "evaluator_claims_prompt": evaluator_claims_prompt,
                    "policy_claims": policy_claims,
                    "evaluator_claims": evalutor_claims,
                    "final_prompt": prompt,
                    "final_response_text": response_text,
                    "tokens": {
                        "policy_claims": policy_tokens,
                        "evaluator_claims": evaluator_tokens,
                        "final_judgement": final_tokens
                    }
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        # --- Save outputs ---
        with open(response_file, "a") as f:
            f.write(f"{task}: {response_text}\n")

        with open(results_file, "a") as f:
            f.write(f"{task}: {result_value}\n")

        # --- Track token usage if available ---
        try:
            if 'gpt' in llm.model_name:
                prompts_token.append(ori_response.usage.prompt_tokens)
                completions_token.append(ori_response.usage.completion_tokens)
                thinkings_token.append(0)
            elif 'gemini' in llm.model_name:
                prompts_token.append(ori_response.usage_metadata.prompt_token_count)
                completions_token.append(ori_response.usage_metadata.candidates_token_count)
                thinkings_token.append(ori_response.usage_metadata.thoughts_token_count)
        except Exception as e:
            print(f"Warning: token usage not available for {task} ({e})")

    
    # --- Report ---
    print("Invalid tasks:", invalid_tasks)
    if prompts_token:
        print("Avg prompt tokens:", np.mean(prompts_token))
        print("Avg completion tokens:", np.mean(completions_token))
        print("Avg thinking tokens:", np.mean(thinkings_token))
    
    return 



if __name__ == '__main__':
    llm_name = "gemini-2.5-pro-preview-05-06"
    method = "CoE"
    save_name = "v5_full"
    check_k = "Proactive"
    final_judger = infer.GeminiGcpWrapper(llm_name, temperature= 0)

    traj_path = "...."
    eval_save_name = "...."

    target_folder = f"saved/{eval_save_name}"
    results_file = f"{target_folder}/llm_{llm_name}_eval_{method}_images{check_k}_{save_name}.txt"
    response_file = f"{target_folder}/llm_{llm_name}_eval_{method}_images{check_k}_{save_name}_response.log"

    query_llm_for_chain_judgement_simplied(final_judger, traj_path, eval_save_name,
                                  response_file, results_file, save_name)