import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util import split_additional_ui_pages, extract_content_from_text,dfs_max_reward, load_trajectory
from android_world.agents.claims import assemble_claims_as_prompt, build_simplified_claims_prompt, extract_claims_from_response
from android_world.agents import infer
from html_representation.html_representation import turn_tree_to_html_input_v2
from tqdm import tqdm
from absl import logging
import time
import pandas as pd
import numpy as np
from PIL import Image
from util import extract_status

evaluate_prompt = (
"""You're an expert in evaluating whether the Screenshot successfully completes the Task.
Here is one example:
Task: Navigate to example.com
Q: What should I expect to see on the screenshot if I've navigated to example.com?
A: I should expect to see the Example website homepage with a clear header, navigation menu, and typical example content. The screenshot shows a generic browser window that lacks the expected website features.
Status: failure
Task: {intent}
Respond in this format:
Q: What should I expect to see on the screenshot if I've {intent}?
A: I should expect to see <describe expected outcome>. The screenshot shows <describe actual screenshot>.
Status: success or failure (don't return anything else).
Start with "Q:"."""
)

a3e_prompt_v2 = ("""You are an expert in evaluating the performance of a mobile GUI agent.
The agent helps a human user interact with a smartphone UI to complete a task.
Given the user’s intent, the agent's action history, and a sequence of raw UI states (in HTML format and images), your job is to decide whether the task was successfully completed.

You must follow a step-by-step analysis:
1. Begin by analyzing the **final UI state** presented.
2. Then examine each prior UI page (in temporal order) from the "Additional Information".
3. For each page, identify whether it includes the key UI elements or feedback mechanisms relevant to the task.
4. Compare what should be expected (based on the task) vs. what is observed.
5. Conclude whether the agent’s behavior achieved the desired outcome.
6. Your final output must strictly follow the format below:
- Start with **"Analysis:"**
- Follow with your observations for both the final UI and the additional pages
- End with **"Status: success"** or **"Status: failure"**

Here is one example:
Task: Navigate to the Weather app  
Analysis: For the final UI State, I should expect to see the Weather app interface with current weather information such as location, temperature, and forecast. However, the screen displays only the home screen with app icons, and no weather content is visible.  
For Additional UI Pages, the agent navigated through unrelated apps and never opened the Weather app.  
Status: failure

---

Now evaluate the following:

Task: {intent}

Action History:  
{history}

Final UI State Description (in HTML):  
{last_html}

Additional UI Pages Visited (in HTML):  
{additional_info}

Respond in this format:
Analysis: For the final UI State, <your analysis>. For Additional UI Pages, <your analysis>.  
Status: success or failure (don't return anything else).
Start with "Analysis:"."""
)

a3e_prompt_v3 = ("""You are an expert in evaluating the performance of a mobile GUI agent (Policy Agent).  
**Workflow overview:**  
1. **User** provides a task intent.  
2. The **Policy Agent** executes UI actions to fulfil that task; its steps are recorded as *Action History*.  
3. The **Evaluator Agent** runs in parallel: it observes each resulting screen **and may actively navigate further** to gather extra evidence after the policy agent finishes its execution.  
   • For every screen it sees, it produces a concise, readable *Monitored Information* summary.  
   • Screens opened during this verification exploration appear in *Additional UI Pages*.  
4. **You**, acting as the judge, must decide whether the task was successfully completed.

You must follow a step-by-step analysis:  
1. Begin with the **final UI state** (use both its raw HTML and monitored summary).  
2. Examine earlier pages in temporal order, using their monitored summaries.  
3. For each page, note key UI elements or feedback relevant to the task.  
4. Compare expected cues (based on the task) to observed evidence.  
5. Decide if the Policy Agent achieved the goal. 
6. Output must strictly follow this format:  
   - Start with **"Analysis:"**  
   - Provide your observations for the final UI and the additional pages  
   - End with **"Status: success"** *or* **"Status: failure"**

Example:
Task: Navigate to the Weather app  
Analysis: For the final UI State, I should expect … However, the screen shows only home icons.  
For Additional UI Pages, the agent opened unrelated apps and never reached Weather.  
Status: failure

---

Now evaluate the following:

Task (from User):  
{intent}

Action History (from Policy Agent):  
{history}

Monitored Information (from Evaluator Agent while observing each Policy Agent step, newest last):  
{monitored_info}

Final UI State (raw HTML, captured by Evaluator Agent):  
{last_html}

Additional UI Pages (raw HTML, captured by Evaluator Agent, newest last):  
{additional_info}

Respond in this exact format:  
Analysis: For the Action History and Monitored Information, <your analysis>. For the final UI State, <your analysis>. For Additional UI Pages, <your analysis>.  
Status: success or failure (don't return anything else).  
Start with "Analysis:"."""
)

a3e_prompt_v5 = ("""You are an expert in evaluating the performance of a mobile GUI agent (Policy Agent).  
**Workflow overview:**  
1. **User** provides a task intent.  
2. The **Policy Agent** executes UI actions to fulfil that task; its steps are recorded as *Action History*.  
3. The **Evaluator Agent** runs in parallel: it observes each resulting screen **and may actively navigate further** to gather extra evidence after the policy agent finishes its execution.  
   • For every screen it sees, it produces a concise, readable *Monitored Information* summary.  
   • Screens opened during this verification exploration appear in *Additional UI Pages*.  
4. **You**, acting as the judge, must decide whether the task was successfully completed.

You must follow a step-by-step analysis:  
1. Begin with the **final UI state** (use both its raw HTML and monitored summary).  
2. Examine earlier pages in temporal order, using their monitored summaries.  
3. For each page, note key UI elements or feedback relevant to the task.  
4. Compare expected cues (based on the task) to observed evidence.  
5. Decide if the Policy Agent achieved the goal. 
6. Output must strictly follow this format:  
   - Start with **"Analysis:"**  
   - Provide your observations for the final UI and the additional pages  
   - End with **"Status: success"** *or* **"Status: failure"**

**Guidelines for the analysis:**
- Ignore evaluator stray actions. The Evaluator Agent may accidentally tap or scroll while scouting for proof. Omit any such unintended actions and their side-effects from both the analysis and the final judgment. Only the intended actions of the Policy Agent should be considered.
- Answer required for question-type tasks. If the original task asks a question, the Policy Agent must include an explicit answer action in its history. Merely displaying the correct screen is not enough. Absence of an answer action means the task fails. State this clearly in the evidence.
- Verify the target entity of each action. For operations involving specific items (e.g., files, folders, or UI elements), ensure that the action is applied to the correct target. Executing the right action on the wrong item should be treated as a failure.
                 
Example:
Task: Navigate to the Weather app  
Analysis: For the final UI State, I should expect … However, the screen shows only home icons.  
For Additional UI Pages, the agent opened unrelated apps and never reached Weather.  
Status: failure

---

Now evaluate the following:

Task (from User):  
{intent}

Action History (from Policy Agent):  
{history}

Monitored Information (from Evaluator Agent while observing each Policy Agent step, newest last):  
{monitored_info}

Final UI State (raw HTML, captured by Evaluator Agent):  
{last_html}

Additional UI Pages (raw HTML, captured by Evaluator Agent, newest last):  
{additional_info}

Respond in this exact format:  
Analysis: For the Action History and Monitored Information, <your analysis>. For the final UI State, <your analysis>. For Additional UI Pages, <your analysis>.  
Status: success or failure (don't return anything else).  
Start with "Analysis:"."""
)


a3e_prompt_v4 = ("""You are an expert in evaluating the performance of a mobile GUI agent (Policy Agent).  
**Workflow overview:**  
1. **User** provides a task intent.  
2. The **Policy Agent** executes UI actions to fulfil that task; its steps are recorded as *Action History*.  
3. The **Evaluator Agent** runs in parallel: it observes each resulting screen **and may actively navigate further** to gather extra evidence after the policy agent finishes its execution.  
   • For every screen it sees, it produces a concise, readable *Monitored Information* summary.  
   • Screens opened during this verification exploration appear in *Additional UI Pages*.  
4. **You**, acting as the judge, must decide whether the task was successfully completed.

You must follow a step-by-step analysis:  
1. Begin with the **final UI state** (use both its raw HTML and monitored summary).  
2. Examine earlier pages in temporal order, using their monitored summaries.  
3. For each page, note key UI elements or feedback relevant to the task.  
4. Decide if the Evaluator Agent collect complete evidence. 
5. Compare expected cues (based on the task) to observed evidence.  
6. Decide if the Policy Agent achieved the goal.  
7. Output must strictly follow this format:  
   - Start with **"Analysis:"**  
   - Provide your observations for the final UI and the additional pages  
   - End with **"Status: success"** *or* **"Status: failure"** or **"Status: incomplete"**
   - If evidence clearly shows goal achievement → **success**  
   - If evidence clearly shows goal failure → **failure**  
   - If evidence is insufficient to decide → **incomplete**

**Guidelines for the analysis:**
- Ignore evaluator stray actions. The Evaluator Agent may accidentally tap or scroll while scouting for proof. Omit any such unintended actions and their side-effects from both the analysis and the final judgment. Only the intended actions of the Policy Agent should be considered.
- Answer required for question-type tasks. If the original task asks a question, the Policy Agent must include an explicit answer action in its history. Merely displaying the correct screen is not enough. Absence of an answer action means the task fails. State this clearly in the evidence.

Example:
Task: Navigate to the Weather app  
Analysis: For the final UI State, I should expect … However, the screen shows only home icons.  
For Additional UI Pages, the agent opened unrelated apps and never reached Weather.  
Status: failure

---

Now evaluate the following:

Task (from User):  
{intent}

Action History (from Policy Agent):  
{history}

Monitored Information (from Evaluator Agent while observing each Policy Agent step, newest last):  
{monitored_info}

Final UI State (raw HTML, captured by Evaluator Agent):  
{last_html}

Additional UI Pages (raw HTML, captured by Evaluator Agent, newest last):  
{additional_info}

Respond in this exact format:  
Analysis: For the Action History and Monitored Information, <your analysis>. For the final UI State, <your analysis>. For Additional UI Pages, <your analysis>.  
Status: success, failure or incomplete (don't return anything else).  
Start with "Analysis:"."""
)

a3e_chain_prompt_v1 = ("""You are an expert in evaluating the performance of a mobile GUI agent (Policy Agent).  
**Workflow overview:**  
1. **User** provides a task intent.  
2. The **Policy Agent** executes UI actions to fulfil that task; its steps are abstracted into **Policy Claims**.  
3. The **Evaluator Agent** observes resulting screens and may explore further to gather evidence. Its observations are abstracted into **Evaluator Claims**.  
4. A claim-alignment process links Policy Claims to the most relevant Evaluator Claims, producing a **Chain of Evidence**:  
   • Each pair shows whether the evaluator confirmed, contradicted, or supplemented the policy’s claim.  
   • Some evaluator claims may appear as extra evidence (not directly aligned).  
5. **You**, acting as the judge, must decide whether the task was successfully completed.

You must follow a step-by-step analysis:  
1. Begin with the **chain of evidence**. For each aligned pair, examine the relation (confirmed, contradicted, compensated, unsupported, or extra_evidence).  
2. Pay special attention to critical claims tied to the task goal: did the evaluator confirm or contradict them?  
3. Consider extra evidence that might prove or disprove success.  
4. Decide if the Policy Agent achieved the goal.  
5. Output must strictly follow this format:  
   - Start with **"Analysis:"**  
   - Provide your observations from the chain of evidence (group by relation type if helpful).  
   - End with **"Status: success"** *or* **"Status: failure"**

**Guidelines for the analysis:**
- Ignore evaluator stray actions. Extra exploratory claims unrelated to the goal should not affect the final decision.  
- Answer required for question-type tasks. If the original task asks a question, the Policy Agent must include an explicit answer claim. Simply navigating to a relevant screen is insufficient. Absence of an answer claim means the task fails.  
- Verify the target entity of each claim. For operations involving specific items (e.g., files, folders, or UI elements), ensure that the claim matches the correct target. Executing the right action on the wrong item should be treated as a failure.  

Example:  
Task: Navigate to the Weather app  
Analysis: The chain of evidence shows the Policy Claim “User opened Weather” contradicted by the Evaluator Claim “Screen shows only home icons.” Extra evidence confirms no Weather screen. Therefore, the agent did not achieve the task.  
Status: failure

---

Now evaluate the following:

Task (from User):  
{intent}

Action History (from Policy Agent):  
{history}
                       
Chain of Evidence (aligned relations between Policy and Evaluator claims):  
{chain_of_evidence}

Respond in this exact format:  
Analysis: <your reasoning based on the chain of evidence>.  
Status: success or failure (don't return anything else).  
Start with "Analysis:"."""
)

a3e_chain_prompt_v2 = ("""You are an expert in evaluating the performance of a mobile GUI agent (Policy Agent).  
**Workflow overview:**  
1. The **Policy Agent** executes UI actions to fulfil the user’s task; its steps are abstracted into Policy Claims.  
2. The **Evaluator Agent** observes resulting screens and may explore further. Its observations are abstracted into Evaluator Claims.  
3. A claim-alignment step links Policy Claims with the most relevant Evaluator Claims. This produces a **Chain of Evidence**, where each link shows the paired claims but the relation is left as *pending*.  
   • Some Evaluator Claims may appear unpaired as extra evidence.  
   • Because the Evaluator acts *after* the Policy Agent, some policy progress may not be visible to the evaluator. This is normal and should not be treated as a contradiction unless the evaluator explicitly disproves the policy claim.  
4. **You**, acting as the judge, must decide whether the task was successfully completed.

You must follow a step-by-step analysis:  
1. Read the **task intent** carefully.  
2. Go through the **chain of evidence**: for each link, judge whether the evaluator’s claim confirms, contradicts, supplements, or does not address the policy claim.  
   • Mark as **confirmed** if the evaluator clearly supports the policy claim.  
   • Mark as **contradicted** only if the evaluator directly disproves the policy claim.  
   • Mark as **compensated** if the evaluator provides relevant missing information.  
   • Mark as **unsupported** if the evaluator does not address the policy claim.  
   • Extra evidence claims may also strengthen or weaken your decision.  
3. Pay special attention to claims that directly relate to the task goal.  
4. Decide if the Policy Agent achieved the goal.  
5. Output must strictly follow this format:  
   - Start with **"Analysis:"**  
   - Provide your reasoning using the chain of evidence (explicitly state relations).  
   - End with **"Status: success"** or **"Status: failure"**

**Guidelines:**  
- Ignore evaluator stray actions (unintended navigation or taps).  
- For question-type tasks, the Policy Agent must produce an explicit answer claim; showing the right screen alone is not enough.  
- Verify the target entity of each claim. Performing the correct action on the wrong item should be treated as failure.  
- Do not mark missing progress as contradictory unless it is explicitly disproved by evaluator evidence.  

Example:  
Task: Navigate to the Weather app  
Analysis: Policy Claim “Opened Weather” was contradicted by Evaluator Claim “Home screen shown.” Extra evidence shows no Weather UI. Therefore, the task was not completed.  
Status: failure

---

Now evaluate the following:

Task (from User):  
{intent}

Chain of Evidence (policy/evaluator claim pairs with pending relations):  
{chain_of_evidence}

Respond in this exact format:  
Analysis: <your reasoning with explicit relation labels>.  
Status: success or failure (don’t return anything else).  
Start with "Analysis:"."""
)


MAX_RETRY = 20
BACKOFF   = 2.0  

def load_gt_dataframe(root_dir: str) -> pd.DataFrame:
    records = []

    for dirpath, _, files in os.walk(root_dir):
        # pdb.set_trace()
        if "evaluation_results.txt" not in files:
            continue
        
        if '0_0' in dirpath:
            continue

        # task name is the directory name without the "_evaluation" suffix
        task_name = os.path.basename(dirpath).replace("_evaluation", "")
        if '_0' in task_name:
            task_name = task_name.replace('_0', '')
        # else:
            # task_name = task_name.replace('', '')
        # task_name = task_name.split("_")[0]
        
        eval_path = os.path.join(dirpath, "evaluation_results.txt")
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s*GT:\s*(\d)", line)
                if m:                               # found "GT: 0" or "GT: 1"
                    gt_val = int(m.group(1))
                    records.append((task_name, gt_val))
                    break

    # pdb.set_trace()
    # Build DataFrame
    df = pd.DataFrame(records, columns=["task_name", "gt_result"])
    df.set_index("task_name", inplace=True)
    return df


def load_gt_from_traj(root_dir: str):
    records = []
    with open(root_dir, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    task_name, gt_result_str = line.split(':')
                    gt_result = float(gt_result_str.strip())
                    records.append((task_name.strip(), gt_result))
                except ValueError:
                    print(f"Warning: Skipping malformed line: {line}")

    df = pd.DataFrame(records, columns=["task_name", "gt_result"])
    df.set_index("task_name", inplace=True)
    return df

def compare_with_gt(llm_evaluate_path, rule_evaluate_path, gt_type="a3j"):
    # Load LLM evaluation results.
    # Expected format per line: "<task>: <result>"
    llm_results = {}
    with open(llm_evaluate_path, "r") as f:
        for line in f:
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                task = parts[0].strip()
                try:
                    result = int(parts[1].strip())
                except ValueError:
                    continue
                llm_results[task] = result
    
    # Load rule-based evaluation results.
    # Expected CSV with an index column named "task" and a column "mean_success_rate"
    if gt_type == "aw":
        rule_df = pd.read_fwf(rule_evaluate_path, header=1, index_col=0)
        rule_df = rule_df[~rule_df.index.str.contains("Average", na=False)]
        print("Available columns in rule-based evaluation CSV:", rule_df.columns.tolist())
        rule_df["gt_result"] = rule_df["Unnamed: 3"].apply(lambda x: x)
    elif gt_type == "a3j":
        rule_df = load_gt_dataframe(rule_evaluate_path)
    elif gt_type == "traj":
        rule_df = load_gt_from_traj(rule_evaluate_path)
    
    # Compare LLM results with rule-based ground truth.
    total = 0
    correct = 0
    details = []
    llm_sr = 0

    tp, tn, fp, fn = 0, 0, 0, 0
    fp_tasks = []
    fn_tasks = []
    for task, gt in rule_df["gt_result"].items():
        if gt == 0.5:
            gt = 0
        if task in llm_results:
            llm_val = llm_results[task]
            llm_sr += llm_val
            total += 1
            if llm_val == gt:
                correct += 1
            
            if llm_val == 1 and gt == 1:
                tp += 1
            elif llm_val == 1 and gt == 0:
                fp += 1
                fp_tasks.append(task)
            elif llm_val == 0 and gt == 0:
                tn += 1
            elif llm_val == 0 and gt == 1:
                fn += 1
                fn_tasks.append(task)

            details.append((task, llm_val, gt))
        else:
            details.append((task, None, gt))
    
    accuracy = correct / total if total > 0 else None

    llm_success_rate = llm_sr/total
    rule_success_rate = rule_df["gt_result"].sum() / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    F1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


    print(f"Accuracy: {accuracy * 100:.2f} ({correct}/{total})")
    print(f"f1_score: {F1_score * 100:.2f}")
    
    print("Confusion Matrix:")
    print(f"TP: {tp/total}, FP: {fp/total}")
    print(f"FN: {fn/total}, TN: {tn/total}")
    # # pdb.set_trace()
    print(f"Acc Correct:{tp/(tp+fn) * 100:.2f}, Acc Wrong: {tn/(tn+fp) * 100:.2f}")
    
    return accuracy, F1_score, fp_tasks, fn_tasks


def success_flag(response_block: str) -> int:
    """
    Return 1 if no condition contains a failure indicator (-1), else 0.

    This function uses a regular expression to robustly detect "-1" as a 
    standalone word, handling various formats like "(Step -1)", ": -1", etc.

    Parameters
    ----------
    response_block : str
        The text block containing condition-step pairs.

    Returns
    -------
    int
        1 → successful (no "-1" values found)  
        0 → unsuccessful (at least one "-1" value found)
    """
    # This regex pattern looks for '-1' as a whole word.
    # \b is a word boundary, so this won't match 'file-1' or 'step-10'.
    failure_pattern = re.compile(r'(?<!\d)-1(?!\d)')

    for line in response_block.splitlines():
        # We only care about the part *after* the first colon.
        if ":" not in line:
            continue
        
        _, value = line.split(":", 1)
        # If the pattern is found in the value part of the line, it's a failure.
        if failure_pattern.search(value):
            return 0 # Failure detected, exit immediately
    
    return 1



def build_prompt_general(intent: str, method: str, history=None, last_html=None, additional_info=None, monitored_info=None, if_three_output=False, scaling_type="default") -> str:
    if method in ["WebRL", "AndroidGen"]:
        step_len = len(history)
        history = "\n".join(history)
        if isinstance(last_html, (list, tuple)):
            html_string = ""
            base_idx = step_len - len(last_html)
            for idx, html in enumerate(last_html):
                html_string += f"Step {base_idx + idx + 1}--\n{html}\n\n"
            last_html = html_string
    
    if method == "ProRe":
        if isinstance(history, list):
            history = "\n".join(history)
        
        if additional_info is None:
            additional_info = "No additional information."
        else:
            if isinstance(additional_info, list):
                additional_info = "\n".join(additional_info)

        if monitored_info:
            if isinstance(monitored_info, list):
                monitored_info = "\n".join(monitored_info)
                prompt = a3e_prompt_v5.format(intent=intent, history=history, last_html=last_html, additional_info=additional_info, monitored_info=monitored_info)

    return prompt


def predict_mm_until_ok(llm, content, images, overhead_measure=False, method="DigiRL") -> str:
    """
    Call llm.predict_mm(content, images) repeatedly until
    `extract_status(response_text)` is True or MAX_RETRY is hit.

    Returns
    -------
    response_text : str           # the non-empty, status-OK text
    Raises RuntimeError on exhaustion.
    """
    for attempt in range(1, MAX_RETRY + 1):
        if method in ["WebRL", "AndroidGen", "ProRe"]:
            response_text, _, ori_response = llm.predict(content)
        else:
            response_text, _, ori_response = llm.predict_mm(content, images)

        if response_text:
            if method in ["AndroidGen", "ProRe"] or extract_status(response_text):
                if overhead_measure:
                    return response_text, ori_response
                return response_text 

        print(f"[predict_mm] attempt {attempt} failed, retrying …")
        time.sleep(BACKOFF)    

    return ''


class Evaluator:
    def __init__(self, llm, method):
        self.llm = llm
        self.method = method

    def evaluate(self, intent: str, images: str, history=None, last_html=None, additional_info=None, monitored_info=None, run_idx=0, scaling_type="default") -> bool:
        prompt = build_prompt_general(intent, self.method, history, last_html, additional_info, monitored_info, scaling_type)
        if run_idx == 0:
            logging.warning(prompt)
            print(prompt)

        if self.method in ["WebRL", "AndroidGen"]:
            response_text = predict_mm_until_ok(self.llm, prompt, images=None, method=self.method)
        else:
            if "gemini" in self.llm.model_name:
                response_text = predict_mm_until_ok(self.llm, prompt, images, method=self.method)
            else:
                response_text, _, output = self.llm.predict_mm(prompt, images)
        
        logging.warning(f"--- {run_idx} Response:\n {response_text}\n\n")
        print(f"--- {run_idx} Response:\n {response_text}\n\n")
        
        if response_text == '':
            return None, response_text
            
        if self.method == "AndroidGen":
            flag = success_flag(response_text)
            return flag, response_text
        else:
            if extract_status(response_text) is not None and 'success' in extract_status(response_text).lower():
                return True, response_text
            elif extract_status(response_text) is not None and 'incomplete' in extract_status(response_text).lower():
                return 'incomplete', response_text
            return False, response_text


    def Chain_of_Chaims(self, policy_agent_trace, evaluator_agent_trace, intent, action_history, evaluator_agent_history, response_file, task):
        policy_claims_prompt, policy_images = build_simplified_claims_prompt(trace=policy_agent_trace, intent=intent, action_history=action_history, role="Policy")
        response, _, policy_ori_response = self.llm.predict_mm(policy_claims_prompt, policy_images)
        policy_claims = extract_claims_from_response(response, role="Policy")

        evaluator_claims_prompt, evaluator_images = build_simplified_claims_prompt(trace=evaluator_agent_trace, action_history=evaluator_agent_history, intent=intent, role="Evaluator")
        response, _, evalutor_ori_response = self.llm.predict_mm(evaluator_claims_prompt, evaluator_images)
        evalutor_claims = extract_claims_from_response(response, role="Evaluator")

        if not policy_claims or not evalutor_claims:
            with open(response_file, "a") as f:
                f.write(f"{task}: {-1}\n")

        prompt = assemble_claims_as_prompt(intent, policy_claims, evalutor_claims)
        
        print("=" * 50)
        print(prompt)
        
        # --- Ask final judge ---
        response_text, _, ori_response = self.llm.predict(prompt)
        print(response_text)
        print("=" * 50)
        # --- Evaluate result ---
        if not response_text or not response_text.strip():
            # No response from LLM → mark as invalid
            result_value = -1
        else:
            status = extract_status(response_text)
            if status and "success" in status.lower():
                result_value = True
            elif status and "failure" in status.lower():
                result_value = False
            else:
                result_value = "incomplete"

        return result_value, response_text

def load_images_from_folder(folder_path):
    """Load all jpg/png images into numpy arrays."""
    images = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith((".jpg", ".png", ".jpeg")) and '_ann' not in fname:
            fpath = os.path.join(folder_path, fname)
            try:
                img = np.array(Image.open(fpath).convert("RGB"))
                images.append(img)
            except Exception as e:
                print(f"⚠️ Failed to load {fpath}: {e}")
    return images


def check_correctness_from_logs(check_k, run_name, gt_path, llm_name="gemini", method="DigiRL", save_name="fn", gt_type="traj"):
    target_folder = f"saved/{run_name}"

    if llm_name == "gpt-4o":
        llm = infer.Gpt4Wrapper(llm_name, temperature=0)
    elif "gemini" in llm_name:
        llm = infer.GeminiGcpWrapper(llm_name, temperature=0)

    evaluator = Evaluator(llm=llm, method=method)

    save_folder = target_folder.replace("/record", "")
    results_file = f"{save_folder}/llm_{llm_name}_eval_{method}_images{check_k}_{save_name}.txt"
    response_file = f"{save_folder}/llm_{llm_name}_eval_{method}_images{check_k}_{save_name}_response.log"

    evaluated_tasks = set()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                parts = line.split(":", 1)
                if parts:
                    evaluated_tasks.add(parts[0].strip())

    
    # Loop through each subfolder = task trial
    for task_folder in tqdm(sorted(os.listdir(target_folder))):
        task_path = os.path.join(target_folder, task_folder)
        
        if not os.path.isdir(task_path):
            continue 
        
        if "_evaluation" not in task_folder:
            continue
        
        task_name = task_folder.split("_evaluation")[0]
        if "_0" in task_name:
            task_name = task_name.replace("_0", "")
        if task_name in evaluated_tasks:
            continue
            
        # pdb.set_trace()
        policy_path = os.path.join(target_folder, task_name)


        log_path = os.path.join(task_path, "app.log")

        if not os.path.exists(log_path):
            print(f"⚠️ No app.log found in {task_path}")
            continue

        # Extract trajectory info
        try:
            intent, action_history, monitored_info, final_ui_state, additional_ui_pages_text = extract_content_from_text(log_path)
        except Exception as e:
            with open(results_file, "a") as f:
                f.write(f"{task_name}: 0\n")
            print(f"⚠️ Failed to parse {log_path}: {e}")
            continue
        

        steps = re.split(r'(?=Step \d+-)', action_history)
        action_history = [s.strip() for s in steps if s.strip()]

        # Load images
        img_dir = os.path.join(policy_path, "screen_shot")
        images = load_images_from_folder(img_dir)
        
        if check_k == -1:
            check_images = images
        elif check_k > len(images):
            check_images = images[:]
        else:
            check_images = images[-check_k:]

        historical_summary = action_history
        
        additional_ui_pages = split_additional_ui_pages(additional_ui_pages_text)

        # UI representation
        if method in ["WebRL", "AndroidGen"]:
            if check_k < len(additional_ui_pages):
                last_html = final_ui_state
            else:
                last_html = final_ui_state
        else:
            last_html = final_ui_state

        # Evaluate
        # pdb.set_trace()
        result, response_text = evaluator.evaluate(intent, check_images, historical_summary, last_html)
        result_value = 1 if result else 0

        # Save logs
        with open(response_file, "a") as f:
            f.write(f"{task_name}: {response_text}\n")
        with open(results_file, "a") as f:
            f.write(f"{task_name}: {result_value}\n")

        print("✅ Evaluation result for:", task_name, "->", result_value)

    acc = 0
    f1 = 0
    return acc, f1


def check_correctness(check_k, run_name, gt_path, llm_name = "gemini", method="DigiRL", save_name="fn", gt_type="traj"):
    target_folder=f"saved/{run_name}"
    
    if llm_name == "gpt-4o":
        llm = infer.Gpt4Wrapper(llm_name, temperature=0)
    elif "gemini" in llm_name:
        llm = infer.GeminiGcpWrapper(llm_name, temperature=0)

    evaluator = Evaluator(llm=llm, method=method)
    results_file = f"{target_folder}/llm_{llm_name}_eval_{method}_images{check_k}_{save_name}.txt"
    response_file = f"{target_folder}/llm_{llm_name}_eval_{method}_images{check_k}_{save_name}_response.log"
    # print(response_file)
    # pdb.set_trace()
    target_traj_folder = os.path.join(target_folder, "task_info")
    evaluated_tasks = set()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                parts = line.split(":", 1)
                if parts:
                    evaluated_tasks.add(parts[0].strip())


    for task_path in tqdm(sorted(os.listdir(target_traj_folder))):
        if task_path.endswith("pkl.gz"):
            if '_0' in task_path:
                task = task_path.replace('_0.pkl.gz', '')
            else:
                task = task_path.replace('.pkl.gz', '')
            
            if task in evaluated_tasks:
                continue

            # if task not in fp_cases:
            #     continue
            
            trajectory_path = os.path.join(target_traj_folder, task_path)
            load_data = load_trajectory(trajectory_path)
            instruction = load_data[0]["goal"]

            try:
                root = load_data[0]['tree']
            except:
                with open(results_file, "a") as f:
                    f.write(f"{task}: 0\n")
                    continue

            _, correct_trace = dfs_max_reward([root])
            images = []
            # pdb.set_trace()
            for node in correct_trace:
                if node.state:
                    images.append(node.state["screenshot_raw"])

            for node in correct_trace[::-1]:
                if node.state and node.state['raw_ui_state']:
                    forest = node.state['raw_ui_state'].forest
                    break
            
            if method in ["WebRL", "AndroidGen"]:
                last_html = []
                for node in correct_trace:
                    if node.state and node.state['raw_ui_state']:
                        forest = node.state['raw_ui_state'].forest
                        html = turn_tree_to_html_input_v2(forest)
                        last_html.append(html)
            else:
                last_html = turn_tree_to_html_input_v2(forest)
            historical_summary = [n.node_info['summary'] for n in correct_trace[:correct_trace.index(node) + 1] if n.node_info['summary']]  # All actions up to this node
            historical_summary = [
                f"Step {i+1}- {action}"
                for i, action in enumerate(historical_summary)
            ]

            if check_k == -1:
                check_images = images
            elif check_k > len(images):
                check_images = images[:]
            else: 
                check_images = images[-check_k:]
        
            if method in ["WebRL", "AndroidGen"]:
                if check_k < len(images):
                    last_html = last_html[-check_k:]

            
            result, response_text = evaluator.evaluate(instruction, check_images, historical_summary, last_html)
            result_value = 1 if result else 0
        
            # pdb.set_trace()
            # logging.warning(f"task {task}: {response_text}")
            
            with open(response_file, "a") as f:
                f.write(f"{task}: {response_text}\n")

            with open(results_file, "a") as f:
                f.write(f"{task}: {result_value}\n")
            
            print("Evaluation result for task:", task, "->", result_value)
            # pdb.set_trace()
    if method == "ProRe":
        acc, f1, _, _ = compare_with_gt(results_file, gt_path, gt_type='a3j')
    else:
        acc, f1, _, _ = compare_with_gt(results_file, gt_path, gt_type=gt_type)
    return acc, f1
