import gzip
import pickle
from MCTS.mcts_node import MCTSNode
from android_world.agents import m3a_utils
from util import load_trajectory
import os 
import pdb
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import json

from util import polish_action

def extract_html_block(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    # Find the outermost <div> that wraps everything
    first_div = soup.find("div")
    if first_div:
        return str(first_div)
    return ""


def extract_ui_elements(text: str):
    pattern = r'UI element \d+: {.*?}(?=\n|$)'  # Match each UI element line
    matches = re.findall(pattern, text)
    return matches


def construct_mcts_tree(episode_data, agent_name="ui_tars"):
    num_steps = len(episode_data['step_number'])
    root = None
    last_node = None
    nodes = []

    for t in range(num_steps):
        # if t == num_steps-1:
        #     # Last step, no action prompt or action output
        #     action_prompt = None
        #     ui_elements = []
        # else:
        action_prompt = episode_data.get('action_prompt', [None]*num_steps)[t]
        ui_elements = extract_ui_elements(action_prompt)
        raw_state = episode_data.get('raw_ui_state', [None]*num_steps)[t]
        if raw_state:
            ori_ui_elements = raw_state.ui_elements
        else:
            ori_ui_elements = []

        ui_elements_txt = '\n'.join(ui_elements)

        state = {
            'screenshot_raw': episode_data.get('raw_screenshot', [None]*num_steps)[t],
            'screenshot_som': episode_data.get('before_screenshot_with_som', [None]*num_steps)[t],
            'orientation': episode_data.get('orientation', [None]*num_steps)[t],
            'physical_frame_boundary': episode_data.get('physical_frame_boundary', [None]*num_steps)[t],
            'logical_screen_size': episode_data.get('logical_screen_size', [None]*num_steps)[t],
            'raw_ui_state': raw_state,
        }
        # pdb.set_trace()
        # Determine action: for t >= 1 only
        if t > 0:
            if agent_name != "ui_tars":
                reason, action = m3a_utils.parse_reason_action_output(
                    episode_data['action_output'][t - 1]
                )
            else:
                action = episode_data['action_output'][t - 1]

            if action is None:
                action = "None"
            else:
                action = polish_action(action)
        else:
            reason, action = None, None

        node_info = {
            'ui_elements': ori_ui_elements,
            'reason': reason,
            'html_desc': ui_elements_txt,
            'action_prompt': episode_data.get('action_prompt', [None]*num_steps)[t],
            'action_output': episode_data.get('action_output', [None]*num_steps)[t],
            'action_output_json': episode_data.get('action_output_json', [None]*num_steps)[t],
            'action_reason': episode_data.get('action_reason', [None]*num_steps)[t],
            'action_raw_response': episode_data.get('action_raw_response', [None]*num_steps)[t],
            'summary_prompt': episode_data.get('summary_prompt', [None]*num_steps)[t],
            'summary': episode_data.get('summary', [None]*num_steps)[t],
            'summary_raw_response': episode_data.get('summary_raw_response', [None]*num_steps)[t],
        }
        
        node = MCTSNode(
            state=state,
            node_info=node_info,
            action=action,
            parent=last_node,
            fast_reward=0.0,
            is_terminal=False
        )
        node.score_details['self_eval'] = 0

        if last_node is not None:
            if last_node.children is None:
                last_node.children = []
            last_node.children.append(node)
        else:
            root = node  # first node is root, has no action

        nodes.append(node)
        last_node = node

    node = MCTSNode(
            state=state,
            node_info=node_info,
            action='{"action_type": "status", "goal_status": "complete"}',
            parent=last_node,
            fast_reward=0.0,
            is_terminal=True
        )
    node.score_details['self_eval'] = 0
    nodes.append(node)
    if last_node.children is None:
        last_node.children = []
    pdb.set_trace()
    last_node.children.append(node)
    # we manually set the last node as terminal
    # pdb.set_trace()
    return root, nodes


def construct_from_history(history, agent_name):
    nodes = []
    root = None
    last_node = None
    for idx, step_data in enumerate(history):
        state = {
                'screenshot_raw': step_data['raw_screenshot'],
                'screenshot_som': step_data['before_screenshot_with_som'],
                'orientation': step_data['orientation'],
                'physical_frame_boundary': step_data['physical_frame_boundary'],
                'logical_screen_size': step_data['logical_screen_size'],
                'raw_ui_state': step_data['raw_ui_state'],
            }
        
        if idx > 0:
            if "tars" in agent_name or "MobileGPT" in agent_name or "AutoDroid" in agent_name:
                action = step_data['action_output']
            else:
                reason, action = m3a_utils.parse_reason_action_output(
                    step_data['action_output']
                )
                action = polish_action(action)
        else:
            reason, action = None, None

        action_prompt = step_data['action_prompt']
        ui_elements = extract_ui_elements(action_prompt)
        raw_state = step_data['raw_ui_state']
        if raw_state:
            ori_ui_elements = raw_state.ui_elements
        else:
            ori_ui_elements = []

        ui_elements_txt = '\n'.join(ui_elements)
            
        node_info = {
            'ui_elements': ori_ui_elements,
            'reason': reason,
            'html_desc': ui_elements_txt,
            'action_prompt': step_data['action_prompt'],
            'action_output': step_data['action_output'],
            'action_output_json': step_data['action_output_json'],
            'action_reason': step_data['action_reason'],
            'action_raw_response': step_data['action_raw_response'],
            'summary_prompt': step_data['summary_prompt'],
            'summary': step_data['summary'],
            'summary_raw_response': step_data['summary_raw_response'],
        }

        node = MCTSNode(
            state=state,
            node_info=node_info,
            action=action,
            parent=last_node,
            fast_reward=0.0,
            is_terminal=False
        )
        node.score_details['self_eval'] = 0

        if last_node is not None:
            if last_node.children is None:
                last_node.children = []
            last_node.children.append(node)
        else:
            root = node  # first node is root, has no action

        nodes.append(node)
        last_node = node

    try:
        state = {
                'screenshot_raw': history[-1]['raw_screenshot'],
                'screenshot_som': history[-1]['before_screenshot_with_som'],
                'orientation': history[-1]['orientation'],
                'physical_frame_boundary': history[-1]['physical_frame_boundary'],
                'logical_screen_size': history[-1]['logical_screen_size'],
                'raw_ui_state': history[-1]['raw_ui_state'],
            }
    except:
        print("Error in setting last state, using empty state")
        state = None
        node_info = None

    # pdb.set_trace()
    node = MCTSNode(
            state=state,
            node_info=node_info,
            action='{"action_type": "status", "goal_status": "complete"}',
            parent=last_node,
            fast_reward=0.0,
            is_terminal=True
        )
    node.score_details['self_eval'] = 0
    nodes.append(node)
    if last_node.children is None:
        last_node.children = []
    last_node.children.append(node)

    return root, nodes


if __name__ == "__main__":
    folder = "saved/m3a_gpt4o_Round110k_Bench_eval/task_info/"
    count = 0
    for file in tqdm(os.listdir(folder)):
        if not file.endswith(".pkl.gz"):
            continue
        count += 1
        policy_agent_traj_save_dir = os.path.join(folder, file)
        try:
            load_data = load_trajectory(policy_agent_traj_save_dir)
        except:
            print(f"{file} is corrupted")
    

        if not load_data:
            print(f"Empty file: {file}")
            continue

        if 'tree' in load_data[0] and load_data[0]['tree'] is not None:
            print(f"with tree in {file}, skipping")
            continue
        
        if str(load_data[0]["episode_data"]) == 'nan':
            print(f"Episode data is nan in {file}")
            continue

        episode_data = load_data[0]["episode_data"]
        
        root_node, all_nodes = construct_mcts_tree(episode_data)

        with gzip.open(policy_agent_traj_save_dir, "wb") as f:
            load_data[0]['tree'] = root_node
            pickle.dump(load_data, f)

        print(f"Loaded {file}, tree root id: {root_node.id}, total steps: {len(all_nodes)}")
