from collections import Counter
from absl import logging
import pdb
import numpy as np
from abc import ABC, abstractmethod
import math
import os
import re
from PIL import Image
from tqdm import trange
from copy import deepcopy
from math import ceil
import torch
import sys
from textwrap import indent
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle

from android_world import constants
from android_world.agents.reward_model import MAX_LENGTH
from android_world.agents.general_evaluator import Evaluator
from android_world.agents import agent_utils, m3a
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.agents.vdroid import MCTSNode, VDroidAgent
from android_world.env import interface
from android_world.env import json_action
from typing import Generic, TypeVar, Optional, NamedTuple, Callable, Hashable, Type
from android_world.task_evals import task_eval
from html_representation.autodroid_repsentation import DeviceState
from html_representation.bbox_representation import turn_tree_to_group_bounding_boxes
from html_representation.html_representation import html_truncate, turn_tree_to_html_input, extract_actions_with_display_id_v2, turn_tree_to_html_input_v2
from util import dfs_max_reward, mark_frames_to_skip, polish_action, mask_irrelevant_info
from prompt_template import *
from typing import Union
import logging as logger
from absl import logging

Node_info = TypeVar("Node_info")
Action = TypeVar("Action")
Example = TypeVar("Example")

LARGE_SCORE_SHIFT = 1e3
SOFT_SCORE_SHIFT = 1e2


class ProRe(VDroidAgent):
    """
    A wrapper around SearchAgent to evaluate task completion.
    It modifies the goal to a verifier-oriented prompt asking
    for key information needed to determine success.
    """
    def __init__(
        self,
        env: interface.AsyncEnv,
        service_name: str,
        local_model_name: str,
        adapter_dir: str,
        actors = None,
        save_dir: str = None,
        goal: str = None,
        n_iters: int = 1,
        family: str = None,
        task: Type[task_eval.TaskEval] = None,
        summary_mode: str = "default",
        llm_name: str = "gemini-2.5-pro-preview-05-06",
        scaling_type: str = "default",
        evaluator_agent_name: str = "sa_gpt_llama31",
        trial_num: int = 0,
        num_actors: int = 2,
        **kwargs
    ):
        # Initialize the VDroidAgent with the modified goal
        super().__init__(
            env=env,
            actors=actors,
            llm_name="gpt-4o",
            adapter_dir=adapter_dir,
            save_dir=save_dir,
            service_name=service_name,
            local_model_name=local_model_name,
            goal=goal,
            task=task,
            n_iters=n_iters,
            family=family,
            summary_mode=summary_mode, 
            num_actors=num_actors,
            **kwargs  # pass through all other parameters
        )
        self.depth_limit = 10
        self.role = "evaluator"
        self.transition_pause = None

        if "gemini" in llm_name:
            llm = infer.GeminiGcpWrapper(llm_name, temperature=0.3 if scaling_type == "sequential" else 0)
        else:
            llm = infer.Gpt4Wrapper(llm_name, temperature=0.3 if scaling_type == "sequential" else 0)
        
        self.final_evaluator = Evaluator(llm=llm, method="ProRe")
        self.evaluator_agent_name = evaluator_agent_name

        self.policy_history = []
        self.replay_path = []
        self.original_goal = goal

        self.response_file = None
        self.configure_for_task(save_dir=save_dir, goal=goal, task=task, trial_num=trial_num)

    def configure_for_task(
        self,
        save_dir: str,
        goal: str,
        task: Type[task_eval.TaskEval] | None,
        trial_num: int = 0,
    ) -> None:
        """Updates task-specific fields when reusing the probing agent."""
        policy_save_dir = save_dir
        evaluation_save_dir = save_dir
        if trial_num > 0:
            policy_save_dir += f"_{trial_num}"
            evaluation_save_dir += f"_{trial_num}"

        evaluation_save_dir += "_evaluation"

        self.goal = goal
        self.original_goal = goal
        self.task = task
        self.save_dir = evaluation_save_dir
        self.save_path = os.path.join(evaluation_save_dir, "screen_shot/")
        self.policy_save_path = os.path.join(policy_save_dir, "screen_shot/")
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.policy_save_path, exist_ok=True)

        self.policy_history = []
        self.replay_path = []
        self.response_file = os.path.join(
            self.save_dir, f"llm_{self.final_evaluator.llm.model_name}_eval_ProRe_response.log"
        )

        log_file = os.path.join(self.save_dir, 'app.log')
        root_logger = logger.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        file_handler = logger.FileHandler(log_file)
        root_logger.addHandler(file_handler)


    def search(self,):
        self._output_cum_reward = -math.inf
        self._output_iter = None
        
        self.construct_root() ## go back to home screen and reset the root node.

        terminal_iter = False
        for idx in trange(self.n_iters, disable=True, desc='MCTS iteration', leave=False):
            self.iter_idx += 1
            path = self.iterate(self.root, self.iter_idx)
            for idx, node in enumerate(path):
                if node.is_terminal:
                    terminal_iter = True
                    break
                
                if idx == len(path) - 1 and node.children is not None:
                    fast_rewards = [child.fast_reward for child in node.children]
                    child = node.children[self.simulate_choice(fast_rewards)]

                    try:
                        converted_action = json_action.JSONAction(
                            **agent_utils.extract_json(child.action),
                        )
                        node.node_info['action_output_json'] = converted_action

                    except Exception as e:
                        converted_action = None
                        print('Failed to convert the output to a valid action.')
                        print(str(e))

                        node.state = node.parent.state.copy()
                        node.node_info['summary'] = (
                            'Can not parse the output to a valid action. Please make sure to pick'
                            ' the action from the list with required parameters (if any) in the'
                            ' correct JSON format!'
                        )
                        node.node_info['ui_elements'] = node.parent.node_info['ui_elements'].copy()
                        if node.parent.node_info['html_desc']:
                            node.node_info['html_desc'] = node.parent.node_info['html_desc']
                        # self._assign_action_failure_penalty(node)  ## we dont do penalty here to avoid double penalty

                    if converted_action is not None:
                        
                        if converted_action.action_type == 'status':
                            if converted_action.goal_status == 'infeasible':
                                print('Agent stopped since it thinks mission impossible.')
                            node.node_info['summary'] = 'Agent thinks the request has been completed.'
                            node.state = node.parent.state.copy()
                            node.is_terminal = True
                            node.node_info['ui_elements'] = node.parent.node_info['ui_elements'].copy()
                            if node.parent.node_info['html_desc']:
                                node.node_info['html_desc'] = node.parent.node_info['html_desc']
                            node.reward, node.score_details = self.reward(node.score_details['self_eval'], node.is_terminal)
                            node.score = node.reward
                            logging.warning(f"The final reward for the finished node is {node.reward} ")
                            
                            terminal_iter = True
                            break
            
            if terminal_iter:
                break


        if self.output_strategy == 'max_reward':
            self._output_cum_reward, self._output_iter = self._dfs_max_reward([self.root])
            if self._output_cum_reward == -math.inf:
                self._output_iter = None
        
        is_done = False
        cal_path = None
        if self._output_iter is not None:
            cal_path = self._output_iter
        else:
            cal_path = path

        images = []
        for node in cal_path:
            if node.is_terminal == True:
                is_done = True

            if node.state:
                images.append(node.state["screenshot_raw"])

        model_name = self.llm.model_name.lower()
        if 'llama-3.2' in model_name or 'llama-3.1' in model_name or 'deepseek' in model_name:
            torch.cuda.empty_cache()
        return is_done, images

    
    def construct_root(self):
        self.step_idx = 0
        self.path_summary = []
        self.history = []
        state = {
            'screenshot_raw': None,
            'screenshot_som': None,
            'orientation': None,
            'physical_frame_boundary': None,
            'logical_screen_size': None,
            'raw_ui_state': None,
        }
        node_info = {
            'ui_elements': None,
            'reason': None,
            'action_prompt': None,
            'action_output': None,
            'action_output_json': None,
            'action_reason': None,
            'action_raw_response': None,
            'summary_prompt': None,
            'summary': None,
            'summary_raw_response': None,
            'html_desc': None,
        }

        ui_state = self.env.get_state(wait_to_stabilize=False)
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary

        ui_elements = ui_state.ui_elements
        logical_screen_size = self.env.logical_screen_size
        state['screenshot_raw'] = ui_state.pixels.copy()
        state['raw_ui_state'] = ui_state
        before_screenshot = ui_state.pixels.copy()

        if self.input_type=="autodroid":
            html_desc = DeviceState(ui_state.forest).state_str
        elif self.input_type=="html":
            html_desc = turn_tree_to_html_input(ui_state.forest)
        elif self.input_type=="image":
            html_desc = None
        
        if self.family == "android_lab" and "cantook" in self.goal.lower():
            html_desc = html_truncate(self.llm.reward_model.tokenizer, html_desc, MAX_LENGTH)
        
        node_info['html_desc'] = html_desc

        available_actions = extract_actions_with_display_id_v2(ui_state.forest, refine_a11y_tree = self.family=="android_lab")
        
        for index, ui_element in enumerate(ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    before_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                    add_image_desc=self.add_image_desc
                )
        group_bounding_boxes = turn_tree_to_group_bounding_boxes(orientation, logical_screen_size, physical_frame_boundary, ui_state.forest)
        if self.add_image_desc:
            m3a_utils.apply_group_bouding_boxes(
                before_screenshot,
                group_bounding_boxes
            )
        
        state['screenshot_som'] = before_screenshot.copy()
        state['orientation'] = orientation
        state['physical_frame_boundary'] = physical_frame_boundary
        state['logical_screen_size'] = logical_screen_size
        state['available_actions'] = available_actions

        node_info['ui_elements'] = ui_elements
        
        self.root = MCTSNode(state=state, node_info=node_info, action=None, parent=None, calc_q=self.calc_q)
        self.store_screen(self.root, iter=1)
        return
    
    def repeat_corresponding_actions(self, path: list[MCTSNode], iter: int):
        new_path: list[MCTSNode] = []
        previous_new_node = None
        fail_to_exact_replay = False
        
        for node_idx, node in enumerate(path):
            # if node_idx == len(path) - 1:
            #     pdb.set_trace()
            if node.action is None and node.depth == 0:
                ui_state = self.get_post_transition_state()
                state = self._build_node_state_from_ui_state(ui_state)
                
                node_info = node.node_info
                node_info["ui_elements"] = ui_state.ui_elements
                node_info["html_desc"] = turn_tree_to_html_input_v2(ui_state.forest)
                
                new_root = MCTSNode(state=state, node_info=node_info, action=None, 
                                    parent=None, is_terminal = False,
                                    score=None, score_details=None, 
                                    calc_q=self.calc_q)
                
                self.policy_store_screen(new_root, iter)
                new_path.append(new_root)
                previous_new_node = new_root

                continue
            
            new_node = MCTSNode(state=None, node_info=node.node_info, action=None, 
                                parent=previous_new_node, is_terminal = False,
                                score=0, score_details=node.score_details, 
                                calc_q=self.calc_q)

            try:
                new_node.action = polish_action(node.action)
                converted_action = json_action.JSONAction(
                    **agent_utils.extract_json(new_node.action),
                )
                new_node.node_info['action_output_json'] = converted_action
                action_index = converted_action.index
                num_ui_elements = len(new_node.parent.node_info['ui_elements'])
            except Exception as e:  # pylint: disable=broad-exception-caught
                print('Failed to convert the output to a valid action.')
                print(str(e))

            if converted_action is not None:
                if converted_action.action_type == 'status':
                    if converted_action.goal_status == 'infeasible':
                        print('Agent stopped since it thinks mission impossible.')
                    new_node.node_info['summary'] = 'Agent thinks the request has been completed.'
                    new_node.state = new_node.parent.state.copy()
                    new_node.is_terminal = True
                    new_node.node_info['ui_elements'] = new_node.parent.node_info['ui_elements'].copy()
                    if new_node.parent.node_info['html_desc']:
                        new_node.node_info['html_desc'] = new_node.parent.node_info['html_desc']
                    new_node.reward, new_node.score_details = self.reward(new_node.score_details['self_eval'], new_node.is_terminal)
                    new_node.score = new_node.reward
                    logging.warning(f"The final reward for the finished node is {new_node.reward} ")
                    new_path.append(new_node)
                    previous_new_node = new_node
                    self.policy_store_screen(new_node, iter)
                    self.policy_history.append(new_node.node_info)
                    break

                else:
                    take_a_step = True
                    if (converted_action.action_type
                        in ['click', 'long_press', 'input_text', 'scroll']
                        and action_index is not None
                    ):
                        if action_index >= num_ui_elements:
                            print(
                                f'Index out of range, prediction index is {action_index}, but the'
                                f' UI element list only has {num_ui_elements} elements.'
                            )
                            
                            new_node.state = new_node.parent.state.copy()
                            new_node.node_info['summary'] = (
                                'The parameter index is out of range. Remember the index must be in'
                                ' the UI element list!'
                            )
                            new_node.node_info['ui_elements'] = new_node.parent.node_info['ui_elements'].copy()
                            if new_node.parent.node_info['html_desc']:
                                new_node.node_info['html_desc'] = new_node.parent.node_info['html_desc']
                            
                            take_a_step = False
                            new_node.is_terminal = True
                            new_node.reward, new_node.score_details = 0, None
                            new_node.score = new_node.reward
                            logging.warning(f"The final reward for the finished node is {new_node.reward} ")
                            new_path.append(new_node)
                            previous_new_node = new_node
                            self.policy_store_screen(new_node, iter)
                            self.policy_history.append(new_node.node_info)
                            fail_to_exact_replay = True
                            break
                        else:
                            m3a_utils.add_ui_element_mark(
                                new_node.parent.state['screenshot_raw'],
                                new_node.parent.node_info['ui_elements'][action_index],
                                action_index,
                                new_node.parent.state["logical_screen_size"],
                                new_node.parent.state["physical_frame_boundary"],
                                new_node.parent.state["orientation"],
                                add_image_desc=self.add_image_desc
                            )
                    
                    if take_a_step:
                        self.env.execute_action(converted_action)
                        ui_state = self.get_post_transition_state()
                        state = self._build_node_state_from_ui_state(ui_state)
                        new_node.state = state
                        new_node.node_info["ui_elements"] = ui_state.ui_elements
                        new_node.node_info["html_desc"] = turn_tree_to_html_input_v2(ui_state.forest)

                self.policy_store_screen(new_node, iter)
                self.policy_history.append(new_node.node_info)
                
            new_path.append(new_node)
            previous_new_node = new_node
        
        for idx, new_node in enumerate(new_path[:-1]):
            new_node.children = [new_path[idx + 1]]
        
        self.replay_path = new_path
        return fail_to_exact_replay
    
    def _build_node_state_from_ui_state(self, ui_state):
        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary
        after_ui_elements = ui_state.ui_elements
        after_screenshot = ui_state.pixels.copy()

        state = {
            'screenshot_raw': None,
            'screenshot_som': None,
            'orientation': None,
            'physical_frame_boundary': None,
            'logical_screen_size': None,
            'raw_ui_state': None,
        }

        state['screenshot_raw'] = after_screenshot.copy()
        for index, ui_element in enumerate(after_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    after_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                    add_image_desc=self.add_image_desc
                )
        group_bounding_boxes = turn_tree_to_group_bounding_boxes(orientation, logical_screen_size, physical_frame_boundary, ui_state.forest)
        if self.add_image_desc:
            m3a_utils.apply_group_bouding_boxes(
                after_screenshot,
                group_bounding_boxes
            )

        state['screenshot_som'] = after_screenshot.copy()
        state['orientation'] = orientation
        state['physical_frame_boundary'] = physical_frame_boundary
        state['logical_screen_size'] = logical_screen_size
        state['raw_ui_state'] = ui_state
        return state
        
    
    def repeat_trace_with_fresh_nodes(self, path: list[MCTSNode], iter: int) -> list[MCTSNode]:
        """
        Replays a recorded path of actions, but constructs a NEW list of nodes with fresh state/UI info
        from the device after each step. The action content is taken from the trace; indices are
        re-resolved against the current UI when needed.
        """
        new_path: list[MCTSNode] = []
        prev_new_node = None

        def _build_state_from_ui_state(ui_state):
            logical_screen_size = self.env.logical_screen_size
            orientation = self.env.orientation
            physical_frame_boundary = self.env.physical_frame_boundary
            after_ui_elements = ui_state.ui_elements
            after_screenshot = ui_state.pixels.copy()

            state = {
                'screenshot_raw': None,
                'screenshot_som': None,
                'orientation': None,
                'physical_frame_boundary': None,
                'logical_screen_size': None,
                'raw_ui_state': None,
            }

            # Mark elements onto the screenshot
            for index, ui_element in enumerate(after_ui_elements):
                if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                    m3a_utils.add_ui_element_mark(
                        after_screenshot,
                        ui_element,
                        index,
                        logical_screen_size,
                        physical_frame_boundary,
                        orientation,
                        add_image_desc=self.add_image_desc
                    )

            # Optional group boxes overlay
            group_bounding_boxes = turn_tree_to_group_bounding_boxes(
                orientation, logical_screen_size, physical_frame_boundary, ui_state.forest
            )
            if self.add_image_desc:
                m3a_utils.apply_group_bouding_boxes(after_screenshot, group_bounding_boxes)

            state['screenshot_raw'] = ui_state.pixels.copy()
            state['screenshot_som'] = after_screenshot.copy()
            state['orientation'] = orientation
            state['physical_frame_boundary'] = physical_frame_boundary
            state['logical_screen_size'] = logical_screen_size
            state['raw_ui_state'] = ui_state
            return state

        def _score_match(target, cand):
            """Simple exact/substring scoring over common attributes."""
            score = 0
            # Prefer exact id/desc/text matches; then substring; then class; bounds fallback
            # Existence helpers
            def norm(x): 
                return x.strip().lower() if isinstance(x, str) else x

            tid = norm(getattr(target, "resource_id", None) or getattr(target, "id", None))
            ttext = norm(getattr(target, "text", None))
            tdesc = norm(getattr(target, "content_desc", None) or getattr(target, "description", None))
            tclass = norm(getattr(target, "class_name", None) or getattr(target, "class", None))
            tbounds = getattr(target, "bounds", None)

            cid = norm(getattr(cand, "resource_id", None) or cand.get("resource-id") or cand.get("id"))
            ctext = norm(getattr(cand, "text", None) or cand.get("text"))
            cdesc = norm(getattr(cand, "content_desc", None) or cand.get("content-desc"))
            cclass = norm(getattr(cand, "class_name", None) or cand.get("class"))
            cbounds = cand.get("bounds")

            # Exacts
            if tid and cid and tid == cid: score += 8
            if tdesc and cdesc and tdesc == cdesc: score += 6
            if ttext and ctext and ttext == ctext: score += 5
            if tclass and cclass and tclass == cclass: score += 2

            # Substrings
            if ttext and ctext and (ttext in ctext or ctext in ttext): score += 2
            if tdesc and cdesc and (tdesc in cdesc or cdesc in tdesc): score += 2

            # Bounds overlap heuristic (very light)
            def area(b):
                if not b: return 0
                # Expect "[x1,y1][x2,y2]" or dict; tolerate both
                if isinstance(b, str) and "]" in b:
                    import re
                    ints = list(map(int, re.findall(r"\d+", b)))
                    if len(ints) == 4:
                        x1, y1, x2, y2 = ints
                    else:
                        return 0
                elif isinstance(b, dict):
                    x1, y1, x2, y2 = b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0)
                else:
                    return 0
                return max(0, x2 - x1) * max(0, y2 - y1)

            def iou(b1, b2):
                if not b1 or not b2: return 0.0
                # very tolerant parser as above
                def parse(b):
                    if isinstance(b, str) and "]" in b:
                        import re
                        ints = list(map(int, re.findall(r"\d+", b)))
                        if len(ints) == 4:
                            return ints
                        return None
                    elif isinstance(b, dict):
                        return [b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0)]
                    return None
                p1, p2 = parse(b1), parse(b2)
                if not p1 or not p2: return 0.0
                x1, y1, x2, y2 = p1
                x1b, y1b, x2b, y2b = p2
                inter_x1, inter_y1 = max(x1, x1b), max(y1, y1b)
                inter_x2, inter_y2 = min(x2, x2b), min(y2, y2b)
                iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
                inter = iw * ih
                if inter == 0: return 0.0
                a1, a2 = area(b1), area(b2)
                union = max(1, a1 + a2 - inter)
                return inter / union

            if tbounds and cbounds:
                j = iou(tbounds, cbounds)
                if j > 0.0:
                    score += 1 + min(2, int(j * 5))  # small bump up to +3

            return score

        def _resolve_action_index(converted_action, prev_ui_elements, current_ui_elements):
            """
            Try to ensure converted_action.index is valid for current_ui_elements.
            1) If existing index in range, keep it.
            2) Else, try to find best candidate using attributes from the ORIGINAL element
            (if we can locate it by prior index) OR from the action fields themselves.
            """
            idx = getattr(converted_action, "index", None)
            if isinstance(idx, int) and 0 <= idx < len(current_ui_elements):
                return idx  # already valid

            # Try to get a 'target signature'
            target_sig = None
            if isinstance(idx, int) and prev_ui_elements and 0 <= idx < len(prev_ui_elements):
                # Original element from old screen
                target_sig = prev_ui_elements[idx]
            else:
                # Construct a pseudo-target from action attributes
                class _T:
                    pass
                t = _T()
                # Fill what we commonly see on JSONAction
                for k in ("resource_id", "id", "text", "content_desc", "description", "class_name", "class", "bounds"):
                    setattr(t, k, getattr(converted_action, k, None))
                target_sig = t

            if not current_ui_elements:
                return None

            # Score all candidates and pick the best
            best_idx, best_score = None, -1
            for i, cand in enumerate(current_ui_elements):
                s = _score_match(target_sig, cand)
                if s > best_score:
                    best_idx, best_score = i, s

            # Require a minimal score to avoid wild clicks; tune as needed
            if best_idx is not None and best_score >= 3:
                return best_idx
            return None

        # Bootstrap: if the first node in the trace is a "screen-only" node (root), capture the current screen
        # to anchor the new path.
        if path and (path[0].action is None and path[0].depth == 0):
            root_node = MCTSNode(parent=None, depth=0)
            ui_state = self.get_post_transition_state()  # or a dedicated current_state getter
            root_node.state = _build_state_from_ui_state(ui_state)
            root_node.node_info = {
                'summary': 'Start screen (fresh capture).',
                'ui_elements': ui_state.ui_elements,
                'html_desc': getattr(path[0].node_info, 'html_desc', None) if isinstance(path[0].node_info, dict) else None,
            }
            # Optional fast reward if you track it for root
            root_node.score = 0.0
            root_node.score_details = {'self_eval': None}
            root_node.is_terminal = False
            self.policy_store_screen(root_node, iter)
            self.policy_history.append(root_node.node_info)
            new_path.append(root_node)
            prev_new_node = root_node

        # Replay each action node from the old path, minting a fresh node each time
        for old_idx, old_node in enumerate(path):
            if old_node.action is None and old_node.depth == 0:
                # already handled above
                continue

            new_node = MCTSNode(parent=prev_new_node, depth=(prev_new_node.depth + 1 if prev_new_node else old_node.depth))
            new_node.node_info = {}
            new_node.action = old_node.action

            converted_action = None
            try:
                # Keep the original action text, but normalize it once
                normalized = polish_action(old_node.action)
                converted_action = json_action.JSONAction(**agent_utils.extract_json(normalized))
                new_node.node_info['action_output_json'] = converted_action
            except Exception as e:
                print('Failed to convert the output to a valid action.')
                print(str(e))

            if converted_action is None:
                # Record and continue without executing
                new_node.node_info['summary'] = 'Invalid action JSON. Skipped.'
                # Keep the current screen as-is (no transition)
                ui_state = self.get_post_transition_state()
                new_node.state = _build_state_from_ui_state(ui_state)
                new_node.is_terminal = False
                self.policy_store_screen(new_node, iter)
                self.policy_history.append(new_node.node_info)
                new_path.append(new_node)
                prev_new_node = new_node
                continue

            # Handle status actions
            if converted_action.action_type == 'status':
                if converted_action.goal_status == 'infeasible':
                    print('Agent stopped since it thinks mission impossible.')
                new_node.node_info['summary'] = 'Agent reports completion.'
                # Carry over the current (fresh) screen
                ui_state = self.get_post_transition_state()
                new_node.state = _build_state_from_ui_state(ui_state)
                new_node.is_terminal = True
                # Reward from your evaluator
                new_node.reward, new_node.score_details = self.reward(new_node.score_details.get('self_eval') if hasattr(new_node, 'score_details') else None,
                                                                    new_node.is_terminal)
                new_node.score = new_node.reward
                logging.warning(f"The final reward for the finished node is {new_node.reward} ")
                self.policy_store_screen(new_node, iter)
                self.policy_history.append(new_node.node_info)
                new_path.append(new_node)
                prev_new_node = new_node
                continue

            # For interactive actions, re-resolve index if needed
            prev_ui = prev_new_node.node_info.get('ui_elements') if (prev_new_node and prev_new_node.node_info) else None
            curr_ui_state = self.get_post_transition_state()  # "pre" state to re-resolve against
            curr_ui = curr_ui_state.ui_elements

            if converted_action.action_type in ['click', 'long_press', 'input_text', 'scroll']:
                resolved_idx = _resolve_action_index(converted_action, prev_ui, curr_ui)
                if resolved_idx is None and getattr(converted_action, "index", None) is not None:
                    # Index unrecoverable; skip step but still record fresh screen/info
                    msg = (f"Index out of range and no suitable match for action index "
                        f"{converted_action.index}; UI has {len(curr_ui)} elements.")
                    print(msg)
                    new_node.node_info['summary'] = msg
                    # Keep current screen (no transition)
                    new_node.state = _build_state_from_ui_state(curr_ui_state)
                    new_node.is_terminal = False
                    self.policy_store_screen(new_node, iter)
                    self.policy_history.append(new_node.node_info)
                    new_path.append(new_node)
                    prev_new_node = new_node
                    continue

                # Update to resolved index (even if it was already valid)
                converted_action.index = resolved_idx

                # Optional: draw highlight on the *current* (pre-action) screen for the target
                try:
                    m3a_utils.add_ui_element_mark(
                        curr_ui_state.pixels,
                        curr_ui[resolved_idx],
                        resolved_idx,
                        self.env.logical_screen_size,
                        self.env.physical_frame_boundary,
                        self.env.orientation,
                        add_image_desc=self.add_image_desc
                    )
                except Exception:
                    pass  # don't fail the whole step for visualization errors

            # Execute the action against the environment
            try:
                self.env.execute_action(converted_action)
                post_ui_state = self.get_post_transition_state()
                new_node.state = _build_state_from_ui_state(post_ui_state)
                new_node.node_info['ui_elements'] = post_ui_state.ui_elements
                # You can set html_desc here if you have a renderer:
                # new_node.node_info['html_desc'] = self.render_html_desc(post_ui_state)
                new_node.is_terminal = False
            except Exception as e:
                new_node.node_info['summary'] = f'Execution failed: {e}'
                # Even on failure, store the freshest *pre* screen so downstream logic can continue
                new_node.state = _build_state_from_ui_state(curr_ui_state)
                new_node.is_terminal = False

            # Bookkeeping and append
            self.policy_store_screen(new_node, iter)
            self.policy_history.append(new_node.node_info)
            new_path.append(new_node)
            prev_new_node = new_node

        return new_path

    def policy_store_screen(self, node: MCTSNode, iter: int):
        if self.if_store_screen:
            pixels = node.state['screenshot_raw']
            img = Image.fromarray(np.uint8(pixels))
            img.save(self.policy_save_path + f"iter_{iter}_step{node.depth + 1}.jpg", 'JPEG')
            
            pixels_ann = node.state['screenshot_som']
            img_ann = Image.fromarray(np.uint8(pixels_ann))
            img_ann.save(self.policy_save_path + f"iter_{iter}_step{node.depth + 1}_ann.jpg", 'JPEG')
    
    def evaluate(self, agent_root, agent_history, agent_goal, task_successful=None, search_mode=True, policy_agent_name="VDroidAgent", scaling_type="default", evaluator_agent_name="ProRe", add_info_type = "raw"):

        if search_mode:
            verify_is_done, search_images = self.search()

        # check_k = 1
        
        _, agent_trace = dfs_max_reward([agent_root])
    
        # images = []
        # monitored_info = []
        
        # for idx, node in enumerate(agent_trace):
        #     if node.state:
        #         images.append(node.state["screenshot_raw"])

        #     if node == agent_trace[-1]:
        #         continue
            
        #     information = 'Policy Agent Step ' + str(idx+1) + '-\n' + node.node_info["ui_snapshot_note"] + '\n'
        #     print(information)
        #     logging.warning(information)
        #     monitored_info.append(information)
        
        # check_images = images[-check_k:] if check_k != -1 else images

        # if search_mode:
        #     check_images = check_images + search_images[1:-1] 

        agent_history = [step_info['summary'] for i, step_info in enumerate(agent_history)] 
        evaluator_agent_history = [step_info['summary'] for i, step_info in enumerate(self.history)]  
        # + ["The agent stops."]
        step_summary = ['Step ' + str(i+1) + '- ' + str(summary) for i, summary in enumerate(agent_history) if summary]

        additional_infomation = []

        _, evaluator_agent_trace = dfs_max_reward([self.root])
        for idx, node in enumerate(evaluator_agent_trace[1:-1]): 
            if node.state:
                raw_ui_state = node.state["raw_ui_state"].forest
                html_input = turn_tree_to_html_input_v2(raw_ui_state)
                additional_infomation.append('Step ' + str(idx + 1) + ' - UI Page:\n' + html_input + '\n') 
        
        if len(additional_infomation) == 0:
            additional_infomation.append("No additional information") 
        
        self.additional_infomation = additional_infomation
        
        valid_node = find_last_valid_state(agent_trace)
        
        if valid_node:
            last_html = turn_tree_to_html_input_v2(valid_node.state['raw_ui_state'].forest)
        else:
            last_html = "No valid state found"
            print("No valid state found in the agent trace.")
        
        result_list = []
        response_text_list = []


        result, response_text = self.final_evaluator.Chain_of_Chaims(agent_trace, evaluator_agent_trace, agent_goal, agent_history, evaluator_agent_history, self.response_file, self.task)
        
        if result == "incomplete":
            print("The evaluation result is incomplete, which means the evaluator agent did not collect enough evidence.")
            result_value = -1
        elif result == None:
            print("No response from LLM.")
            result_value = -1
        elif result:
            result_value = 1
        else:
            result_value = 0
        
        result_list.append(result_value)
        response_text_list.append(response_text)

        vote_counts = Counter(result_list)
        final_result, count = vote_counts.most_common(1)[0]

        results_file = os.path.join(self.save_dir, "evaluation_results.txt")
        with open(results_file, "a") as f:
            f.write(f"Agent: {final_result}\n")
            f.write(f"GT: {task_successful}\n")

        return final_result, response_text_list


def pretty_ui_snapshot(json_str: str) -> str:
    """
    Convert the JSON returned under 'Key Informations:' into a concise,
    judge-friendly summary.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return "‼️  Extracted data is not valid JSON."

    lines: list[str] = []

    # ---------- Visible elements ----------
    elems = data.get("visible_elements", [])
    lines.append("Key Visible Elements:")
    for idx, el in enumerate(elems, 1):
        role  = el.get("role", "?")
        label = el.get("label", "")
        # any extra keys the model decided to add
        extras = {k: v for k, v in el.items() if k not in ("role", "label")}
        extra_str = ", ".join(f"{k}: {v}" for k, v in extras.items())
        bullet = f"{idx}. {role} — “{label}”"
        if extra_str:
            bullet += f"  ({extra_str})"
        lines.append(bullet)

    # ---------- Missing expected ----------
    missing = data.get("missing_expected", [])
    if missing:
        lines.append("\nMissing Expected UI Elements:")
        lines.extend(f"{idx}. {m}" for idx, m in enumerate(missing, 1))

    # ---------- Compressed summary ----------
    summary = data.get("compressed_summary", "")
    if summary:
        lines.append(f"\nCompressed Summary: {summary}")

    return "\n".join(lines)


def extract_step_information(policy_agent_trace, llm, original_goal, measure_overhead=False):
    """
    This function iterates over the evaluator_agent's trace, generates a prompt for each step, 
    and asks the LLM to extract helpful information about the task completion at each step.
    """
    # Initialize a list to collect the extracted information
    extracted_info = []
    prompts_token = []
    completions_token = []
    thinkings_token = []
    for idx, node in enumerate(policy_agent_trace):
        if node.state:

            raw_ui_state = node.state["raw_ui_state"].forest
            screenshot = node.state["screenshot_raw"]

            html_input = turn_tree_to_html_input_v2(raw_ui_state)

            # Generate a prompt to ask the LLM for helpful information from the current step
            prompt = information_compression_prompt.format(
                html_input=html_input,
                goal=original_goal, 
            )

            step_info, _, ori_response = llm.predict_mm(prompt, [node.state["screenshot_raw"]])

            step_info = step_info.replace("**", '')

            JSON_BLOCK_RE = re.compile(
                r"Key Informations:\s*"  # Literal header
                r"(?:"                  # Start of a non-capturing group for the alternatives
                    # Alternative 1: Fenced JSON block
                    r"```(?:json)?\s*"   # Opening fence (optional 'json' language specifier)
                    r"(\{.*?\})"         # Capture Group 1: The JSON content (non-greedy)
                    r"\s*```"            # Closing fence
                r"|"                    # OR
                    # Alternative 2: Bare JSON object
                    r"(\{.*\})"          # Capture Group 2: The JSON content (greedy, captures to the end)
                r")",
                re.DOTALL | re.IGNORECASE,
            )

            match = JSON_BLOCK_RE.search(step_info)

            if match:
                # Extract and return the final answer, clean any leading/trailing whitespace
                # key_info = match.group(1).strip()
                key_info = match.group(1) or match.group(2)
                key_info = key_info.strip()
                ui_snapshot_note = pretty_ui_snapshot(key_info)
            else:
                # If no match is found, return an error or empty string
                ui_snapshot_note = "No information extracted."
            
            node.node_info["ui_snapshot_note"] = ui_snapshot_note
            node.node_info["raw_extracted_key_information"] = key_info

            # pdb.set_trace()
            information = 'Policy Agent Step ' + str(idx+1) + '-\n' + ui_snapshot_note + '\n'
            print(information)
            logging.warning(information)
            extracted_info.append(information)

            llm_name = llm.model_name

            try:
                if 'gpt' in llm_name:
                    prompt_token = ori_response.usage.prompt_tokens
                    completion_token = ori_response.usage.completion_tokens
                    thinking_token = 0
                elif 'gemini' in llm_name:
                    prompt_token = ori_response.usage_metadata.prompt_token_count
                    completion_token = ori_response.usage_metadata.candidates_token_count
                    thinking_token = ori_response.usage_metadata.thoughts_token_count
            except Exception as e:
                print(f"Error extracting token usage: {e}")
                prompt_token = np.nan
                completion_token = np.nan
                thinking_token = np.nan

            prompts_token.append(prompt_token)
            completions_token.append(completion_token)
            thinkings_token.append(thinking_token)
            
    if measure_overhead:
        return extracted_info, prompts_token, completions_token, thinkings_token
    return extracted_info



def extract_step_information_v2(policy_agent_trace, llm, original_goal, measure_overhead=False, agent_type="Policy", monitor_type="skip", monitored_info=None):
    """
    This function iterates over the evaluator_agent's trace, generates a prompt for each step, 
    and asks the LLM to extract helpful information about the task completion at each step.
    """
    # Initialize a list to collect the extracted information
    extracted_info = []
    prompts_token = []
    completions_token = []
    thinkings_token = []

    skip_flags, _ = mark_frames_to_skip(policy_agent_trace)
    if monitored_info is not None:
        steps = re.split(r'(?=Policy Agent Step \d+-)', monitored_info.strip())

        # Remove empty strings if any
        monitored_info  = [s.strip() for s in steps if s.strip()]
        
    for idx, node in enumerate(policy_agent_trace):
        if node.state:
            
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

            screenshot = node.state["screenshot_raw"]

            if skip_flags[idx]:
                node.node_info["ui_snapshot_note"] = f"Skipped: duplicated or home screen."
                node.node_info["raw_extracted_key_information"] = None
                extracted_info.append(f"Policy Agent Step {idx + 1} skipped: duplicated or home screen.\n")
                continue
            
            if monitor_type == "select":
                # ---- Step 1: Ask if this page is helpful ----
                helpfulness_prompt = page_helpfulness_prompt.format(
                    goal=original_goal
                )

                helpful_response, _, ori_helpful_resp = llm.predict_mm(
                    helpfulness_prompt, [screenshot]
                )
                
                # Default values
                is_helpful = True
                reason = "Parsing failed"

                print(helpful_response)

                try:
                    # Regex to capture Analysis and Helpful sections
                    HELPFUL_RE = re.compile(
                        r"Analysis:\s*(.+?)\s*Helpful:\s*(true|false)",
                        re.DOTALL | re.IGNORECASE
                    )
                    match = HELPFUL_RE.search(helpful_response)

                    if match:
                        analysis_text = match.group(1).strip()
                        helpful_str = match.group(2).lower()
                        is_helpful = helpful_str == "true"
                        reason = analysis_text  # reuse the analysis as the "reason"
                    else:
                        is_helpful = True
                        reason = "No match for Analysis/Helpful format" ## we keep it for accuracy.
                except Exception as e:
                    is_helpful = False
                    reason = f"Error parsing response: {e}"
                
                if not is_helpful:
                    # Mark and skip this node
                    node.node_info["ui_snapshot_note"] = f"Skipped: not helpful for evaluation."
                    node.node_info["raw_extracted_key_information"] = None
                    extracted_info.append(f"Step {idx + 1} skipped: not helpful for evaluation. Reason {reason}")
                    continue

            
            if monitored_info is not None and idx < len(monitored_info):
                information = monitored_info[idx] + '\n'
                if "No information extracted" not in information:
                    print(information)
                    logging.warning(information)
                    extracted_info.append(information)
                    continue
                
            if "ui_snapshot_note" in node.node_info:
                ui_snapshot_note = node.node_info["ui_snapshot_note"]
                information = f'{agent_type} Agent Step ' + str(idx+1) + '-\n' + ui_snapshot_note + '\n'
                print(information)
                logging.warning(information)
                extracted_info.append(information)
                continue
            
            
            # Generate a prompt to ask the LLM for helpful information from the current step
            prompt = information_compression_prompt.format(
                html_input=html_input,
                goal=original_goal, 
            )

            step_info, _, ori_response = llm.predict_mm(prompt, [node.state["screenshot_raw"]])

            step_info = step_info.replace("**", '')

            JSON_BLOCK_RE = re.compile(
                r"Key Informations:\s*"  # Literal header
                r"(?:"                  # Start of a non-capturing group for the alternatives
                    # Alternative 1: Fenced JSON block
                    r"```(?:json)?\s*"   # Opening fence (optional 'json' language specifier)
                    r"(\{.*?\})"         # Capture Group 1: The JSON content (non-greedy)
                    r"\s*```"            # Closing fence
                r"|"                    # OR
                    # Alternative 2: Bare JSON object
                    r"(\{.*\})"          # Capture Group 2: The JSON content (greedy, captures to the end)
                r")",
                re.DOTALL | re.IGNORECASE,
            )

            match = JSON_BLOCK_RE.search(step_info)

            if match:
                # Extract and return the final answer, clean any leading/trailing whitespace
                # key_info = match.group(1).strip()
                key_info = match.group(1) or match.group(2)
                key_info = key_info.strip()
                ui_snapshot_note = pretty_ui_snapshot(key_info)
            else:
                # If no match is found, return an error or empty string
                ui_snapshot_note = "No information extracted."
            
            # pdb.set_trace()
            node.node_info["ui_snapshot_note"] = ui_snapshot_note
            node.node_info["raw_extracted_key_information"] = key_info

            # pdb.set_trace()
            information = f'{agent_type} Agent Step ' + str(idx+1) + '-\n' + ui_snapshot_note + '\n'
            print(information)
            logging.warning(information)
            extracted_info.append(information)

            llm_name = llm.model_name

            try:
                if 'gpt' in llm_name:
                    prompt_token = ori_response.usage.prompt_tokens
                    completion_token = ori_response.usage.completion_tokens
                    thinking_token = 0
                elif 'gemini' in llm_name:
                    prompt_token = ori_response.usage_metadata.prompt_token_count
                    completion_token = ori_response.usage_metadata.candidates_token_count
                    thinking_token = ori_response.usage_metadata.thoughts_token_count
            except Exception as e:
                print(f"Error extracting token usage: {e}")
                prompt_token = np.nan
                completion_token = np.nan
                thinking_token = np.nan

            prompts_token.append(prompt_token)
            completions_token.append(completion_token)
            thinkings_token.append(thinking_token)
            
    if measure_overhead:
        return extracted_info, prompts_token, completions_token, thinkings_token
    return extracted_info


def find_last_valid_state(agent_trace):
    """
    Iterates backward through the agent trace to find the last valid state (where state is not None).
    
    Args:
    - agent_trace (list): The list of agent trace nodes to check.
    
    Returns:
    - valid_node (object): The node with the last valid state, or None if no valid state is found.
    """
    # Iterate backward through the agent trace to find the last node with a non-None state
    for node in reversed(agent_trace):
        if node.state is not None:
            return node
    return None  # Return None if no valid state is found

import gzip

if __name__ == '__main__':
    llm_name = "gpt-4o"
    llm = infer.Gpt4Wrapper(llm_name, temperature=0)
    data_path = "saved/sa_gpt_llama31_41_mm_ability_scale/task_info/AudioRecorderRecordAudio_0.pkl.gz"
    with gzip.open(data_path, "rb") as f:
        load_data = pickle.load(f)

    original_goal = load_data[0]['goal']
    root = load_data[0]['tree']
    _, evaluator_agent_trace = dfs_max_reward([root])

    extracted_info = extract_step_information_v2(evaluator_agent_trace, original_goal=original_goal, llm=llm)
