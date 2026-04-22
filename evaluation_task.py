import re
from android_world.agents import agent_utils, base_agent, infer
from android_world.agents.general_evaluator import predict_mm_until_ok

analysis_goal_prompt = (
    "You are an expert in mobile-UI task verification.\n"
    "There are two agents:\n\n"
    "• The **Task-Execution Agent** already attempted the task.\n"
    "• The **Information-Collection Agent** (V-Droid) ONLY navigates the UI to gather "
    "*evidence* that the task was or wasn’t completed.\n"
    "  It does **NOT** repeat the task; it just locates proof (screens, labels, icons).\n\n"
    "Your job:\n"
    "1. Write some analysis explaining what UI evidence/status would confirm the task is done.\n"
    "2. Output ONE concise goal (≤ 20 words) that tells the information-collection agent "
    "exactly what proof to look for.\n\n"
    "The goal must sound like the examples below—short, direct, and in the same tone.\n\n"
    "### Style Examples\n"
    "\"What is the cheapest flight from Los Angeles to Tokyo using Skyscanner?\"\n"
    "\"What are the 1M to 3M GBP to EUR exchange rates?\"\n"
    "\"go to settings and make weeks start on Monday in simple calendar\"\n"
    "\"Mark Hamlet as read in Cantook.\"\n\n"
    "### Your turn\n"
    "Original task: {goal}\n\n"
    "Respond with the following format:\n"
    "Analysis: <some tasks-related analysis>\n"
    "Goal: <concise goal matching the style examples>\n"
    "Do not add anything else."
)


analysis_goal_prompt_revised = (
    "You are an expert in mobile-UI task verification.\n"
    "There are two agents:\n\n"
    "• The **Policy Agent** already attempted the task.\n"
    "• The **Evaluator Agent** ONLY navigates the UI to gather "
    "*evidence* that the task was or wasn’t completed.\n"
    "  It does **NOT** repeat the task; it just locates proof (screens, labels, icons).\n\n"
    "Your job:\n"
    "1. Write some analysis explaining what UI evidence/status would confirm the task is done.\n"
    "2. Output ONE concise goal (≤ 20 words) that tells the evaluator agent "
    "exactly what proof to look for.\n\n"
    "The goal must sound like the examples below—short, direct, and in the same tone.\n\n"
    "### Style Examples\n"
    "\"What is the cheapest flight from Los Angeles to Tokyo using Skyscanner?\"\n"
    "\"What are the 1M to 3M GBP to EUR exchange rates?\"\n"
    "\"go to settings and make weeks start on Monday in simple calendar\"\n"
    "\"Mark Hamlet as read in Cantook.\"\n\n"
    "### Your turn\n"
    "Original task: {goal}\n\n"
    "A previous evaluation goal was:\n"
    "{previous_goal}\n\n"
    "The Evaluator Agent collected the following, which are insufficient to confirm completion:\n"
    "{collected_info}\n\n"
    "Revise the evaluation goal based on the previous evaluation goal and the original task:\n"
    "Respond exactly as:\n"
    "Analysis: <outline the exact UI evidence needed, pinpoint why the earlier collection failed, and suggest how to refine the evaluation goal for comprehensive verification>\n"
    "Goal: <revised concise goal>\n"
    "Do not add anything else."
)


analysis_goal_prompt_revised_v2 = analysis_goal_prompt_revised_v4 = (
    "You are an expert in mobile-UI task verification.\n"
    "There are two agents:\n\n"
    "• The **Policy Agent** already attempted the task.\n"
    "• The **Evaluator Agent** ONLY navigates the UI to gather "
    "*evidence* that the task was or wasn’t completed.\n"
    "  It does **NOT** repeat the task; it just locates proof (screens, labels, icons).\n\n"
    "Your job:\n"
    "1. Write some analysis explaining what UI evidence/status would confirm the task is done.\n"
    "2. Output ONE concise goal (≤ 20 words) that tells the evaluator agent "
    "exactly what proof to look for.\n\n"
    "The goal must sound like the examples below—short, direct, and in the same tone.\n\n"
    "### Style Examples\n"
    "\"What is the cheapest flight from Los Angeles to Tokyo using Skyscanner?\"\n"
    "\"What are the 1M to 3M GBP to EUR exchange rates?\"\n"
    "\"go to settings and make weeks start on Monday in simple calendar\"\n"
    "\"Mark Hamlet as read in Cantook.\"\n\n"
    "### Guidelines\n"
    "• Always focus on *evidence* of task completion, not repeating the task.\n"
    "• If collected info is insufficient, refine the goal so it is more precise and comprehensive.\n"
    "• If the Evaluator Agent’s history shows *loops* (e.g., scrolling down repeatedly at bottom of page), "
    "revise the goal to avoid the loop. Suggest an opposite or alternative action "
    "(e.g., scroll up, switch view, open details tab) that can reveal missing evidence.\n"
    "• If the previous goal was too weak or narrow (e.g., “find one file”), strengthen it "
    "by making the requirement broader or more explicit (e.g., “list all files in the folder”).\n"
    "• Always learn from the failure of the previous attempt — the refined goal must correct the weakness.\n"
    "• Keep the revised goal direct, actionable, and ≤ 20 words.\n\n"
    "***Your turn***\n"
    "### Task Information\n"
    "Original task: {goal}\n\n"
    "The previous evaluation goal was:\n"
    "{previous_goal}\n\n"
    "### Evaluator Agent Action History:\n"
    "{history}\n\n"
    "### Collected Observations (by Evaluator Agent)\n"
    "{collected_info}\n\n"
    "Revise the evaluation goal based on the previous evaluation goal and the original task:\n"
    "Respond exactly as:\n"
    "Analysis: <outline the exact UI evidence needed, pinpoint why the earlier collection failed "
    "(including looping/redundant or weak/narrow instructions), and suggest how to refine the evaluation goal for comprehensive verification>\n"
    "Goal: <revised concise goal>\n"
    "Do not add anything else."
)



def probing_tasks_generation(goal, llm_name, modified_goal_old=None, additional_collected_infor_old=[], history_old=[], type="llm", overhead_measure=False):
    if type == "rule":
        eval_goal = f"What is the key information to verify whether the task '{goal}' is completed?"
        prompt = "rule-based generation"
        response = "rule-based generation"
        return eval_goal, prompt, response

    if llm_name == "gpt-4o" or llm_name == "deepseek-r1":
        llm = infer.Gpt4Wrapper(llm_name, temperature=0)
    elif "gemini" in llm_name:
        llm = infer.GeminiGcpWrapper(llm_name, temperature=0)
    
    if modified_goal_old is not None and len(additional_collected_infor_old) > 0:
        collected_info = "\n".join(additional_collected_infor_old)
        history = "\n".join(history_old)
        prompt = analysis_goal_prompt_revised_v2.format(goal=goal, previous_goal=modified_goal_old, collected_info=collected_info, history=history)
    else:
        prompt = analysis_goal_prompt.format(goal=goal)
    # print(prompt)
    if "gemini" in llm_name:
        ori_response = None
        response = predict_mm_until_ok(llm, prompt, images=None, method="ProRe")
    else:
        response, _, ori_response = llm.predict(prompt)
    # print(response)

    match = re.search(r"^Goal:\s*(.+)$", response, flags=re.MULTILINE)
    if not match:
        eval_goal = f"What is the key information to verify whether the task '{goal}' is completed?"
    else:
        eval_goal = match.group(1).strip()
    if overhead_measure:
        return eval_goal, prompt, response, ori_response
    return eval_goal, prompt, response