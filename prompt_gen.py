from generate_tools import LLMCALL
import utils

import numpy as np
import logging
logging.basicConfig(filename='Inheritance.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import random
import os
import io
import pickle
from collections import defaultdict
import json
import torch
import time


    
class TreeNode:
    def __init__(self, prompt, node_id=''):
        self.prompt = prompt
        self.left = None
        self.middle = None 
        self.right = None
        self.node_id = node_id


def perform_action(node, action, model_e, topic, node_id):
    max_new_tokens = 100

    generate_prompt = utils.construct_action_prompt(action = action, question = node.prompt, topic = topic)
    generate_question = LLMCALL(model_name = model_e, stop_sequences = None, max_new_tokens = max_new_tokens).qwen_api(question = generate_prompt)
    logging.info(80 * '=')
    logging.info(f'Action:{action}')
    logging.info(80 * '=')
    logging.info(f'prompt:{generate_question}')
    return TreeNode(prompt = generate_question, node_id = node_id)

def build_tree(node, depth, model_e, topic, current_depth = 0, position = 'root'):
    if depth == 0:
        return
    
    left_id = f"{str(node.node_id)}L{str(current_depth+1)}"
    middle_id = f"{str(node.node_id)}M{str(current_depth+1)}"
    right_id = f"{str(node.node_id)}R{str(current_depth+1)}"
    
    # 执行分形动作A
    node.left = perform_action(node, 'ask', model_e = model_e, topic = topic, node_id=left_id)
    time.sleep(4)  # 延迟执行
    build_tree(node.left, depth - 1, model_e, topic, current_depth + 1, 'left')
    
    # 执行分形动作B
    node.middle = perform_action(node, 'expend', model_e = model_e, topic = topic, node_id=middle_id)
    time.sleep(4)  # 延迟执行
    build_tree(node.middle, depth - 1, model_e, topic, current_depth + 1, 'middle')
    
    # 执行分形动作C
    node.right = perform_action(node, 'negative', model_e = model_e, topic = topic, node_id=right_id)
    time.sleep(4)  # 延迟执行
    build_tree(node.right, depth - 1, model_e, topic, current_depth + 1, 'left')

def tree_to_dict(node):
    if node is None:
        return None
    return {
        'ID': node.node_id,
        'prompt': node.prompt,
        'left': tree_to_dict(node.left),
        'middle': tree_to_dict(node.middle),
        'right': tree_to_dict(node.right)
    }

def save_tree_to_json(root, filename):
    tree_dict = tree_to_dict(root)
    with open(filename, 'w') as f:
        json.dump(tree_dict, f, indent=4)

        
if __name__ == "__main__":
    # 示例使用
    prompt = 'Is cherubism inherited ? '  # 可以使用LLM生成prompt
    topic = '\'Inheritance\''
    model_e = 'qwen-plus'

    root = TreeNode(prompt = prompt, node_id="root")
    build_tree(node = root, depth = 25, model_e = model_e, topic = topic)  # 构建深度为3的三叉树

    save_tree_to_json(root, 'Inheritance.json')

