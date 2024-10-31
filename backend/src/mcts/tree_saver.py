import json

def save_tree(root_node, filename):
    tree_dict = node_to_dict(root_node)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tree_dict, f, ensure_ascii=False, indent=2)

def node_to_dict(node):
    node_dict = {
        'state': node.state,
        'action': node.action,
        'W': node.W,
        'N': node.N,
        'children': [node_to_dict(child) for child in node.children]
    }
    return node_dict
