class Node:
    def __init__(self, name,value=None):
        self.name = name   
        self.children = []
        self.value = value  # Only leaf nodes strictly need values initially

def minimax_tree(node,depth,max_d, is_maximizing):
    # Base Case: If the node is a leaf, return its value
    if not node.children or depth== max_d :
        return node.value

    if is_maximizing:
        best_val = float('-inf')
        for child in node.children:
            value = minimax_tree(child,depth+1,max_d ,False)
            best_val = max(best_val, value)
        node.value = best_val  # Store the best value for the maximizing node
        return best_val
    else:
        best_val = float('inf')
        for child in node.children:
            value = minimax_tree(child,depth+1,max_d, True)
            best_val = min(best_val, value)
        node.value = best_val  # Store the best value for the minimizing node
        return best_val

root = Node('A')
n1, n2 = Node('B'), Node('C')
root.children = [n1, n2]

n3, n4, n5, n6 = Node('D'), Node('E'), Node('F'), Node('G')
n1.children = [n3, n4]
n2.children = [n5, n6]

# Leaves (Name, then Numeric Value)
n3.children = [Node('n7', 3), Node('n8', 5)]
n4.children = [Node('n9', 2), Node('n10', 9)]
n5.children = [Node('n11', 0), Node('n12', 1)]
n6.children = [Node('n13', 4), Node('n14', 6)]
# Calculation
result = minimax_tree(root,0,3,True)
print(f"The optimal value for the root is: {result}")
print("Values for each node:")
print(f"Node A: {root.value}")
print(f"Node B: {n1.value}")
print(f"Node C: {n2.value}")
print(f"Node D: {n3.value}")
print(f"Node E: {n4.value}")
print(f"Node F: {n5.value}")
print(f"Node G: {n6.value}")
