import numpy as np
import pandas as pd


# An expression tree node
class expr_tree:
    # Constructor to create a node 
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# A utility function to check if 'c' is an operator
def isOperator(element):
    if element == '+' or element == '-' or element == '*' or element == '/' or element == '^':
        return True
    return False

def isMod(element):
    if element == "|":
        return True
    return False


def isFunc(element):
    if element.startswith("FP") or element.startswith("FN") or element.startswith("TP") or element.startswith("TN"):
        return True
    return False

# Assumes that Y and predicted_y contain 0,1 binary classification
def eval_estimate(element, Y, predicted_Y, T):
    error = np.subtract(Y, predicted_Y)
    # element will be of the form FP(A) or FN(A) or TP(A) or TN(A)
    type_attribute = element[3:-1]
    type_Y = Y[T.astype(str)==type_attribute]
    type_error = error[T.astype(str)==type_attribute]
    if element.startswith("TP"):
        estimate_array = pd.Series(np.ones_like(Y))[T.astype(str) == type_attribute]
        estimate_array = estimate_array[type_error == 0]
        estimate_array = estimate_array[type_Y == 1]
        return len(estimate_array)
    elif element.startswith("TN"):
        estimate_array = pd.Series(np.ones_like(Y))[T.astype(str) == type_attribute]
        estimate_array = estimate_array[type_error == 0]
        estimate_array = estimate_array[type_Y == 0]
        return len(estimate_array)
    elif element.startswith("FP"):
        return len(type_error[type_error==-1])
    elif element.startswith("FN"):
        return len(type_error[type_error==-1])
    return None


# A utility function to do inorder traversal
def inorder(t_node):
    if t_node is not None:
        inorder(t_node.left)
        print(t_node.value, end=' ')
        inorder(t_node.right)


def eval_expr_tree(t_node, Y, predicted_Y, T):
    if t_node is not None:
        x = eval_expr_tree(t_node.left, Y, predicted_Y, T)
        y = eval_expr_tree(t_node.right, Y, predicted_Y, T)
        if x is None:
            if isFunc(t_node.value):
                return eval_estimate(t_node.value, Y, predicted_Y, T)
            return float(t_node.value)
        elif y is None:
            # only one unary operator supported
            if isMod(t_node.value):
                return np.abs(np.float(x))
            return None
        else:
            if t_node.value == '+':
                return x + y
            elif t_node.value == '-':
                return x - y
            elif t_node.value == '*':
                return x * y
            elif t_node.value == '^':
                return x ** y
            elif t_node.value == '/':
                return x / y
            elif isFunc(t_node.value):
                return eval_estimate(t_node.value, Y, predicted_Y, T)
            elif isMod(t_node.value):
                return abs(float(x))
            return None
    return None


def eval_simple_expr_tree(t_node):
    if t_node is not None:
        x = eval_simple_expr_tree(t_node.left)
        y = eval_simple_expr_tree(t_node.right)
        if x is None:
            return float(t_node.value)
        elif y is None:
            # only one unary operator supported
            if isMod(t_node.value):
                return abs(float(x))
            return None
        else:
            if t_node.value == '+':
                return x + y
            elif t_node.value == '-':
                return x - y
            elif t_node.value == '*':
                return x * y
            elif t_node.value == '^':
                return x ** y
            elif t_node.value == '/':
                return x / y
            return None

# Returns root of constructed tree for
# given postfix expression
def construct_expr_tree(rev_polish_notation):
    rev_polish_notation = rev_polish_notation.split(' ')
    stack = []
    # Traverse through every element of input expression
    for element in rev_polish_notation:
        # if operand, simply push into stack 
        if not isOperator(element) and not isMod(element):
            t = expr_tree(element)
            stack.append(t)
        # Operator/mod
        else:
            # if mod, right node will be None
            if isMod(element):
                t = expr_tree(element)
                t1 = None
                t2 = stack.pop()
            else:
                # Pop the operands
                t = expr_tree(element)
                t1 = stack.pop()
                t2 = stack.pop()
            # make them children
            t.right = t1
            t.left = t2
            # Add this subexpression to stack
            stack.append(t)
    # Only element  will be the root of expression tree
    t = stack.pop()
    return t

