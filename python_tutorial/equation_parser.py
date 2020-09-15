import numpy as np
import pandas as pd
# Author: Parul Gupta

# An expression tree node
class expr_tree:
    # Constructor to create a node 
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# A utility function to check if 'element' is an operator
def isOperator(element):
    if element == '+' or element == '-' or element == '*' or element == '/' or element == '^':
        return True
    return False


# A utility function to check if 'element' is mod function
def isMod(element):
    if element == "|":
        return True
    return False


# A utility function to check if 'element' is one of FP, FN, TP, TN
def isFunc(element):
    if element.startswith("FP") or element.startswith("FN") or element.startswith("TP") or element.startswith("TN"):
        return True
    return False


# Assumes that Y and predicted_y contain 0,1 binary classification
# Suppose we are calculating for FP(A).
# Assume X to be an indicator function defined only in case type=A
# s.t. x_i = 1 if FP occurred for ith datapoint and x_i = 0 otherwise.
# Our data samples can be assumed to be independent and identically distributed.
# Our estimate of p, \hat{p} = 1/n * \sum(x_i).
# We can safely count this as binomial random variable.
# E[\hat{p}] = 1/n * np = p
# As we do not know p, we approximate it to \hat{p}.
def eval_estimate(element, Y=None, predicted_Y=None, T=None):
    error = np.subtract(Y, predicted_Y)
    # element will be of the form FP(A) or FN(A) or TP(A) or TN(A)
    type_attribute = element[3:-1]
    type_Y = Y[T.astype(str)==type_attribute]
    type_error = error[T.astype(str)==type_attribute]
    if element.startswith("TP"):
        estimate_array = pd.Series(np.ones_like(Y))[T.astype(str) == type_attribute]
        estimate_array = estimate_array[type_error == 0]
        estimate_array = estimate_array[type_Y == 1]
        # print(len(estimate_array)/len(type_error))
        return len(estimate_array)/len(type_error)
    elif element.startswith("TN"):
        estimate_array = pd.Series(np.ones_like(Y))[T.astype(str) == type_attribute]
        estimate_array = estimate_array[type_error == 0]
        estimate_array = estimate_array[type_Y == 0]
        return len(estimate_array)/len(type_error)
    elif element.startswith("FP"):
        return len(type_error[type_error==-1])/len(type_error)
    elif element.startswith("FN"):
        return len(type_error[type_error==-1])/len(type_error)
    return None


# A utility function to do inorder traversal
def inorder(t_node):
    if t_node is not None:
        inorder(t_node.left)
        print(t_node.value, end=' ')
        inorder(t_node.right)


# To evaluate estimate of the expression tree
def eval_expr_tree(t_node, Y=None, predicted_Y=None, T=None):
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


# Returns root of constructed tree for given postfix expression
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

