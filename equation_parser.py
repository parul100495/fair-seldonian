import numpy as np
from get_bounds import *
from inequalities import *
# Author: Parul Gupta

####################
# Construct Parser #
####################
class expr_tree:
    """
    An expression tree node
    """
    def __init__(self, value):
        """
        Constructor to create a node
        :param value:
        """
        self.value = value
        self.left = None
        self.right = None


def isOperator(element):
    """
    # A utility function to check if 'element' is an operator
    :param element: expr_tree node
    :return: bool
    """
    if element == '+' or element == '-' or element == '*' or element == '/' or element == '^':
        return True
    return False


def isMod(element):
    """
    A utility function to check if 'element' is mod function
    :param element: expr_tree node
    :return: bool
    """
    if element == "abs":
        return True
    return False


def isFunc(element):
    """
    A utility function to check if 'element' is one of FP, FN, TP, TN
    :param element: expr_tree node
    :return: bool
    """
    if element.startswith("FP") or element.startswith("FN") or element.startswith("TP") or element.startswith("TN"):
        return True
    return False


def construct_expr_tree_base(rev_polish_notation):
    """
    Returns root of constructed tree for given postfix expression

    :param rev_polish_notation: string with space as delimiter ' '
    :return: expr_tree node
    """
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


#################
# Evaluate tree #
#################
def eval_expr_tree_base(t_node, Y, predicted_Y, T):
    """
    To evaluate estimate of the expression tree
    :param t_node: expr_tree node
    :param Y: pandas::Series
    :param predicted_Y: tensor
    :param T: pandas::Series
    :return: estimate value: float
    """
    if t_node is not None:
        x = eval_expr_tree_base(t_node.left, Y, predicted_Y, T)
        y = eval_expr_tree_base(t_node.right, Y, predicted_Y, T)
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


##########################
# Evaluate conf interval #
##########################
def eval_expr_tree_conf_interval_base(t_node, Y, predicted_Y, T, delta, inequality,
                                      candidate_safety_ratio, predict_bound, modified_h):
    """
    To evaluate confidence interval of the expression tree
    :param t_node: expr_tree node
    :param Y: pandas::Series
    :param predicted_Y: tensor
    :param T: pandas::Series
    :param conf_prob: float in [0, 1]
    :return:
    """
    if t_node is not None:
        if t_node.right is not None and t_node.right.value is not None:
            # propagate conf bound for binary operator
            child_delta = delta/2
        else:
            child_delta = delta
        l_x, u_x = eval_expr_tree_conf_interval_base(t_node.left, Y, predicted_Y, T, child_delta,
                                                     inequality, candidate_safety_ratio, predict_bound, modified_h)
        l_y, u_y = eval_expr_tree_conf_interval_base(t_node.right, Y, predicted_Y, T, child_delta,
                                                     inequality, candidate_safety_ratio, predict_bound, modified_h)
        if l_x is None and u_x is None:
            if isFunc(t_node.value):
                return eval_func_bound(t_node.value, Y, predicted_Y, T, delta,
                                       inequality, candidate_safety_ratio, predict_bound, modified_h)
            # number value
            return float(t_node.value), float(t_node.value)
        elif l_y is None and u_y is None:
            # only one unary operator supported
            if isMod(t_node.value):
                return eval_math_bound(l_x, u_x, l_y, u_y, 'abs')
            return None, None
        else:
            if t_node.value == '+':
                return eval_math_bound(l_x, u_x, l_y, u_y, '+')
            elif t_node.value == '-':
                return eval_math_bound(l_x, u_x, l_y, u_y, '-')
            elif t_node.value == '*':
                return eval_math_bound(l_x, u_x, l_y, u_y, '*')
            elif t_node.value == '^':
                return eval_math_bound(l_x, u_x, l_y, u_y, '^')
            elif t_node.value == '/':
                return eval_math_bound(l_x, u_x, l_y, u_y, '/')
            elif isFunc(t_node.value):
                return eval_func_bound(t_node.value, Y, predicted_Y, T, delta,
                                       inequality, candidate_safety_ratio, predict_bound, modified_h)
            elif isMod(t_node.value):
                return eval_math_bound(l_x, u_x, l_y, u_y, 'abs')
            return None, None
    return None, None


##############
# Print Tree #
##############
def inorder(t_node):
    """
    A utility function to do inorder traversal
    :param t_node: expr_tree node
    :return: None
    """
    if t_node is not None:
        inorder(t_node.left)
        print(t_node.value, end=' ')
        inorder(t_node.right)



