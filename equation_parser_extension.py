import numpy as np
from inequalities import *
from get_bounds import *
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

    def add_delta(self, delta):
        """
        Add delta value to the node
        :return:
        """
        self.delta = delta


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
    if element.startswith("FP") or element.startswith("FN") or\
        element.startswith("TP") or element.startswith("TN"):
        return True
    return False



def construct_expr_tree(rev_polish_notation, delta, check_bound, check_constant):
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
    configure_delta(t, delta, check_bound, check_constant)
    return t


def configure_delta(t_node, delta, check_bound, check_constant):
    if check_constant:
        add_deltas_constant(t_node, delta)
    else:
        add_deltas(t_node, delta)
    if check_bound:
        hash_map = {}
        check_node_dup(t_node, hash_map)
        change_deltas(t_node, hash_map)


def add_deltas_constant(t_node, delta):
    """
    Add delta to the tree - constant incorporation
    :param t_node:
    :param delta:
    """
    if t_node is not None:
        if t_node.left is not None and t_node.left.value is not None:
            if isConstant(t_node.left.value):
                child_delta_left = delta
            elif t_node.right is not None and t_node.right.value is not None:
                if isConstant(t_node.right.value):
                    child_delta_left = delta
                else:
                    child_delta_left = delta/2
            else:
                child_delta_left = delta
            add_deltas_constant(t_node.left, child_delta_left)
        t_node.add_delta(delta)
        if t_node.right is not None and t_node.right.value is not None:
            if isConstant(t_node.right.value):
                child_delta_right = delta
            elif isConstant(t_node.left.value):
                child_delta_right = delta
            else:
                child_delta_right = delta / 2
            add_deltas_constant(t_node.right, child_delta_right)


def add_deltas(t_node, delta):
    """
    Add delta to the tree - no constant incorporation
    :param t_node:
    :param delta:
    """
    if t_node is not None:
        if t_node.left is not None and t_node.left.value is not None:
            if t_node.right is not None and t_node.right.value is not None:
                child_delta_left = delta/2
            else:
                child_delta_left = delta
            add_deltas(t_node.left, child_delta_left)
        t_node.add_delta(delta)
        if t_node.right is not None and t_node.right.value is not None:
            child_delta_right = delta / 2
            add_deltas(t_node.right, child_delta_right)


def check_node_dup(t_node, hash_map):
    """
    Check leaf node duplicates and change delta accordingly
    :param t_node:
    :return:
    """
    if t_node is not None:
        check_node_dup(t_node.left, hash_map)
        if isFunc(t_node.value):
            if t_node.value in hash_map:
                list_of_delta = hash_map[t_node.value]
            else:
                list_of_delta = []
            list_of_delta.append(t_node.delta)
            hash_map[t_node.value] = list_of_delta
        check_node_dup(t_node.right, hash_map)


def isConstant(t_node_value):
    """
    Check for constant numeric term
    :param t_node_value:
    :return:
    """
    try:
        float(t_node_value)
        return True
    except Exception:
        return False


def change_deltas(t_node, hash_map):
    """
    Change the value of delta stored
    :param t_node:
    :param hash_map:
    :return:
    """
    for k, v in hash_map.items():
        if len(v) > 1:
            change_delta_value(t_node, k, sum(v))


def change_delta_value(t_node, element, delta):
    """
    Change the value of delta for the element in the tree
    :param t_node:
    :param element:
    :param delta:
    :return:
    """
    if t_node is not None:
        change_delta_value(t_node.left, element, delta)
        if t_node.value == element:
            t_node.delta = delta
        change_delta_value(t_node.right, element, delta)


#################
# Evaluate tree #
#################
def eval_expr_tree(t_node, Y=None, predicted_Y=None, T=None):
    """
    To evaluate estimate of the expression tree
    :param t_node: expr_tree node
    :param Y: pandas::Series
    :param predicted_Y: tensor
    :param T: pandas::Series
    :return: estimate value: float
    """
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


##########################
# Evaluate conf interval #
##########################
def eval_expr_tree_conf_interval(t_node, Y, predicted_Y, T, inequality,
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
        l_x, u_x = eval_expr_tree_conf_interval(t_node.left, Y, predicted_Y, T,
                                                inequality, candidate_safety_ratio,
                                                predict_bound, modified_h)
        l_y, u_y = eval_expr_tree_conf_interval(t_node.right, Y, predicted_Y, T,
                                                inequality, candidate_safety_ratio,
                                                predict_bound, modified_h)
        if l_x is None and u_x is None:
            if isFunc(t_node.value):
                return eval_func_bound(t_node.value, Y, predicted_Y, T,
                                       t_node.delta, inequality, candidate_safety_ratio,
                                       predict_bound, modified_h)
            # number value
            if t_node.left and t_node.right:
                print(t_node.value, t_node.left.value, t_node.right.value)
            return float(t_node.value), float(t_node.value)
        elif l_y is None and u_y is None:
            # only one unary operator supported
            if isMod(t_node.value):
                bound = eval_math_bound(l_x, u_x, l_y, u_y, 'abs')
                return bound
            return None, None
        else:
            if t_node.value == '+':
                return eval_math_bound(l_x, u_x, l_y, u_y, '+')
            if t_node.value == '-':
                bound = eval_math_bound(l_x, u_x, l_y, u_y, '-')
                return bound
            elif t_node.value == '*':
                return eval_math_bound(l_x, u_x, l_y, u_y, '*')
            elif t_node.value == '^':
                return eval_math_bound(l_x, u_x, l_y, u_y, '^')
            elif t_node.value == '/':
                return eval_math_bound(l_x, u_x, l_y, u_y, '/')
            elif isFunc(t_node.value):
                return eval_func_bound(t_node.value, Y, predicted_Y, T, t_node.delta, inequality, predict_bound, safety_size, modified_h)
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
        print(t_node.value, t_node.delta)
        inorder(t_node.right)



