import numpy as np
import pandas as pd
import sys
import math
from enum import Enum
from scipy import stats
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



def construct_expr_tree(rev_polish_notation):
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


####################
# Inequality class #
####################
class Inequality(Enum):
    T_TEST = 1
    HOEFFDING_INEQUALITY = 2


#################
# Evaluate tree #
#################
def eval_expr_tree(t_node, Y=None, predicted_Y=None, T=None):
    """
    To evaluate estimate of the expression tree
    :param t_node: expr_tree node
    :param Y: pandas::Series
    :param predicted_Y: pandas::Series
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


def eval_estimate(element, Y, predicted_Y, T):
    """
    Assumes that Y and predicted_y contain 0,1 binary classification
    Suppose we are calculating for FP(A).
    Assume X to be an indicator function defined only in case type=A
    s.t. x_i = 1 if FP occurred for ith datapoint and x_i = 0 otherwise.
    Our data samples can be assumed to be independent and identically distributed.
    Our estimate of p, \hat{p} = 1/n * \sum(x_i).
    We can safely count this as binomial random variable.
    E[\hat{p}] = 1/n * np = p
    As we do not know p, we approximate it to \hat{p}.

    :param element: expr_tree node
    :param Y: pandas::Series
    :param predicted_Y: pandas::Series
    :param T: pandas::Series
    :return: estimate value: float
    """
    error = np.subtract(Y, predicted_Y)
    # element will be of the form FP(A) or FN(A) or TP(A) or TN(A)
    type_attribute = element[3:-1]
    type_Y = Y[T.astype(str) == type_attribute]
    type_error = error[T.astype(str) == type_attribute]
    if element.startswith("TP"):
        estimate_array = pd.Series(np.ones_like(Y))[T.astype(str) == type_attribute]
        estimate_array = estimate_array[type_error == 0]
        estimate_array = estimate_array[type_Y == 1]
        return len(estimate_array) / len(type_error)
    elif element.startswith("TN"):
        estimate_array = pd.Series(np.ones_like(Y))[T.astype(str) == type_attribute]
        estimate_array = estimate_array[type_error == 0]
        estimate_array = estimate_array[type_Y == 0]
        return len(estimate_array) / len(type_error)
    elif element.startswith("FP"):
        return len(type_error[type_error == -1]) / len(type_error)
    elif element.startswith("FN"):
        return len(type_error[type_error == -1]) / len(type_error)
    return None


##########################
# Evaluate conf interval #
##########################
def eval_expr_tree_conf_interval(t_node, Y, predicted_Y, T, delta, inequality, predict_bound, safety_size):
    """
    To evaluate confidence interval of the expression tree
    :param t_node: expr_tree node
    :param Y: pandas::Series
    :param predicted_Y: pandas::Series
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
        l_x, u_x = eval_expr_tree_conf_interval(t_node.left, Y, predicted_Y, T, child_delta, inequality, predict_bound, safety_size)
        l_y, u_y = eval_expr_tree_conf_interval(t_node.right, Y, predicted_Y, T, child_delta, inequality, predict_bound, safety_size)
        if l_x is None and u_x is None:
            if isFunc(t_node.value):
                return eval_func_bound(t_node.value, Y, predicted_Y, T, delta, inequality, predict_bound, safety_size)
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
                return eval_func_bound(t_node.value, Y, predicted_Y, T, delta, inequality, predict_bound, safety_size)
            elif isMod(t_node.value):
                return eval_math_bound(l_x, u_x, l_y, u_y, 'abs')
            return None, None
    return None, None


def eval_math_bound(l_x, u_x, l_y=None, u_y=None, operator=None):
    if operator == '+':
        return eval_add_bound(l_x, u_x, l_y, u_y)
    elif operator == '-':
        return eval_subtract_bound(l_x, u_x, l_y, u_y)
    elif operator == '*':
        return eval_multiply_bound(l_x, u_x, l_y, u_y)
    elif operator == '^':
        # power is not supported as of now
        return l_x, u_x
    elif operator == '/':
        return eval_div_bound(l_x, u_x, l_y, u_y)
    elif isMod(operator):
        return eval_abs_bound(l_x, u_x)
    return None, None


def eval_abs_bound(l_x, u_x):
    """
    :param l_x: lower bound
    :param u_x: upper bound
    :return: lower and upper bound of abs operation
    """
    if l_x is not None and u_x is not None:
        if l_x == math.inf or u_x == math.inf or l_x == -math.inf or u_x == math.inf:
            return 0, math.inf
        elif l_x <= 0 and u_x <= 0:
            return -u_x, -l_x
        elif l_x >= 0 and u_x >= 0:
            return l_x, u_x
        elif l_x <= 0 <= u_x:
            return 0, max(-l_x, u_x)
    return None, None


def eval_div_bound(l_x, u_x, l_y, u_y):
    """
    :param l_x: lower bound of left child
    :param u_x: upper bound of left child
    :param l_y: lower bound of right child
    :param u_y: upper bound of right child
    :return: lower and upper bound of div operation
    """
    if l_x is not None and u_x is not None and l_y is not None and u_y is not None:
        if (l_x == -math.inf and u_x == math.inf) or l_y <= 0 <= u_y:
            # x is unbounded or 0 in y
            return -math.inf, math.inf
        elif l_x >= 0 and l_y >= 0:
            # both x, y are positive
            if u_y == math.inf:
                lower = 0
            else:
                lower = l_x / u_y
            if u_x == math.inf:
                upper = math.inf
            else:
                upper = u_x / l_y
            return lower, upper
        elif u_x <= 0 and u_y <= 0:
            # both x, y are negative
            if l_y == -math.inf:
                lower = 0
            else:
                lower = u_x / l_y

            if l_x == -math.inf:
                upper = math.inf
            else:
                upper = l_x / u_y
            return lower, upper
        elif l_x >= 0 >= u_y:
            # x is positive and y is negative
            if u_x == math.inf:
                lower = -math.inf
            else:
                lower = l_x / u_y
            if l_y == -math.inf:
                upper = 0
            else:
                upper = l_x / l_y
            return lower, upper
        elif u_x <= 0 <= l_y:
            # x is negative and y is positive
            if l_x == -math.inf:
                lower = -math.inf
            else:
                lower = l_x / l_y
            if u_y == math.inf:
                upper = 0
            else:
                upper = u_x / u_y
            return lower, upper
        elif l_x <= 0 <= u_x and l_y >= 0:
            # 0 in x and y is positive
            if l_x == -math.inf:
                lower = -math.inf
            else:
                lower = l_x / l_y
            if u_x == math.inf:
                upper = math.inf
            else:
                upper = u_x / l_y
            return lower, upper
        elif l_x <= 0 <= u_x and u_y <= 0:
            # 0 in x and y is negative
            if u_x == math.inf:
                lower = -math.inf
            else:
                lower = u_x / u_y
            if l_x == -math.inf:
                upper = math.inf
            else:
                upper = l_x / u_y
            return lower, upper
    return None, None


def eval_multiply_bound(l_x, u_x, l_y, u_y):
    """
    :param l_x: lower bound of left child
    :param u_x: upper bound of left child
    :param l_y: lower bound of right child
    :param u_y: upper bound of right child
    :return: lower and upper bound of multiply operation
    """
    if l_x is not None and u_x is not None and l_y is not None and u_y is not None:
        if (l_x == -math.inf and u_x == math.inf) or (l_y == -math.inf and u_y == math.inf):
            # one of x, y is unbounded
            return -math.inf, math.inf
        elif l_x >= 0 and l_y >= 0:
            # both x, y are positive
            if u_x == math.inf or u_y == math.inf:
                return l_x * l_y, math.inf
            return l_x * l_y, u_x * u_y
        elif u_x <= 0 and u_y <= 0:
            # both x, y are negative
            if l_x == -math.inf or l_y == -math.inf:
                return u_x * u_y, math.inf
            return u_x * u_y, l_x * l_y
        elif l_x >= 0 >= u_y:
            # x is positive and y is negative
            if u_x == math.inf or l_y == -math.inf:
                return -math.inf, l_x * u_y
            return u_x * l_y, l_x * u_y
        elif u_x <= 0 <= l_y:
            # x is negative and y is positive
            if l_x == -math.inf or u_y == math.inf:
                return -math.inf, u_x * l_y
            return l_x * u_y, u_x * l_y
        elif l_x <= 0 <= u_x and l_y >= 0:
            # 0 in x and y is positive
            if l_x == -math.inf or u_y == math.inf:
                lower = -math.inf
            else:
                lower = l_x * u_y
            if u_x == math.inf or u_y == math.inf:
                upper = math.inf
            else:
                upper = u_x * u_y
            return lower, upper
        elif l_x <= 0 <= u_x and u_y <= 0:
            # 0 in x and y is negative
            if u_x == math.inf or l_y == -math.inf:
                lower = -math.inf
            else:
                lower = u_x * l_y
            if l_x == -math.inf or l_y == -math.inf:
                upper = math.inf
            else:
                upper = l_x * l_y
            return lower, upper
        elif  l_x >= 0 and l_y <= 0 <= u_y:
            # 0 in y and x is positive
            if u_x == math.inf or l_y == -math.inf:
                lower = -math.inf
            else:
                lower = u_x * l_y
            if u_x == math.inf or u_y == math.inf:
                upper = math.inf
            else:
                upper = u_x * u_y
            return lower, upper
        elif u_x <= 0 and l_y <= 0 <= u_y:
            # 0 in y and x is negative
            if l_x == -math.inf or u_y == math.inf:
                lower = -math.inf
            else:
                lower = l_x * u_y
            if l_x == -math.inf or l_y == -math.inf:
                upper = math.inf
            else:
                upper = l_x * l_y
            return lower, upper
        elif l_x <= 0 <= u_x and l_y <= 0 <= u_y:
            # 0 in x and 0 in y
            if l_x == -math.inf or l_y == -math.inf or u_x == math.inf or u_y == math.inf:
                # unbounded
                return -math.inf, math.inf
            else:
                return min(l_x * u_y, u_x * l_y), max(u_x * u_y, l_x * l_y)
    return None, None


def eval_subtract_bound(l_x, u_x, l_y, u_y):
    """
    :param l_x: lower bound of left child
    :param u_x: upper bound of left child
    :param l_y: lower bound of right child
    :param u_y: upper bound of right child
    :return: lower and upper bound of subtract operation
    """
    if l_x is not None and u_x is not None and l_y is not None and u_y is not None:
        # lower bound
        if l_x == -math.inf or u_y == math.inf:
            lower = -math.inf
        else:
            lower = l_x - u_y

        # upper bound
        if u_x == math.inf or l_y == -math.inf:
            upper = math.inf
        else:
            upper = u_x - l_y
        return lower, upper
    return None, None


def eval_add_bound(l_x, u_x, l_y, u_y):
    """
    :param l_x: lower bound of left child
    :param u_x: upper bound of left child
    :param l_y: lower bound of right child
    :param u_y: upper bound of right child
    :return: lower and upper bound of add operation
    """
    if l_x is not None and u_x is not None and l_y is not None and u_y is not None:
        # lower bound
        if l_x == -math.inf or l_y == -math.inf:
            lower = -math.inf
        else:
            lower = l_x + l_y

        # upper bound
        if u_x == math.inf or u_y == math.inf:
            upper = math.inf
        else:
            upper = u_x + u_y
        return lower, upper
    return None, None


def eval_func_bound(element, Y, predicted_Y, T, delta, inequality, predict_bound, safety_size):
    estimate = eval_estimate(element, Y, predicted_Y, T)
    num_of_elements = len(Y)
    if inequality == Inequality.T_TEST:
        std_dev = get_std_dev(element, estimate, predicted_Y, T, num_of_elements)
        if predict_bound:
            return predict_t_test(estimate, std_dev, safety_size, delta)
        return eval_t_test(estimate, std_dev, num_of_elements, delta)
    elif inequality == Inequality.HOEFFDING_INEQUALITY:
        if predict_bound:
            return predict_hoeffding(estimate, safety_size, delta)
            #return predict_hoeffding_modified(estimate, safety_size, num_of_elements, delta)
        return eval_hoeffding(estimate, num_of_elements, delta)
    return None, None


def get_std_dev(element, estimate, predicted_Y, T, num_of_elements):
    # element will be of the form FP(A) or FN(A) or TP(A) or TN(A)
    type_attribute = element[3:-1]
    type_Y = predicted_Y[T.astype(str) == type_attribute]
    sum_term = (type_Y - estimate)**2
    return math.sqrt(np.sum(sum_term) / (num_of_elements - 1))


def eval_t_test(estimate, std_dev, num_of_elements, delta):
    t = stats.t.ppf(1 - delta, num_of_elements - 1)
    int_size = (std_dev / math.sqrt(num_of_elements)) * t
    return estimate - int_size, estimate + int_size

def predict_t_test(estimate, std_dev, safety_size, delta):
    t = stats.t.ppf(1 - delta, safety_size - 1)
    int_size = 2 * (std_dev / math.sqrt(safety_size)) * t
    return estimate - int_size, estimate + int_size

def eval_hoeffding(estimate, num_of_elements, delta):
    int_size = math.sqrt(math.log(1/delta) / (2 * num_of_elements))
    return estimate - int_size, estimate + int_size

def predict_hoeffding(estimate, safety_size, delta):
    constant_term = math.sqrt(math.log(1/delta) / (2 * safety_size))
    int_size = 2 * constant_term
    return estimate - int_size, estimate + int_size

def predict_hoeffding_modified(estimate, num_of_elements, safety_size, delta):
    constant_term1 = math.sqrt(math.log(1/delta) / (2 * num_of_elements))
    constant_term2 = math.sqrt(math.log(1/delta) / (2 * safety_size))
    int_size = constant_term1 + constant_term2
    return estimate - int_size, estimate + int_size

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



