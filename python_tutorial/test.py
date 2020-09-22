from equation_parser import *
# Author: Parul Gupta


if __name__ == "__main__":
    # test 1
    rev_polish_notation = "2 3 - abs 5 *"
    r = construct_expr_tree(rev_polish_notation)
    # Display infix notation
    print("Infix expression is:")
    inorder(r)
    # Evaluate parsed tree
    print("\nEvaluation: ", eval_expr_tree(r))

    # test 2
    rev_polish_notation = "TP(0) TP(1) - abs"
    r = construct_expr_tree(rev_polish_notation)
    # Display infix notation
    print("Infix expression is:")
    inorder(r)
    # Evaluate parsed tree
    Y = pd.Series(np.array([0, 0, 0, 1, 1, 1]))
    predicted_Y = pd.Series(np.array([1, 1, 1, 1, 1, 1]))
    T = pd.Series(np.array([0, 1, 0, 1, 0, 1]))
    print("\nEvaluation: ", eval_expr_tree(r, Y, predicted_Y, T))
    # Evaluate conf interval for TP(0)
    delta = 0.05
    l_t_test, u_t_test = eval_func_bound("TP(0)", Y, predicted_Y, T, delta,
                                         Inequality.T_TEST)
    l_hoeffding, u_hoeffding = eval_func_bound("TP(0)", Y, predicted_Y, T,
                                               delta, Inequality.HOEFFDING_INEQUALITY)
    print("Confidence Interval for TP(0):")
    print(f" T test: [{l_t_test}, {u_t_test}]")
    print(f" Hoeffding Inequality: [{l_hoeffding}, {u_hoeffding}]")
    # Evaluate conf interval for TP(1)
    l_t_test, u_t_test = eval_func_bound("TP(1)", Y, predicted_Y, T, delta,
                                           Inequality.T_TEST)
    l_hoeffding, u_hoeffding = eval_func_bound("TP(1)", Y, predicted_Y, T,
                                                 delta, Inequality.HOEFFDING_INEQUALITY)
    print("Confidence Interval for TP(1):")
    print(f" T test: [{l_t_test}, {u_t_test}]")
    print(f" Hoeffding Inequality: [{l_hoeffding}, {u_hoeffding}]")
    # Evaluate conf interval
    l_t_test, u_t_test = eval_expr_tree_conf_interval(r, Y, predicted_Y, T, delta,
                                           Inequality.T_TEST)
    l_hoeffding, u_hoeffding = eval_expr_tree_conf_interval(r, Y, predicted_Y, T,
                                                 delta, Inequality.HOEFFDING_INEQUALITY)
    print("Confidence Interval for expression tree:")
    print(f" T test: [{l_t_test}, {u_t_test}]")
    print(f" Hoeffding Inequality: [{l_hoeffding}, {u_hoeffding}]")



