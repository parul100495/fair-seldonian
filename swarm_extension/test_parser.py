from equation_parser import *
# Author: Parul Gupta


if __name__ == "__main__":
    # test 1
    rev_polish_notation = "TP(0.0) TP(1.0) - abs 0.8 TP(1.0) * -"
    r = construct_expr_tree_base(rev_polish_notation)
    # Display infix notation
    print("Infix expression is:")
    inorder(r)

    Y = pd.Series ( np.array ( [ 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 ] ) )
    predicted_Y = pd.Series ( np.array ( [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ] ) )
    T = pd.Series ( np.array ( [ 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 ] ) )

    # Evaluate conf interval for TP(1)
    l_t_test, u_t_test = eval_expr_tree_conf_interval_base(r, Y, predicted_Y, T, 0.05, Inequality.HOEFFDING_INEQUALITY, False, 5, False)
    print(l_t_test, u_t_test)



