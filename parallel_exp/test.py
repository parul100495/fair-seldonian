from equation_parser_extension import *
# Author: Parul Gupta
from synthetic_data import *
import pandas as pd

if __name__ == "__main__":
    rev_polish_notation = "TP(1) TP(0) - 0.2 -"
    delta = 0.1
    r = construct_expr_tree(rev_polish_notation, delta, False, True)
    # Display infix notation
    print("Infix expression is:")
    inorder(r)
    print("")
    # Evaluate parsed tree
    X, Y, T = get_data(40, 3, 0.5, 0.5, 0.7, 0)
    print(T[T.astype(str) == "1"].shape[0] / T.shape[0])
    # print(pd.concat([X, Y, T], axis = 1))
    male_y = Y[T.astype(str) == "1"]
    print(male_y[male_y.astype(str) == "1.0"].shape[0] / male_y.shape[0])
    female_y = Y[T.astype(str) == "0"]
    print(female_y[female_y.astype(str) == "1.0"].shape[0] / female_y.shape[0])
    predicted_Y = pd.Series([1.0 for i in range(T.shape[0])])
    print(predicted_Y.shape)
    print("\nEvaluation: ", eval_expr_tree(r, Y, predicted_Y, T))

    # # Evaluate conf interval
    l_hoeffding, u_hoeffding = eval_expr_tree_conf_interval(r, Y, predicted_Y, T,
                                                 Inequality.HOEFFDING_INEQUALITY, True, 40, False)
    print ( "Confidence Interval for expression tree:" )
    print ( f" Hoeffding Inequality: [{l_hoeffding}, {u_hoeffding}]" )
    l_hoeffding, u_hoeffding = eval_expr_tree_conf_interval ( r, Y, predicted_Y, T,
                                                              Inequality.T_TEST, True, 40, False )
    print("Confidence Interval for expression tree:")
    print(f" T-test Inequality: [{l_hoeffding}, {u_hoeffding}]")



