from equation_parser import *
# Author: Parul Gupta


if __name__ == "__main__":
    # test 1
    rev_polish_notation = "2 3 - | 5 *"
    r = construct_expr_tree(rev_polish_notation)
    # Display infix notation
    print("Infix expression is:")
    inorder(r)
    # Evaluate parsed tree
    print("\nEvaluation: ", eval_expr_tree(r))

    # test 2
    rev_polish_notation = "TP(0) TP(1) - |"
    r = construct_expr_tree(rev_polish_notation)
    # Display infix notation
    print("Infix expression is:")
    inorder(r)
    # Evaluate parsed tree
    Y = pd.Series(np.array([0, 0, 0, 1, 1, 1]))
    predicted_Y = pd.Series(np.array([1, 1, 1, 1, 1, 1]))
    T = pd.Series(np.array([0, 1, 0, 1, 0, 1]))
    print("\nEvaluation: ", eval_expr_tree(r, Y, predicted_Y, T))
