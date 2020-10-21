from equation_parser_extension import *
# Author: Parul Gupta


if __name__ == "__main__":
    # test 1
    rev_polish_notation = "TP(0.0) TP(1.0) - abs 0.8 TP(1.0) * -"
    r = construct_expr_tree(rev_polish_notation, 0.5)
    # Display infix notation
    print("Infix expression is:")
    inorder(r)




