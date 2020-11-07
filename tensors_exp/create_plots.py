import matplotlib.pyplot as plt
from gather_results import gather_results
import numpy as np

csv_path = 'exp/lag_exp/csv/'
img_path = 'exp/lag_exp/images/'


def loadAndPlotResults(fileName, ylabel, output_file, is_yAxis_prob, legend_loc):
    file_ms, file_QSA, file_QSA_stderror, file_LS, file_LS_stderror = np.loadtxt(fileName, delimiter = ',',
                                                                                   unpack = True )

    fig = plt.figure()

    plt.xlim(min(file_ms), max(file_ms))
    plt.xlabel( "Amount of data", fontsize = 16 )
    plt.xscale( 'log' )
    plt.xticks( fontsize = 12 )
    plt.ylabel( ylabel, fontsize = 16 )

    if is_yAxis_prob:
        plt.ylim(-0.1, 1.1)
    # else:
    # plt.ylim(-0.2, 2.2)
    # plt.plot([1, 100000], [1.25, 1.25], ':k');
    # plt.plot([1, 100000], [2.1,  2.1],  ':k');

    plt.plot ( file_ms, file_QSA, 'b-', linewidth = 3, label = 'QSA' )
    plt.errorbar ( file_ms, file_QSA, yerr = file_QSA_stderror, fmt = '.k' );
    plt.plot ( file_ms, file_LS, 'r-', linewidth = 3, label = 'LogRes' )
    plt.errorbar ( file_ms, file_LS, yerr = file_LS_stderror, fmt = '.k' );
    plt.legend ( loc = legend_loc, fontsize = 12 )
    plt.tight_layout ()

    plt.savefig( output_file, bbox_inchesstr = 'tight' )
    plt.show( block = False )


if __name__ == "__main__":
    gather_results()

    loadAndPlotResults(csv_path + 'fs.csv', 'Log Loss', img_path + 'tutorial7MSE_py.png', False, 'lower right')
    loadAndPlotResults(csv_path + 'solutions_found.csv', 'Probability of Solution',
                         img_path + 'tutorial7PrSoln_py.png', True, 'best' )
    loadAndPlotResults(csv_path + 'failures_g1.csv', r'Probability of $g(a(D))>0$',
                         img_path + 'tutorial7PrFail1_py.png', True, 'best' )
    loadAndPlotResults(csv_path+'upper_bound.csv',     r'upper bound', img_path+'tutorial7PrFail2_py.png', False,  'best')
    plt.show()
