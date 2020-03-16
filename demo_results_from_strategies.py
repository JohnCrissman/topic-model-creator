# demo_results_from_strategies
import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re

def main():
    df = pd.read_csv('results/accuracy_results_for_all_strategies.csv', encoding = "ISO-8859-1")
    print(df)
    df2 = df.set_index('rows')
    # df1 = df.loc[:,'strategy_1':]
    print(df2)
    plt.figure(figsize=(30,20))
    sns.heatmap(df2, cmap="jet", vmin=0, vmax=0.75)
    plt.title("Accuracy Map - 4 Strategies, 5 Artificial Intelligence Techniques, Latent Dirichlet Topics = 5,10,15,20,25,30")
    plt.xlabel('\n\nStrategy 1:  Each patient is one data point.  Focusing on barriers that are NOT language/interpreter.\n' +
               'Strategy 2:  Each patient is one data point.  Eliminated language/interpreter data points.\n' +
               'Strategy 3:  Each visit from a patient is a data point.  Data points are classified as the barrier given in the visit.\n' +
               'Strategy 4:  Each visit from a patient is a data point.  Eliminated language/interpreter data points.')
    plt.ylabel('Artificial Intelligence Techniques\n\n')
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.subplots_adjust(left=0.26, bottom=0.21, right=1.00,
                            top=0.95, wspace=0.20, hspace=0.20)
    
    plt.show()
    
    






if __name__ == "__main__":
    main()