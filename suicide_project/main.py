from skimpy import skim
import pandas as pd

def main():
    fin = "../data/DataSet_Combined_SI_SNI_Baseline_FE.csv"
    df= pd.read_csv(fin)
    skim(df)


if __name__ == "__main__":
    main()
