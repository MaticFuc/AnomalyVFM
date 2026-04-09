import pandas as pd
import os

def log_results(results, out_fold, file="results.csv", new_file=False):
    new_file = not os.path.exists(f"{out_fold}/{file}")
    mode = "w" if new_file else "a"
    header = True if new_file else False
    index = False

    df = pd.DataFrame(results)
    df.to_csv(f"{out_fold}/{file}", mode=mode, header=header, index=index)