import pandas as pd

def change(x):
	return " ".join(map(str, x.translate(None, '[]').split(',')))

subm = pd.read_csv("./output/to_kaggle/subm_fnn.csv")
subm['ad_id'] = subm.ad_id.apply(change)
subm.to_csv("./output/to_kaggle/subm_fnn_1.csv", index=False)