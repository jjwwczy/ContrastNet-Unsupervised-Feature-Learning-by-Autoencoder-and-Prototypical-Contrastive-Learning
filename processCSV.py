import csv
import pandas as pd
data = pd.read_csv(r'Test2.csv')
for dataname in ['SA','PU']:
    FixDataset=data.loc[data.dataset==dataname]
    for temperature in [0.01]:
        FixTem=FixDataset.loc[FixDataset.temperature==temperature]
        for Windowsize in [19,21,23,25,27]:
            FixWindow=FixTem.loc[FixTem.windowsize==Windowsize]
            for Method in ['Contrast','AAE','VAE']:
                FixMethod=FixWindow.loc[FixWindow.feature==Method]

                add_info =FixMethod[FixMethod.columns[0:6]].values[0].tolist()+FixMethod[FixMethod.columns[6:]].values.mean(axis=0).tolist()
                csvFile = open("MeanTest.csv", "a")
                writer = csv.writer(csvFile)
                writer.writerow(add_info)
                csvFile.close()

                print(FixMethod)
