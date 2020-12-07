from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
Datadir='./DataArray/'
Name=['ContrastFeature','AAE_Features','VAE_Features']

for i in range(len(Name)):
    Feature=np.load(Datadir+Name[i]+'.npy')
    y=np.load(Datadir + 'y.npy')

    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(Feature,y)


    plt.figure(figsize=(10, 10))
    # plt.subplot(121)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=y,label=Name[i], cmap='rainbow')
    plt.legend()
    plt.savefig('SA_'+Name[i]+'.png')
    plt.show()