import matplotlib.pyplot as plt
fig,axs=plt.subplots(nrows=8,ncols=1,figsize=(1*7,8*7))
repr_name="Word2vec"
reduc_name="PCA"
print(f"{repr_name} Embeddings")
print(f"Clustering For {reduc_name}")
try:
    X_reduced=repr_dict[repr_name]["reduc_dict"][reduc_name]["X_reduced"]
    k=0
    for cluster_method in cluster_methods:
        if cluster_method.split()[0] == "CAH":
            method=cluster_method.split()[1]
            cah=CAH(hyperparams=PARAMS[cluster_method.split()[0]])
            cah.create_dendogram(X_reduced,ax=axs[k])
            axs[k].set_title(f"Reduction {reduc_name}: Dendogram ({cluster_method})")
            k+=1
            PARAMS[cluster_method.split()[0]]["method"]=cluster_method.split()[1]
        else:
            labels_model=globals()[cluster_method.split()[0]](hyperparams=PARAMS[cluster_method.split()[0]]) \
                .fit_predict(X_reduced)
            axs[k].scatter(X_reduced[:,0],X_reduced[:,1],c=labels_model)
            axs[k].set_title(f"Reduction {reduc_name} with {cluster_method}")
            k+=1
            repr_dict[repr_name]["reduc_dict"][reduc_name]["cluster_methods"][cluster_method]["labels"]=labels_model
except ValueError as e:
    print(e)