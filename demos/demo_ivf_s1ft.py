import faiss

quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

centroids = index.quantizer.reconstruct_n(0, index.nlist,)