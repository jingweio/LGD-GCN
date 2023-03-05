## LGD-GCN
[TNNLS 2022] Code for paper ["Learning Disentangled Graph Convolutional Networks Locally and Globally"](https://livrepository.liverpool.ac.uk/3162421/1/Learning_Disentangled_Graph_Convolutional_Networks_Locally_and_Globally.pdf). This paper is based on [a lighter version](https://arxiv.org/abs/2104.11893) and all results and analysis have been reworked substantially.

## Abstract
Graph Convolutional Networks (GCNs) emerge as the most successful learning models for graph-structured data. Despite their success, existing GCNs usually ignore the entangled latent factors typically arising in real-world graphs, which results in non-explainable node representations. Even worse, while the emphasis has been placed on local graph information, the global knowledge of the entire graph is lost to certain extent. In this work, to address these issues, we propose a novel framework for GCNs, termed as LGD-GCN, taking advantage of both local and global information for disentangling node representations in the latent space. Specifically, we propose to represent a disentangled latent continuous space with a statistical mixture model, by leveraging neighborhood routing mechanism *locally*. From the latent space, various new graphs can then be disentangled and learned, to overall reflect the hidden structures with respect to different factors. On one hand, a novel regularizer is designed to encourage *inter-factor diversity* for model expressivity in the latent space. On the other hand, the factor-specific information is encoded *globally* via employing a message passing along these new graphs, so as to strengthen *intra-factor consistency*. Extensive evaluations on both synthetic and five benchmark data sets show that LGD-GCN brings significant performance gains over the recent competitive models in both disentangling and node classification. Particularly, LGD-GCN is able to outperform averagely the disentangled state-of-the-arts by 7.4% on social network data sets.

## Pipeline
<img src="https://github.com/jingweio/LGD-GCN/blob/main/lgd_pipeline.png"/>

## Citation
```
@article{guo2022learning,
  title={Learning Disentangled Graph Convolutional Networks Locally and Globally},
  author={Guo, Jingwei and Huang, Kaizhu and Yi, Xinping and Zhang, Rui},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}
```


## References
	[1] Ma, Jianxin, et al. "Disentangled graph convolutional networks." International conference on machine learning. PMLR, 2019.
