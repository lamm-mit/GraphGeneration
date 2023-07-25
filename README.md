# GraphGeneration: Modeling and design of hierarchical bio-inspired de novo spider web structures using deep learning and additive manufacturing 

Spider webs are incredible biological structures, comprising thin but strong silk filament and arranged into highly complex hierarchical architectures with striking mechanical properties (e.g., lightweight but high strength).  While simple 2D orb webs can easily be mimicked, the modeling and synthesis of artificial, bio-inspired 3D-based web structures is challenging, partly due to the rich set of design features. Here we use deep learning as a way to model and synthesize such 3D web structures, where generative models are conditioned based on key geometric parameters (incl.: average edge length, number of nodes, average node degree, and others). To identify construction principles, we use inductive representation sampling of large spider web graphs and develop and train three distinct conditional generative models to accomplish this task: 1) An analog diffusion model with sparse neighbor representation, 2) a discrete diffusion model with full neighbor representation, and 3) an autoregressive transformer architecture with full neighbor representation. We find that all three models can produce complex, de novo bio-inspired spider web mimics and successfully construct samples that meet the design conditioning that reflect key geometric features (including, the number of nodes,   spatial orientation, and edge lengths). We further present an algorithm that assembles inductive samples produced by the generative deep learning models into larger-scale structures based on a series of geometric design targets, including helical forms and parametric curves. 

[1] W. Lu, N.A. Lee, M.J. Buehler, "Modeling and design of hierarchical bio-inspired de novo spider web structures using deep learning and additive manufacturing," PNAS, 120 (31) e2305273120, 2023, 
https://www.pnas.org/doi/10.1073/pnas.2305273120 

![PNASFig1](https://user-images.githubusercontent.com/101393859/229208797-3b47c9e2-a1ae-454c-842e-cff57584eaaf.png)

### Installation 

```
conda create -n GraphGeneration python=3.8
conda activate GraphGeneration
```
```
git clone https://github.com/lamm-mit/HierarchicalDesign/
cd GraphGeneration
```

Then, install GraphGeneration:

```
pip install -e .
```

Start Jupyter Lab (or Jupyter Notebook):
```
jupyter-lab --no-browser
```

### Model overview

#### Model 1: Analog diffusion model with sparse neighbor representation

In this approach, we describe graph representations z as defined in Figure S2 in the paper. Since we use a sparse representation, the input structure consists of a list of embeddings of length N, padded with zeros for graph samples of length less than N. 

#### Model 2: Diffusion model with full neighbor matrix representation

The model is constructed similarly to Model 1, except for the representation z. Here, z captures node positions and a full adjacency matrix.

#### Model 3: Autoregressive transformer architecture with full neighbor matrix representation

The graph representation is identical to Model 2. However, instead of using a diffusion model we use an autoregressive transformer architecture with full adjacency matrix representation. 

```
max_length2=32
embed_dim_neighbor=3
max_neighbors2=5
GWebT = GraphWebTransformer(
        dim=128,
        depth=3,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        predict_neighbors=True,#True, #False,#only xyz
        max_length=max_length2,
        neigh_emb_trainable=True,
        max_norm=1.,#embedding ayer mnormed
        pos_emb_fourier=True,
        pos_emb_fourier_add=False,
        text_embed_dim = 256,
        embed_dim_position=256,
        embed_dim_neighbor=embed_dim_neighbor,
        use_categorical_for_neighbors = False,
        predict_distance_matrix=True,
        cond_drop_prob = 0.25,
        max_text_len = 14
    ).cuda()

sequences= torch.randn(4, 14 ).cuda()
output=torch.randint (0,max_length2, (4, 36 , 32)).cuda().float()

loss=GWebT(
        sequences=sequences,#conditioning
        output=output,
        text_mask = None,
        return_loss = True,
        encode_graphs=True,
       )
loss.backward()
print ("Loss: ", loss)

result = GWebT.generate(sequences=sequences,
        tokens_to_generate=max_length2, 
        cond_scale = 1., temperature=3, use_argmax=True,
     ) 
print (result.shape) #(b, [x,y,z, N1, N2, N3, .. N_max_neighbor, max_length])
```

### Model weights and data

- [Dataset](https://www.dropbox.com/s/38jwpqtz6c8rcey/dataset_webs_medium.pt?dl=0) 
- [Model 1 weights](https://www.dropbox.com/s/a0i1h32jf4nmnaf/statedict_save-model-epoch_4327.pt?dl=0)
- [Model 2 weights](https://www.dropbox.com/s/og7wzqysxhkff5o/statedict_save-model-epoch_2001.pt?dl=0)
- [Model 3 weights](https://www.dropbox.com/s/ul85wjtqul6wid0/statedict_save-model-epoch_772.pt?dl=0)


### Schematic of the models implemented

![PNASFig2](https://user-images.githubusercontent.com/101393859/229208831-88c2df9f-e0b8-49cf-b900-0d152ff37759.png)

```
@article{WeiLeeBuehler_2023,
    title   = {Modeling and design of hierarchical bio-inspired de novo spider web structures using deep learning and additive manufacturing},
    author  = {W. Lu, N.A. Lee, M.J. Buehler},
    journal = {PNAS},
    year    = {2023},
    volume  = {120},
    pages   = {e2305273120},
    url     = {https://www.pnas.org/doi/10.1073/pnas.2305273120}
}
```
