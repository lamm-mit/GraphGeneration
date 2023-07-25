# GraphGeneration: Modeling and design of hierarchical bio-inspired de novo spider web structures using deep learning and additive manufacturing 

*Corresponding author, email: mbuehler@mit.edu  

Spider webs are incredible biological structures, comprising thin but strong silk filament and arranged into complex hierarchical architectures with striking mechanical properties (e.g., lightweight but high strength, achieving diverse mechanical responses).  While simple 2D orb webs can easily be mimicked, the modeling and synthesis of 3D-based web structures remain challenging, partly due to the rich set of design features. Here we provide a detailed analysis of the heterogenous graph structures of spider webs, and use deep learning as a way to model and then synthesize artificial, bio-inspired 3D web structures. The generative models are conditioned based on key geometric parameters (including average edge length, number of nodes, average node degree, and others). To identify graph construction principles, we use inductive representation sampling of large experimentally determined spider web graphs, to yield a dataset that is used to train three conditional generative models: 1) An analog diffusion model inspired by nonequilibrium thermodynamics, with sparse neighbor representation, 2) a discrete diffusion model with full neighbor representation, and 3) an autoregressive transformer architecture with full neighbor representation. All three models are scalable, produce complex, de novo bio-inspired spider web mimics, and successfully construct graphs that meet the design objectives. We further propose algorithm that assembles web samples produced by the generative models into larger-scale structures based on a series of geometric design targets, including helical and parametric shapes, mimicking, and extending natural design principles towards integration with diverging engineering objectives. Several webs are manufactured using 3D printing and tested to assess mechanical properties.  

#### Significance statement

We report a graph-focused deep learning technique to capture the complex design principles of graph architectures – here exemplified for 3D spider webs – and use the model to generate a diverse array of de novo bio-inspired structural designs. The work lays the foundation for spider web generation and explores bio-inspired design using rigorous principles. A set of innovative spider web designs is constructed and manufactured, consisting of varied designs with diverging architectural complexity. In future work, this method can be applied to other heterogeneous hierarchical structures including a broad class of architected materials, and hence offers new opportunities for fundamental biological understanding and to meet a set of diverse design opportunities via the use of generative artificial intelligence for materials applications.

Keywords: Hierarchical materials; Bioinspired; Deep Learning; Generative Diffusion Model; Multiscale Modeling; Materials by Design; Transformer; Generative AI

[1] W. Lu, N.A. Lee, M.J. Buehler, "Modeling and design of hierarchical bio-inspired de novo spider web structures using deep learning and additive manufacturing," PNAS, 120 (31) e2305273120, 2023, 
https://www.pnas.org/doi/10.1073/pnas.2305273120 

![PNASFig1](https://user-images.githubusercontent.com/101393859/229208797-3b47c9e2-a1ae-454c-842e-cff57584eaaf.png)

### Installation 

```
conda create -n GraphGeneration python=3.8
conda activate GraphGeneration
```
```
git clone https://github.com/lamm-mit/GraphGeneration/
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

```
from GraphDiffusion import AnalogDiffusionSparse

max_neighbors = 5
predict_neighbors=True
pred_dim=3+max_neighbors*predict_neighbors
 
context_embedding_max_length=y_data.shape[1]
model =AnalogDiffusionSparse( 
                max_length=max_length,
                pred_dim=pred_dim,
                channels=128,
                unet_type='cfg',  
                context_embedding_max_length=context_embedding_max_length,
                pos_emb_fourier=True,
                pos_emb_fourier_add=False,
                text_embed_dim = 256,
                embed_dim_position=256,
                predict_neighbors=predict_neighbors,
                    )  .to(device)  
```

#### Model 2: Diffusion model with full neighbor matrix representation

The model is constructed similarly to Model 1, except for the representation z. Here, z captures node positions and a full adjacency matrix.

```
from GraphDiffusion import AnalogDiffusionFull 

predict_neighbors=True
pred_dim=3+ max_length

context_embedding_max_length=y_data.shape[1]

model =AnalogDiffusionFull( 
                max_length=max_length,
                pred_dim=pred_dim,
                channels=256,
                unet_type='cfg', #'base', #'cfg',
                context_embedding_max_length=context_embedding_max_length,
                pos_emb_fourier=True,
                pos_emb_fourier_add=False,
                text_embed_dim = 256,
                embed_dim_position=256,
                predict_neighbors=predict_neighbors,
                    )  .to(device)  
```
                    
#### Model 3: Autoregressive transformer architecture with full neighbor matrix representation

The graph representation is identical to Model 2. However, instead of using a diffusion model we use an autoregressive transformer architecture with full adjacency matrix representation. 

```
from GraphDiffusion import GraphWebTransformer

embed_dim_neighbor=32

GWebT = GraphWebTransformer(
        dim=512,
        depth=12,
        dim_head = 64,
        heads = 16,
        dropout = 0.,
        ff_mult = 4,
        max_length=max_length,
        neigh_emb_trainable=False,
        max_norm=1.,
        pos_emb_fourier=True,
        pos_emb_fourier_add=False,
        text_embed_dim = 64,
        embed_dim_position=64,
        embed_dim_neighbor=embed_dim_neighbor,
        predict_neighbors=True,#False,#whether or not to predict neighbors..
        pos_fourier_graph_dim=67,#fourier pos encoding of entire graph
        use_categorical_for_neighbors = False,
        predict_distance_matrix=True,
        cond_drop_prob = 0.25,
        max_text_len = y_data.shape[1],     
).cuda()

sequences= torch.randn(4, 14 ).cuda()
output=torch.randint (0,max_length, (4, 36 , 32)).cuda().float()

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
        tokens_to_generate=max_length, 
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

### Sample results (details, see paper)

Graph construction 

![image](https://github.com/lamm-mit/GraphGeneration/assets/101393859/f2cd9d43-e4e3-42ab-a27a-8b7de6c3f13e)

Sample graph (helical shape)

![image](https://github.com/lamm-mit/GraphGeneration/assets/101393859/81c230c2-a38d-421e-b814-a833a27749c4)

Additive manufacturing of attractor designs

![image](https://github.com/lamm-mit/GraphGeneration/assets/101393859/57cab90f-7368-4b64-8d59-7b88b370e267)

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
