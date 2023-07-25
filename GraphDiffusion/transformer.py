#code based on https://github.com/lucidrains/parti-pytorch and other sources

from typing import List
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
 
import numpy as np
import torch
from tqdm.notebook import trange, tqdm

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# normalization

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 2d relative positional bias

class RelPosBias2d(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        arange = torch.arange(size)

        pos = torch.stack(torch.meshgrid(arange, arange, indexing = 'ij'), dim = -1)
        pos = rearrange(pos, '... c -> (...) c')
        rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

        rel_pos = rel_pos + size - 1
        h_rel, w_rel = rel_pos.unbind(dim = -1)
        pos_indices = h_rel * (2 * size - 1) + w_rel
        self.register_buffer('pos_indices', pos_indices)

    def forward(self, qk):
        i, j = qk.shape[-2:]

        bias = self.pos_bias(self.pos_indices[:i, :(j - 1)])
        bias = rearrange(bias, 'i j h -> h i j')

        bias = F.pad(bias, (j - bias.shape[-1], 0), value = 0.) # account for null key / value for classifier free guidance
        return bias

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden, bias = False),
        nn.GELU(),
        LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False,
        rel_pos_bias = False,
        encoded_fmap_size = None
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, inner_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper

        self.null_kv = nn.Parameter(torch.randn(dim_head))

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM

        self.to_kv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, dim_head, bias = False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

        # positional bias

        self.rel_pos_bias = None

        if rel_pos_bias:
            assert exists(encoded_fmap_size)
            self.rel_pos_bias = RelPosBias2d(encoded_fmap_size, heads)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = default(context, x)
        context = self.norm_context(context)

        kv = self.to_kv(context)

        null_kv = repeat(self.null_kv, 'd -> b 1 d', b = batch)
        kv = torch.cat((null_kv, kv), dim = 1)

        sim = einsum('b h i d, b j d -> b h i j', q, kv)

        if exists(self.rel_pos_bias):
            pos_bias = self.rel_pos_bias(sim)
            sim = sim + pos_bias

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            context_mask = F.pad(context_mask, (1, 0), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        out = einsum('b h i j, b j d -> b h i d', attn, kv)

        return self.to_out(out)
    
    ####################################
    
    #https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        return emb[None, :, :orig_ch].repeat(batch_size, 1, 1)


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = (
            torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class FixEncoding(nn.Module):
    """
    :param pos_encoder: instance of PositionalEncoding1D, PositionalEncoding2D or PositionalEncoding3D
    :param shape: shape of input, excluding batch and embedding size
    Example:
    p_enc_2d = FixEncoding(PositionalEncoding2D(32), (x, y)) # for where x and y are the dimensions of your image
    inputs = torch.randn(64, 128, 128, 32) # where x and y are 128, and 64 is the batch size
    p_enc_2d(inputs)
    """

    def __init__(self, pos_encoder, shape):
        super(FixEncoding, self).__init__()
        self.shape = shape
        self.dim = len(shape)
        self.pos_encoder = pos_encoder
        self.pos_encoding = pos_encoder(
            torch.ones(1, *shape, self.pos_encoder.org_channels)
        )
        self.batch_size = 0

    def forward(self, tensor):
        if self.batch_size != tensor.shape[0]:
            self.repeated_pos_encoding = self.pos_encoding.to(tensor.device).repeat(
                tensor.shape[0], *(self.dim + 1) * [1]
            )
            self.batch_size = tensor.shape[0]
        return self.repeated_pos_encoding    
    
    
    #############################################
    
    # classes
def pad_sequence (output_xyz, max_length):         #pad
    
    device = output_xyz.device
    output=torch.zeros((output_xyz.shape[0],  output_xyz.shape[1] , max_length)).to(device)
    output[:,:,:output_xyz.shape[2]]=output_xyz #just positions for now....
    return output

class GraphWebTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        #position_dim_graph=16,
        
        max_length=1024,
        
        embed_dim_position=64,
         embed_dim_neighbor=6,
        
         neigh_emb_trainable=False,
         max_norm=1.,#embedding ayer mnormed
            
        predict_neighbors=True,#whether or not to predict neighbors..
        
        pos_fourier_graph_dim=128,#fourier pos encoding of entire graph
        
        ### conditioining 
        pos_emb_fourier=True,
        pos_emb_fourier_add=False,
      
        text_embed_dim = 128,
        cond_drop_prob = 0.25,
        max_text_len = 128,
        max_neighbors=5,
        
        use_categorical_for_neighbors = False, #if False, use fixed embeddings and MSE
        
        predict_distance_matrix=False,#if True predict positions and distance matrix
        
    ):
        super().__init__()

        self.predict_distance_matrix=predict_distance_matrix
        
        if predict_distance_matrix:
            predict_neighbors=False #predict just one slab... no embeddigs etc
        self.pos_emb_fourier=pos_emb_fourier
        self.pos_emb_fourier_add=pos_emb_fourier_add
        self.embed_dim_neighbor=embed_dim_neighbor
        self.predict_neighbors=predict_neighbors
        self.pos_fourier_graph_dim=pos_fourier_graph_dim
        self.use_categorical_for_neighbors=use_categorical_for_neighbors
        self.max_neighbors=max_neighbors
        
        
        self.neigh_emb_trainable=neigh_emb_trainable 
        #################################################
        # text conditioning
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last
        
        
        self.GELUact= nn.GELU()
        if self.pos_emb_fourier:
            if self.pos_emb_fourier_add==False:
                text_embed_dim=text_embed_dim+embed_dim_position
            if self.pos_emb_fourier_add:
                print ("Add pos encoding... ", text_embed_dim, embed_dim_position)
                
            self.p_enc_1d = PositionalEncoding1D(embed_dim_position)    
            
        self.max_text_len = max_text_len
        #################################################
        
         
        self.max_length=max_length 
        
        self.max_tokens = max_length+1 #there are as many tokens as there is length in a graph
        if self.predict_neighbors:
            #enc dec for neighbor embeddings
            #same embedding layer for ALL neighbors
            
            
            if not self.neigh_emb_trainable:
                self.neigh_embs=nn.Embedding(self.max_tokens, #this is neighbor types we can have .. i.e. equatl to number of nodes                                       
                                                    #self.embed_dim_neighbor, padding_idx=0, max_norm=max_norm, norm_type =2) 
                                                    self.embed_dim_neighbor,  max_norm=max_norm, norm_type =2) 
                
                #not trainable
                #self.neigh_embs.requires_grad = neigh_emb_trainable 
                self.neigh_embs.weight.requires_grad = neigh_emb_trainable 
            else:
                self.neigh_embs=nn.Embedding(self.max_tokens, #this is neighbor types we can have .. i.e. equatl to number of nodes                                       
                                                    #self.embed_dim_neighbor, padding_idx=0 ) 
                                                    self.embed_dim_neighbor  ) 
            
        
        #######################
        # prediction of graphs
        self.pred_dim=3+self.max_neighbors*self.predict_neighbors*embed_dim_neighbor+self.predict_distance_matrix *self.max_length
        if predict_distance_matrix:
            predict_neighbors=False #predict just one slab... no embeddigs etc
        
        if self.use_categorical_for_neighbors:
            self.logits_dim=  3+self.max_neighbors*self.max_tokens
            self.xyz_and_neigbor_dim = 3+self.max_neighbors
            # if use categorical loss for neighbors then pred_dim is 3+ one hot encoding of neighbors X max_neighbors
        else:
            self.logits_dim=self.pred_dim #if use MSE loss pred_dim is 3+embeddig of beighbors
            self.xyz_and_neigbor_dim = self.pred_dim
            
        print (f"Prediction dimension of output: {self.xyz_and_neigbor_dim}")
            
    
        self.p_enc_1d_graph = PositionalEncodingPermute1D(self.pos_fourier_graph_dim)    
            
        self.start_token = nn.Parameter (torch.randn(self.pred_dim+self.pos_fourier_graph_dim))
        print ("Internal pred dim: ", self.pred_dim, "Four graph enc dim: ", pos_fourier_graph_dim,
              "Logits dim: ", self.logits_dim)

    
        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

        # projecting to logits

        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #Attention(dim, causal = True, encoded_fmap_size = self.image_encoded_dim, rel_pos_bias = True, 
                #          dim_head = dim_head, heads = heads, dropout = dropout),
                Attention(dim, causal = True, rel_pos_bias = False, 
                          dim_head = dim_head, heads = heads, dropout = dropout),
                
                Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
        
        self.to_dim = nn.Linear( self.pred_dim+self.pos_fourier_graph_dim, dim, bias = False)
        #self.to_logits.weight = self.image_token_embed.weight

        # default device

       

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        sequences=None,#conditioning
        *,
        cond_scale = 3.,
        text_mask=None,
        filter_thres = 0.9,
         temperature = 1.,
        tokens_to_generate=None,
        use_argmax=True,#True= use argmax, otherwise gumbel
        start_seq=None, #can provide starting sequence to begin with. Start token will be added automatically
         
    ):
        device = next(self.parameters()).device
      

        if not exists(text_mask):
            text_mask = torch.ones(sequences.shape[:2], dtype = torch.bool).to(device)

        
        
         
        batch = sequences.shape[0]
        if not exists (tokens_to_generate):
            image_seq_len =self.max_length #just set to max length...
        else:
            image_seq_len=tokens_to_generate
        
        print (f"Generating {image_seq_len} tokens...")
       
            

        #for MSE: self.xyz_and_neigbor_dim is just pred_dim 
        #for categorical: 3+max_neigbors
        #print ("self.xyz_and_neigbor_dim", self.xyz_and_neigbor_dim)
        
        if not exists (start_seq):
            output = torch.empty((batch, self.xyz_and_neigbor_dim, 0), device = device )
        else:
            image_seq_len=image_seq_len-start_seq.shape[2]
            
            print ("Starting sequence provided...", start_seq.shape, "Total length to be generated: ",image_seq_len)#, start_seq)
            
            output = start_seq
            assert start_seq.shape[0]==batch and start_seq.shape[1]==self.xyz_and_neigbor_dim, f'Starting sequence has\
                        incorrect dimension: batch {start_seq.shape[0]} vs. {batch}, \
                        embedding depth: {start_seq.shape[1]} vs. {self.xyz_and_neigbor_dim}'
        
        ######################### SAMPLING ##########################################
         # OPTION 2:  Use MSE for xyz and categorical for the neighbors 
        if self.use_categorical_for_neighbors:
            
            
            #ones_top_single=torch.ones (output.shape[0],1,1).to(device)
            
            #print ("output, condx, text mask", output.shape, sequences.shape, text_mask.shape)
            for j in tqdm( range(image_seq_len) ):
                
                #since the original input to the model has 0: ordina, 1,2,3 xyz, and so on
                #we need to add a fake 0 dimension to the input - this is ones_top
                #ones_top=torch.empty (output.shape[0],1,output.shape[2]).to(device)
                #print (output.shape, ones_top.shape)
               # output_= torch.cat((output, ones_top), dim = -2)
                
                #print ("output_ shape ", output_.shape)
                logits  = self.forward_with_cond_scale(
                #sampled = self.forward (
                    sequences = sequences,
                    text_mask = text_mask,
                    output = output,
                    shift_input_depth=0,#usually is one when original training input is used
                )[:, :, -1]
                
                #print ("Predicted logits shape", logits.shape)
                #the first three 0..2 are xyz, the rest the logits
                xyz_logits= logits[:,:3] 
                #print ("xyz logits ", xyz_logits.shape)
                #image_tokens
                for i in range ( self.max_neighbors ):
                    neighbor_logits_i=logits[:,
                                           3+i*self.max_tokens:3+(i+1)*self.max_tokens
                                           ]
                    #neighbor_logits_i=neighbor_logits_i.squeeze (2)
                    
                    if not use_argmax:

                        #rearr_logits=rearrange(logits, 'b n c -> b c n')
                        #print (f"for {i} ", neighbor_logits_i.shape  )

                        #print ("neighbor_logits_i ", neighbor_logits_i.shape)
                        #print ("neighbor_logits_i ", neighbor_logits_i)
                        filtered_logits = top_k(neighbor_logits_i, thres = filter_thres)

                        #print ("filtered_logits ", filtered_logits.shape)
                        #print (filtered_logits)
                        sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
                        #print ("Sampled ", sampled.shape)
                        #print ("Sampled ", sampled)
                        #print ("sampled shape ", sampled.shape)

                    else:
                        sampled=torch.argmax (neighbor_logits_i, -1)
                        #print ("use argmax")
                    
                    
                    #print ("argmax result ", sampled_argmax)
                    sampled = rearrange(sampled, 'b -> b 1 ')
                    
                    #print ("output, sampled (to add new prediction), xyz_logits ", output.shape, sampled.shape, xyz_logits.shape)
                    
                    if i==0:
                        output = torch.cat((xyz_logits, sampled), dim = -1)
                    else:
                        output = torch.cat((output, sampled), dim = -1)
                output=output.unsqueeze(2)   
                
                if j==0:
                    output_f=output
                if j>0:
                    output_f=torch.cat((output_f, output), dim = -1)
                    
                #print ("output after all neighbors added up ", output.shape)
                    
            #print (output.shape, xyz_logits.shape)
           # output = torch.cat((output, xyz_logits), dim = -2)
            
                
            #print ("output ", output.shape)
            return output_f
                
        # OPTION 1:  Use MSE for xyz and embedding 
        if not self.use_categorical_for_neighbors:

            #print ("output, condx, text mask", output.shape, sequences.shape, text_mask.shape)
            for _ in tqdm( range(image_seq_len) ):
                sampled = self.forward_with_cond_scale(
                #sampled = self.forward (
                    sequences = sequences,
                    text_mask = text_mask,
                    output = output,
                    encode_graphs=False,# we are looping with encoded data
                    shift_input_depth=0,
                )
                #sampled=torch.permute(sampled, (0,2,1)  )


                #  print ("sampled shape ", sampled.shape)
                sampled=sampled[:, :, -1]#take LAST prediction....

               # sampled = rearrange(sampled, 'b -> b 1')
                sampled = rearrange(sampled, 'b c -> b c 1')
              #  print ("output", output.shape, "sampled shape ", sampled.shape)
                output = torch.cat((output, sampled), dim = -1)


            if self.predict_neighbors: 
                ind_list=[]
                #for i in range (len (self.neigh_embs)):
                print (f"Now we inverse embeddings, loop over all {self.max_neighbors} neighbors")
                for i  in range (self.max_neighbors):

                    ll=self.embed_dim_neighbor

                    out=output[:,3+i*ll:3+(i+1)*ll ]
                    #print (out.shape)
                    out=torch.permute(out, (0,2,1)  )
                    #print(out.shape)

                    indices=invert_embedding (out, self.neigh_embs )
                    #print (indices)
                    t=torch.Tensor (indices)
                    #print ("tensor added for neighbor ",i, t.shape)
                    if i==0:
                        ind_list=t.unsqueeze (1)
                    else:
                        ind_list=torch.cat((ind_list, t.unsqueeze (1)), 1) 

                    

                #print ("indices shape: ", ind_list.shape, output.shape)

                output=torch.cat((output[:,0:3,:], ind_list.to(device) ), 1) 

            return output         
    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        
      #  print ("forw1")
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)
      #  print ("forw2")
        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        sequences=None,#conditioning
        output=None,
        text_mask = None,
        cond_drop_prob = None,
        
        return_loss = False,
      shift_input_depth=1, #since first deppth is 1,2,3,4,5... 
        encode_graphs=True, #set to False when generationgng 
         
        
        
    ):
        
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        device = next(self.parameters()).device
        ########################## conditioning #################################### 
        
        cond_x=sequences.float().unsqueeze(2)
        
       # print ("cond_x shape ", cond_x.shape)
        cond_x= self.fc1(cond_x)
        
        
        cond_x=self.GELUact(cond_x) 
        
       
        if self.pos_emb_fourier:
            #pos_matrix_i_=self.pos_matrix_i.repeat(x.shape[0], 1, 1).to(device=device) 
            #pos_emb_x = self.pos_emb_x( pos_matrix_i_)
            #pos_emb_x = torch.squeeze(pos_emb_x, 1)
            pos_fourier_xy=self.p_enc_1d(cond_x) 
           # print ("Fourier condition shape (b, length, depth) ", pos_fourier_xy.shape)
            #pos_emb=self.sinu_pos_emb (pos_matrix_i_)
            if self.pos_emb_fourier_add:
                cond_x=x+pos_fourier_xy
                
            #print (f"x shape, pos shape ", x.shape, pos_fourier_xy.shape)
            else:
                cond_x= torch.cat( (cond_x,   pos_fourier_xy), 2)
        ########################## END conditioning ####################################   
        
        
        
        if not self.predict_neighbors: 
            if self.predict_distance_matrix:
                output= output [:,shift_input_depth:3+shift_input_depth+self.max_length,:]
                #print ("otuput ", output.shape)
            elif encode_graphs:
                output =output[:,shift_input_depth:shift_input_depth+3, :]
    
            else:
                output =output[:,0:3, :]
                 
        if self.predict_neighbors: 
            ###########################################################
            ### Encoder graphs

            if encode_graphs:

            
                pos_1=shift_input_depth
                pos_2=shift_input_depth+3
                
               # print (pos_1, pos_2)
                
                output_xyz =output[:,pos_1:pos_2, :]
                output_neighbors =output[:,pos_2:pos_2+self.max_neighbors, :].long()
                #print (output_neighbors)

                #print (output_xyz.shape,output_neighbors.shape )
                output= pad_sequence (output_xyz, self.max_length)          #pad

                for i  in range (self.max_neighbors):
                    #grab next neihhor tensor:
                    x_neigh_l=output_neighbors[:,i,:] 


                    x_neigh_l = torch.unsqueeze(x_neigh_l, dim=-1)

                    if self.neigh_emb_trainable:
                        x_cc =  self.neigh_embs(x_neigh_l)
                    else:
                        with torch.no_grad():
                            x_cc =  self.neigh_embs(x_neigh_l)

                    #.to(device=device)


                    #print ("x_cc shape, output shape: ", x_cc.shape, output.shape)
                    x_cc = torch.squeeze(x_cc, 2)
                    x_cc=torch.permute(x_cc, (0,2,1)  )
                    #print ("x_cc shape, output shape: ", x_cc.shape, output_xyz.shape)


                    if i==0:
                        output= torch.cat( (output_xyz, x_cc  ), 1)

                    else:    
                        output= torch.cat( (output, x_cc ), 1)
                ###########################################################        


        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           self.pos_fourier_graph_dim,
                                                           output.shape[2] ).to(device) ) 
        
        
    
        output=torch.cat( (output, pos_fourier_graph ), 1)
 
        ##################

        output=torch.permute(output, (0,2,1)  )
        
        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = output.shape[0])
        
        
        output = torch.cat((start_tokens, output), dim = 1)
        
    
        
        if return_loss:
            #output = output[:, :-1, :] 
            #print (output.shape)
            output, target = output[:, :-1,:], output[:, 1:,:self.logits_dim]
            #print (output.shape, target.shape)  


         
        # enforce max text len

        if not exists(text_mask):
            text_mask = torch.ones(cond_x.shape[:2], dtype = torch.bool).to(device)
            
        cond_x, text_mask = map(lambda t: t[:, :self.max_text_len], (cond_x, text_mask))

        # classifier free guidance conditional dropout
        batch=output.shape[0]
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        # attend
        #print (output.shape)
        x = self.to_dim(output)
        x = self.init_norm(x)

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)
        
        

        # to logits

        logits = self.to_logits(x)
        logits=torch.permute(logits, (0,2,1)  )

        if not return_loss:
            return logits
        
        #OPTION 2: Use MSE for xyz and categorical for the neighbors 
        if self.use_categorical_for_neighbors:

            #loss_xyz = 
            
            target=torch.permute(target, (0,2,1)  )
            #yields #[b, depth, length]
           
            
            #print ("logits, target", logits.shape, target.shape)
            loss_xyz = F.mse_loss(
                # logits, output
                logits[:,:3,:], target[:,:3,:]
            )
            
            loss_neigh=0
            #now get logits for each of the neighbor positions
            #0,1,2: x,y,z ... 3, 4, ... 3+max_neighbors are neighbors 
            
            #print (logits.shape, target.shape)
            
            for i in range ( self.max_neighbors ):
                neighbor_logits_i=logits[:,
                                       3+i*self.max_tokens:3+(i+1)*self.max_tokens, 
                                       :]
                target_i = output_neighbors[:,i,:].long()
                
                rearr_logits_i=neighbor_logits_i
                #rearr_logits=rearrange(logits, 'b n c -> b c n')
                #print (f"for {i} ", rearr_logits_i.shape,target_i.shape )
                
                #print (f"for {i} ",rearr_logits_i, "target ", target_i)
                loss_i=F.cross_entropy(
                           rearr_logits_i,
                            target_i,
                          #  ignore_index = 0
                        )
                #print (loss_i)
                loss_neigh =loss_neigh+ loss_i
                
            loss=loss_xyz +loss_neigh 
            
            return loss
                
            
        #OPTION 1: Use MSE for everything, and use embedding 
        if not self.use_categorical_for_neighbors:

            #logits=logits[:,:,:-1]
           # target=output[:,:,:-1]
            target=torch.permute(target, (0,2,1)  )
            #print ("logits, target", logits.shape, target.shape) ##[b, depth, length]
            loss = F.mse_loss(
                # logits, output
                logits, target
            )

            return loss