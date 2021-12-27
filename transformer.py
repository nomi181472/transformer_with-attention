import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, embed_size,heads): # embed size 256 then head will be 8
        super(SelfAttention,self).__init__() #initiaze parent class
        self.embed_size=embed_size
        self.heads=heads
        self.head_dim=embed_size//heads
        assert (self.heads* self.head_dim==embed_size), "self.heads* self.head_dim==embed_size"
        self.values=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.keys=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out=nn.Linear(self.heads*self.head_dim,embed_size)
    def forward(self,values,keys,queries,mask):
        N=queries.shape[0]
        value_len,key_len,query_len=values.shape[1],keys.shape[1],queries.shape[1]
        #splitting embeddings into self.heads pieces
        values=values.reshape(N,value_len,self.heads,self.head_dim)
        keys=keys.reshape(N,key_len,self.heads,self.head_dim)
        queries=queries.reshape(N,query_len,self.heads,self.head_dim)
        values=self.values(values)
        keys=self.keys(keys)
        queries=self.queries(queries)
        enery=torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        if mask is not None:
            mask=enery.masked_fill(mask==0,float("-1e20"))
        attension=torch.softmax(enery/(self.embed_size**(1/2)),dim=3)
        out=torch.einsum("nhql,nlhd->nqhd",[attension,values]).reshape(
            N,query_len,self.heads*self.head_dim
        )
        out=self.fc_out(out)
        return out
        #attension shape (n,head,query_len,key_len)
        #value shape (N,value_len,Head,head_dim)
        #(n,query_len,heads,head_dim)
class TransformerBlock(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansions):
        super(TransformerBlock,self).__init__()
        self.attension=SelfAttention(embed_size=embed_size,heads=heads)
        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)
        self.feed_forward=nn.Sequential(
            nn.Linear(embed_size,forward_expansions*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansions*embed_size,embed_size)

        )
        self.dropout=nn.Dropout(dropout)
    def forward(self,value,key,query,mask):
        attention=self.attension(value,key,query,mask)
        x=self.dropout(self.norm1(attention+query)) # skip connect
        forward=self.feed_forward(x)
        out=self.dropout(self.norm2(forward+x))
        return out
class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansions,
                 dropout,
                 max_length):
        super(Encoder,self).__init__()
        self.embed_size=embed_size
        self.device=device
        self.word_embedding=nn.Embedding(src_vocab_size,embed_size)
        self.position_embedding=nn.Embedding(max_length,embed_size)
        self.layers=nn.ModuleList([
            TransformerBlock(
                embed_size=embed_size,
                heads=heads,
                dropout=dropout,
                forward_expansions=forward_expansions

            )
            for _ in range(num_layers)

        ])
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,mask):
        N,seq_length=x.shape
        positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        out=self.dropout(self.word_embedding(x)*self.position_embedding(positions))
        for layer in self.layers:
            out =layer(out,out,out,mask)
        return out
class DecorderBlock(nn.Module):
    def __init__(self,embed_size,heads,forward_expansions,dropout,device):
        super(DecorderBlock, self).__init__()
        self.attention=SelfAttention(embed_size=embed_size,heads=heads)
        self.norm=nn.LayerNorm(embed_size)
        self.transformer_block=TransformerBlock(embed_size=embed_size,heads=heads,dropout=dropout,forward_expansions=forward_expansions)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,value,key,src_mask,trg_mask):
        attension=self.attention(x,x,x,trg_mask)
        query=self.dropout(self.norm(attension+x))
        out=self.transformer_block(value,key,query,src_mask)
        return out
class Decorder(nn.Module):
    def __init__(self,trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansions,
                 dropout,
                 device,
                 max_length):
        super(Decorder, self).__init__()
        self.device=device
        self.word_embedding=nn.Embedding(trg_vocab_size,embed_size)
        self.position_embedding=nn.Embedding(max_length,embed_size)
        self.layers=nn.ModuleList(
            [DecorderBlock(embed_size=embed_size,heads=heads,forward_expansions=forward_expansions,dropout=dropout,device=device) for _ in range(num_layers)]
        )
        self.fc_out=nn.Linear(embed_size,trg_vocab_size)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,enc_out,src_mask,trg_mask):
        N,seq_length =x.shape

        positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        x=self.dropout((self.word_embedding(x)+self.position_embedding(positions)))
        for layer in self.layers:
            x=layer(x,enc_out,enc_out,src_mask,trg_mask)
        out =self.fc_out(x)
        return out
class Transformer (nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 num_layers=6,
                 forward_expansions=4,
                 heads=8,
                 dropout=0.1,
                 device="cpu",
                 max_length=100):
        super(Transformer, self).__init__()
        self.encoder=Encoder(src_vocab_size=src_vocab_size,embed_size=embed_size,num_layers=num_layers,heads=heads,device=device,forward_expansions=forward_expansions,dropout=dropout,
                             max_length=max_length)
        self.decorder=Decorder(trg_vocab_size=trg_vocab_size,embed_size=embed_size,num_layers=num_layers,heads=heads,
                               forward_expansions=forward_expansions,dropout=dropout,device=device,max_length=max_length)
        self.src_pad_idx=src_pad_idx
        self.trg_pd_idx=trg_pad_idx
        self.device=device
    def make_src_mask(self,src):
        src_mask=  (src!= self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    def make_trg_mask(self,trg):
        N,trg_len=trg.shape
        trg_mask=torch.tril(torch.ones((trg_len,trg_len))).expand(
            N,1,trg_len,trg_len
        )
        return trg_mask
    def forward(self,src,trg):
        src_mask=self.make_src_mask(src)
        trg_mask=self.make_trg_mask(trg)
        enc_srf=self.encoder(src,src_mask)
        out=self.decorder(trg,enc_srf,src_mask,trg_mask)
        return out



if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x=torch.tensor([[1,5,6,4,3,9,5,2],[1,8,7,3,4,5,6,7]]).to(device)
    trg=torch.tensor([[1,7,4,3,5,9,5,2],[1,5,6,2,4,5,6,7]]).to(device)
    src_pad_idx=0
    trg_pad_idx=0
    src_vocab_size=10
    trg_vocab_size = 10
    model=Transformer(src_vocab_size=src_vocab_size,trg_vocab_size=trg_vocab_size,src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx).to(device)
    out=model(x,trg[:,:-1])
    print(out.shape)
















