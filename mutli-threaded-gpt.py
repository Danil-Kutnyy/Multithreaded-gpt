import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import sys

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
vocab_size = 65
device = 'cpu'
print('device:', device)
n_embd = 32
n_head = 3
n_layer = 3
dropout = 0.15
# ------------

#insted of nn.linear, this has block_size more weigths, each each token has separate weigths
#in this way transformer can leverage varuous parallel computation
class MultiLinear(nn.Module):
    """Custom Linear layer with separate weights for each time-step"""

    def __init__(self, in_features, out_features, num_timesteps, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_timesteps = num_timesteps

        # Create a single large weight matrix for all time steps
        self.weight = nn.Parameter(torch.randn(num_timesteps, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(num_timesteps, out_features)) if bias else None

    def forward(self, x):
        B, T, C = x.shape
        assert T <= self.num_timesteps, f"Input time steps {T} exceed layer's configured time steps {self.num_timesteps}"

        # Apply separate weight for each time step
        out = torch.einsum('btc,tcd->btd', x, self.weight)  # (B, T, C) @ (T, C, D) -> (B, T, D)

        if self.bias is not None:
            out += self.bias

        return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = MultiLinear(n_embd, head_size, block_size, bias=False)
        self.query = MultiLinear(n_embd, head_size, block_size, bias=False)
        self.value = MultiLinear(n_embd, head_size, block_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = MultiLinear(head_size * num_heads, n_embd, block_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            MultiLinear(n_embd, 4 * n_embd, block_size),
            nn.ReLU(),
            MultiLinear(4 * n_embd, n_embd, block_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        #this is additional layer, with is not present in original transformer
        #it is a simple fully conected layer, that agreagates information
        #within all transformer threads into simgle embedded size dimension.
        #By default its weighs are set to simply copy last token embeding vector
        #from previous layer ( self.ln_f ).
        self.cap = nn.Linear(block_size*n_embd, n_embd)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, MultiLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        
        #set matrix cap weigths specificcly programme values
        with torch.no_grad():
            weight_matrix = torch.randn(n_embd, block_size*n_embd) * 0.0001
            for i in range(n_embd):
                weight_matrix[i, block_size*n_embd-n_embd + i] = 1.0
            self.cap.weight = torch.nn.Parameter(weight_matrix)
            self.cap.bias.zero_()

       

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        
        #additianl agreaation of all conputed tokens
        x = x.view(batch_size, block_size*n_embd)
        x = self.cap(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            #predicts only last token
            targets = targets[:,-1]
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


#laod weigths from normal transfmer. This function allow you to download weigths from normal transofrmer 
#to this multithreded version with apropriate broadcasting.
#not that thsi version has additionla cap layer. By defaulr its weigths are configured
#to take last token from the whole block 
def load_standart_transformer_weigths(model, save_path):
    #save_path = 'path/to/normal_transofrmer_weigths'
    trained_linear_weigths = torch.load(save_path)
    reshaped_trained = {}
    remooved_cap = [(k, v) for (k,v) in model.state_dict().items() if k[:3] != 'cap']
    for ((k_load, v_laod), (k, v)) in zip(trained_linear_weigths.items(), remooved_cap):
        if k[:3] == 'cap':
           reshaped_trained[k] = v
        else:
            if v.shape!=v_laod.shape:
                try:
                    reshaped_trained[k] = v_laod.transpose(0, 1).unsqueeze(0).repeat(block_size, 1, 1)
                except IndexError:
                     reshaped_trained[k] = v_laod.unsqueeze(0).repeat(block_size, 1)
            else:
                reshaped_trained[k] = v_laod
    [reshaped_trained.update({k:v}) for (k,v) in model.state_dict().items() if k[:3] == 'cap']
    model.load_state_dict(reshaped_trained)







