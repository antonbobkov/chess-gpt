# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:43:44 2023

@author: anton
"""

import chess
import chess_game_tracker

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
block_size = 64 # what is the maximum context length for predictions?
eval_iters = 100
n_embd = 400
n_head = 4
n_layer = 4
dropout = 0.0


device = 'cpu'

tokens = None
token_to_int = {}
int_to_token = {}
vocab_size = None

def load_tokens(tokens_filepath):
  global tokens
  global token_to_int
  global int_to_token
  global vocab_size
  
  with open(tokens_filepath, "r") as f:
    tokens = f.read().splitlines()
      
  tokens = ["ZERO", "START"] + tokens
  vocab_size = len(tokens)

  for i, token in enumerate(tokens):
    token_to_int[token] = i
    int_to_token[i] = token
    
def encode(lst):
    return [token_to_int[token] for token in lst]
def decode(lst):
    return [int_to_token[i] for i in lst]


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
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
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
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

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

def LoadModel(model_filepath):
  
  assert vocab_size > 0
  assert len(tokens) > 0
  
  v1_model = BigramLanguageModel()
  v1_model.load_state_dict(torch.load(model_filepath))
  return v1_model

def EncodeMoveSequence(move_list):
    game = chess_game_tracker.Game()
    for dec_move in move_list:
        game.make_dec_move(dec_move)
    return game, encode(['START'] + game.enc_moves)

def GetWeightedMovesFromModel(mdl, move_list):
    game, enc_moves = EncodeMoveSequence(move_list)
    #print (enc_moves, decode(enc_moves))

    t = torch.tensor(enc_moves, dtype=torch.long)
    t = t[-block_size:]
    t = t[None, :]
    #print (t, t.shape)

    logits, _ = mdl(t)
    logits = logits[:, -1, :]
    logits = logits[0]

    # print (logits.shape)

    legal = game.encoded_legal_moves()
    moves_and_weights = list(enumerate(logits.tolist()))
    moves_and_weights = sorted(moves_and_weights, key=lambda x: -x[1])
    
    final_list = []

    for i, val in moves_and_weights:
        if int_to_token[i] in legal:
            final_list.append((int_to_token[i], game.decode_possible_move(int_to_token[i]), val))
        else:
            final_list.append((int_to_token[i], None, val))
    return final_list

def GetTopLegalMove(moves_and_weights):
    for _, mv, _ in moves_and_weights:
        if mv is not None:
            return mv

def test_game(model):
  moves = []
  
  for i in range(10):
      moves_and_weights = GetWeightedMovesFromModel(model, moves)
      move = GetTopLegalMove(moves_and_weights)
      # print (move)
      moves.append(move)
      
  print (chess.Board().variation_san(moves))
  
if __name__ == "__main__":
  load_tokens("tokens.txt")
  mm = LoadModel("v1_model_state.pt")
  test_game(mm)
