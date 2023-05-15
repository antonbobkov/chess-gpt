# -*- coding: utf-8 -*-
"""
Created on Sun May 14 15:39:59 2023

@author: anton
"""

import chess
import chess_encode
import chess_decode

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.enc = chess_encode.EncoderBoard()
        self.dec = chess_decode.DecoderBoard()
        self.moves = []

    def encoded_legal_moves(self):
        return [self.enc.EncodeMove(mv, make_move=False) for mv in self.board.legal_moves]
    
    def make_move(self, enc_move):
        move = self.dec.DecodeMove(enc_move)
        self.enc.EncodeMove(move, make_move=True)
        self.board.push(move)
        self.moves.append(move)
        return move
    
    def get_game_pgn(self):
        return chess.Board().variation_san(self.moves)
