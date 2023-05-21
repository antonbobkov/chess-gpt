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
        self.enc_moves = []

    def encoded_legal_moves(self):
        return [self.enc.EncodeMove(mv, make_move=False) for mv in self.board.legal_moves]
    
    def decode_possible_move(self, enc_move):
        return self.dec.DecodeMove(enc_move, make_move=False)
    
    def make_move(self, enc_move):
        move = self.dec.DecodeMove(enc_move)
        self.enc.EncodeMove(move, make_move=True)
        self.board.push(move)
        self.moves.append(move)
        return move
      
    def make_dec_move(self, dec_move):
        enc_move = self.enc.EncodeMove(dec_move, make_move=True)
        dec_move_v2 = self.dec.DecodeMove(enc_move)
        assert dec_move == dec_move_v2
        self.board.push(dec_move)
        self.moves.append(dec_move)
        self.enc_moves.append(enc_move)
      
    def get_game_pgn(self):
        return chess.Board().variation_san(self.moves)
