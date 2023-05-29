# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:35:37 2023

@author: anton
"""

import chess
import chess_util

def encode_square(sq):
    return chess.square_name(sq)

def encode_pawn(move):
    fr_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    fr_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    h = to_file - fr_file
    v = to_rank - fr_rank
    
    prefix = "Pawn:" + chess.square_name(move.from_square) + ":"
    
    if (move.promotion != None):
        return prefix + "Promote" + chess.piece_symbol(move.promotion)
    
    if h < 0:
        return prefix + "CaptureLeft"
    if h > 0:
        return prefix + "CaptureRight"
    if abs(v) == 1:
        return prefix + "Push"
    if abs(v) == 2:
        return prefix + "TwoPush"
      
def encode_non_pawn(piece, move):
    return piece[2:] + ":" + encode_square(move.to_square)

class EncoderBoard:
    def __init__(self):
        _, self.square_to_piece = chess_util.initial_setup()
    
    def EncodeMove(self, move, make_move=True):
        if move.promotion != None:
            return None
        
        if move.from_square in self.square_to_piece:
            piece = self.square_to_piece[move.from_square]
            if make_move:
              self.square_to_piece[move.to_square] = piece
            if make_move and chess_util.is_castle(piece, move):
                rook, rook_sq = chess_util.get_castled_rook_position(move)
                self.square_to_piece[rook_sq] = rook
            return encode_non_pawn(piece, move)
        else:
            if make_move and move.to_square in self.square_to_piece:
                del self.square_to_piece[move.to_square]
            return encode_pawn(move)

