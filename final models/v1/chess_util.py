# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:37:30 2023

@author: anton
"""

import chess

def initial_setup():
    piece_to_square = {}

    piece_to_square["W:R1"] = chess.A1
    piece_to_square["W:N1"] = chess.B1
    piece_to_square["W:BD"] = chess.C1
    piece_to_square["W:Q"]  = chess.D1
    piece_to_square["W:K"]  = chess.E1
    piece_to_square["W:BL"] = chess.F1
    piece_to_square["W:N2"] = chess.G1
    piece_to_square["W:R2"] = chess.H1

    piece_to_square["B:R1"] = chess.A8
    piece_to_square["B:N1"] = chess.B8
    piece_to_square["B:BL"] = chess.C8
    piece_to_square["B:Q"]  = chess.D8
    piece_to_square["B:K"]  = chess.E8
    piece_to_square["B:BD"] = chess.F8
    piece_to_square["B:N2"] = chess.G8
    piece_to_square["B:R2"] = chess.H8
    
    square_to_piece = {}
    
    for piece in piece_to_square.keys():
        square_to_piece[piece_to_square[piece]] = piece
    
    return piece_to_square, square_to_piece

def is_castle(piece, move):
    fr_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    return ":K" in piece and abs(to_file-fr_file) == 2
    
def get_castled_rook_position(move):
    fr_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    rank = chess.square_rank(move.from_square)

    if to_file > fr_file: # short
        rook = "R2"
    else: # long
        rook = "R1"
    
    if (rank == 0):
        rook = "W:" + rook
    else:
        rook = "B:" + rook
        
    return rook, chess.square((to_file + fr_file) // 2, rank)

