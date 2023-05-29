# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:40:00 2023

@author: anton
"""

import chess
import chess_util


def DecodePawnMove(curr_pos, enc, is_white):
    dx = 0
    dy = 1
    if enc == "Push":
        pass
    elif enc == "TwoPush":
        dy = 2
    elif enc == "CaptureLeft":
        dx = -1
    elif enc == "CaptureRight":
        dx = 1
    else:
        assert False, enc
    if not is_white:
        dy = -dy
    return chess.square(
        chess.square_file(curr_pos) + dx,
        chess.square_rank(curr_pos) + dy)

class DecoderBoard:
    def __init__(self):
        self.piece_to_square, _ = chess_util.initial_setup()
        # self.board = chess.Board()
        self.is_white = True
        
    def AddColor(self, piece):
        if self.is_white: 
            return "W:" + piece
        else: 
            return "B:" + piece
        
    def WhoseMove(self):
        if self.is_white: 
            return "White"
        else: 
            return "Black"
        
    def NextMove(self):
        self.is_white = not self.is_white
    
    def DecodeMove(self, move):
        move = move.split(":")
        piece = move[0]
        
        if piece in ["Q", "BD", "BL", "R1", "R2", "K", "N1", "N2"]:
            piece = self.AddColor(piece)
            curr_pos = self.piece_to_square[piece]
            next_pos = chess.parse_square(move[1])
        elif piece == "Pawn":
            curr_pos = chess.parse_square(move[1])
            next_pos = DecodePawnMove(curr_pos, move[2], self.is_white)
        else:
            assert False, piece
            
        if piece != "Pawn":
            assert piece in self.piece_to_square
            self.piece_to_square[piece] = next_pos
            
        move = chess.Move(curr_pos, next_pos)
        
        if (chess_util.is_castle(piece, move)):
            rook, rook_sq = chess_util.get_castled_rook_position(move)
            # print (rook, rook_sq, chess.square_name(rook_sq))
            self.piece_to_square[rook] = rook_sq
            
        self.NextMove()
        
        # print(piece, chess.square_name(curr_pos), chess.square_name(next_pos))
            
        return move
