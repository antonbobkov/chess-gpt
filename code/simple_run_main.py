# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:38:48 2023

@author: anton
"""

import chess
import chess_encode
import chess_decode


filepath = "C:\\Users\\anton\\Documents\\code\\chessGPT\\data\\lichess_db_standard_rated_2014-07.pgn"

with open(filepath) as pgn:
    game = chess.pgn.read_game(pgn)

encoder = chess_encode.EncoderBoard()
decoder = chess_decode.DecoderBoard()

i = 1
alg = (chess.Board().variation_san(game.mainline_moves())).split()

for move in game.mainline_moves():
    enc_move = encoder.EncodeMove(move)
    if enc_move == None:
        break
    print (decoder.WhoseMove(), alg[i])
    dec_move = decoder.DecodeMove(enc_move)
    print (move, enc_move)
    print (dec_move)
    assert move == dec_move
    i = i + 1
    if i % 3 == 0:
        i = i + 1