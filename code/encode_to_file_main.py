# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:44:10 2023

@author: anton
"""

import chess
import chess.pgn
import chess_encode
import chess_decode
import time

filepath = "C:\\Users\\anton\\Documents\\code\\chessGPT\\data\\lichess_db_standard_rated_2014-07.pgn"
intermediate_filepath = "C:\\Users\\anton\\Documents\\code\\chessGPT\\data\\intermediate_tmp.txt"
tokens_filepath = "C:\\Users\\anton\\Documents\\code\\chessGPT\\data\\tokens_tmp.txt"


def encode_game_with_check(game):
    encoder = chess_encode.EncoderBoard()
    decoder = chess_decode.DecoderBoard()
    moves = []

    for move in game.mainline_moves():
        enc_move = encoder.EncodeMove(move)
        if enc_move == None:
            break
        moves.append(enc_move)
        dec_move = decoder.DecodeMove(enc_move)
        assert move == dec_move
    return moves


i = 0
seconds = time.time()

token_set = set()
with open(filepath) as pgn:
    with open(intermediate_filepath, "w") as out_file:
        while True:
            game = chess.pgn.read_game(pgn)
            move_seq = encode_game_with_check(game)
            token_set.update(move_seq)
            out_file.write(" ".join(move_seq) + "\n")
            i = i + 1
            if i % 1000 == 0:
                print (i, round(time.time() - seconds, 2), "sec")
                seconds = time.time()
            if i > 1000:
                break
token_list = sorted(list(token_set))

# print(token_list)

with open(tokens_filepath, "w") as out_file:
    out_file.write("\n".join(token_list) + "\n")
