"""
Some example strategies for people who want to create a custom, homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""

from __future__ import annotations
import chess
from chess.engine import PlayResult
import random
from engine_wrapper import MinimalEngine
from typing import Any

import sys

sys.path.append('..\\final_models\\v1')

import v1_model

tokens_filepath = "..\\final_models\\v1\\tokens.txt"
model_filepath = "..\\final_models\\v1\\v1_model_state.pt"
v1_model.load_tokens(tokens_filepath)
chess_v1_model = v1_model.LoadModel(model_filepath)
v1_model.test_game(chess_v1_model)

print ("ML model loaded okay")



class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    pass


# Strategy names and ideas from tom7's excellent eloWorld video

class ChessGPTMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose a random move."""
        print ("$$$ ", args[-1])
        game = args[-2]
        print(game.state["moves"].split())
        moves = []
        for move in game.state["moves"].split():
            moves.append(chess.Move.from_uci(move))
            
        moves_and_weights = v1_model.GetWeightedMovesFromModel(chess_v1_model, moves)
        final_move = v1_model.GetTopLegalMove(moves_and_weights)
        print(final_move, type(final_move))
        
        return PlayResult(final_move, None)


class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


