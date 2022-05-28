### MCTS for arbitary size TicTacToe 

```angular2html
usage: mcts.py [-h] [-b BOARD] [-w WIN] [-f] [-m MCTS] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -b BOARD, --board BOARD
  -w WIN, --win WIN     minimum winning length
  -f, --first           AI first move
  -m MCTS, --mcts MCTS  Number of MCTS iterations
  -v, --verbose
```

Example:

Start a game. MCTS based AI makes the first move on the board of size 5 with a winning length of 4 

```angular2html
 python3 mcts.py --board 5 --win 4 --first
```