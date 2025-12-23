# Pacman Multi-Agent AI ᗧ • • • 🍒

## Project Overview
This project focuses on the implementation of intelligent agents for the classic Pacman game. The goal was to apply **Adversarial Search** and **Probabilistic reasoning** concepts to create a player capable of winning against intelligent ghosts.

## Implemented Agents & Algorithms

### 1. Reflex Agent
* **Logic:** A rule-based agent using custom feature evaluation (food distance, ghost proximity).
* **Performance:** Achieved a **100% Win Rate** on `testClassic` and consistent scoring on `mediumClassic` by balancing risk (ghosts) and reward (food).

### 2. Minimax Agent
* **Logic:** Implementation of the Minimax algorithm to assume optimal play from the ghosts (adversaries).
* **Depth:** Handles recursive search depth where 1 level includes Pacman + all ghosts.
* **Result:** Successfully anticipates ghost traps.

### 3. Alpha-Beta Pruning
* **Logic:** Optimized the Minimax search by pruning irrelevant branches of the game tree.
* **Impact:** Allows for deeper search trees within the same time constraints (efficient decision making).

### 4. Expectimax Agent
* **Logic:** Models the ghosts not as optimal adversaries but as agents with probabilistic behavior.
* **Performance:** Achieved a **71% Win Rate** on `minimaxClassic`, outperforming standard Minimax by taking calculated risks.

## Technologies
* **Language:** Python
* **Core Concepts:** Game Theory, Recursion, Heuristics, State Machines.

## Usage
To run the AlphaBeta agent (example):
```bash
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=4
