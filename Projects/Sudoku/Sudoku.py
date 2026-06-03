#-------------------------------------------------------------------------
# Code Written by Maxwell Pizzica and Ryan Adolph
# for CSCI 446 - Artificial Intelligence
#
# This program contains a Sudoku solver with implementations of:
#   - Simple Backtracking (bt)
#   - Genetic Algorithm (ga)
#-------------------------------------------------------------------------

import os
import matplotlib.pyplot as plt
import copy
import random

ALGORITHM   = 'ga'       # 'bt' or 'ga'
PUZZLE_TYPE = 'hard'
PUZZLE_PATH = 'sample_data/Hard-P2.txt'

# Genetic Algorithm Hyperparameters
POPULATION_SIZE = 700
MAX_ITERATIONS  = 200
ELITE_K         = 3      # number of elites carried forward each generation
STRAGGLER_N     = 25     # number of worst individuals pruned when avg_fitness < 30
MUTATION_RATE   = 0.2
NEW_BLOOD_EVERY = 20     # inject fresh individuals every N generations
NEW_BLOOD_COUNT = 50

# -----------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------

def build_grid(path):
    grid = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            row = [0 if x.strip() == "?" else int(x.strip())
                   for x in line.strip().split(",")]
            grid.append(row)
    return grid


def print_grid(grid):
    for row in grid:
        print(" ".join(str(n) for n in row))


def write_output(grid, filename):
    with open(filename, "w") as f:
        for row in grid:
            f.write(" ".join(map(str, row)) + "\n")


# -----------------------------------------------------------------------
# Main dispatcher
# -----------------------------------------------------------------------

def sudoku_main():
    base       = os.path.splitext(PUZZLE_PATH)[0]   # e.g. "sample_data/Hard-P2"
    basename   = os.path.basename(base)              # "Hard-P2"
    difficulty, puzzle_id = basename.split('-')
    number     = puzzle_id[1:]
    puzzle_num = f"puzzle{int(number):02d}"

    grid = build_grid(PUZZLE_PATH)
    print("Initial puzzle:")
    print_grid(grid)
    print()

    if ALGORITHM == 'bt':
        simple_backtracking(grid)
        result = grid

    elif ALGORITHM == 'ga':
        result = genetic_algorithm_runner(grid)

    else:
        raise ValueError(f"Unknown algorithm '{ALGORITHM}'. Choose 'bt' or 'ga'.")

    out_file = f"{GROUP_ID}_{ALGORITHM}_{PUZZLE_TYPE}_{puzzle_num}.txt"
    write_output(result, out_file)
    print("Solution:")
    print_grid(result)
    print(f"\nOutput written to {out_file}")


# -----------------------------------------------------------------------
# Simple Backtracking
# -----------------------------------------------------------------------

def is_valid(grid, r, c, k):
    """Return True if placing k at (r, c) is valid under Sudoku rules."""
    if k in grid[r]:
        return False
    if k in (grid[i][c] for i in range(9)):
        return False
    br, bc = (r // 3) * 3, (c // 3) * 3
    if k in (grid[i][j] for i in range(br, br + 3) for j in range(bc, bc + 3)):
        return False
    return True


def simple_backtracking(grid, r=0, c=0, count=0):
    if r == 9:
        print("Total Nodes Visited:", count)
        return True
    if c == 9:
        return simple_backtracking(grid, r + 1, 0, count)
    if grid[r][c] != 0:
        return simple_backtracking(grid, r, c + 1, count)
    for k in range(1, 10):
        if is_valid(grid, r, c, k):
            count += 1
            grid[r][c] = k
            if simple_backtracking(grid, r, c + 1, count):
                return True
            grid[r][c] = 0
    return False


# -----------------------------------------------------------------------
# Genetic Algorithm — helpers
# -----------------------------------------------------------------------

def get_fixed_mask(original_grid):
    """
    Return a 9x9 boolean mask: True where the cell was given (non-zero)
    in the ORIGINAL puzzle. Must be called before any filling.
    """
    return [[original_grid[r][c] != 0 for c in range(9)] for r in range(9)]


def generate_population(original_grid, fixed_mask, size):
    """
    Fill each empty row with a random permutation of the missing digits
    (1-9 minus whatever fixed values are already in that row).
    This guarantees every row in every individual has exactly one of each digit.
    """
    population = []
    all_digits = list(range(1, 10))
    for _ in range(size):
        individual = copy.deepcopy(original_grid)
        for r in range(9):
            present   = {individual[r][c] for c in range(9) if fixed_mask[r][c]}
            missing   = [v for v in all_digits if v not in present]
            random.shuffle(missing)
            idx = 0
            for c in range(9):
                if not fixed_mask[r][c]:
                    individual[r][c] = missing[idx]
                    idx += 1
        population.append(individual)
    return population


def fitness(grid):
    """
    Number of conflicts across all columns and all 3×3 sub-grids.
    (Rows are conflict-free by construction in generate_population.)
    A perfect solution has fitness == 0.
    """
    score = 0
    # Column conflicts
    for c in range(9):
        col = [grid[r][c] for r in range(9)]
        score += 9 - len(set(col))
    # Sub-grid conflicts
    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            box = [grid[r][c]
                   for r in range(box_r, box_r + 3)
                   for c in range(box_c, box_c + 3)]
            score += 9 - len(set(box))
    return score


def tournament_select(population):
    """
    Pick 3 random individuals; return the best two by fitness.
    """
    contestants = random.sample(population, 3)
    contestants.sort(key=fitness)
    return contestants[0], contestants[1]


def crossover(parent1, parent2):
    """Row-uniform crossover: each row inherited randomly from one parent."""
    child = []
    for r in range(9):
        child.append(parent1[r][:] if random.random() < 0.5 else parent2[r][:])
    return child


def mutate(child, fixed_mask, mutation_rate=MUTATION_RATE):
    """
    With probability mutation_rate, swap two non-fixed cells within the same row.
    Swapping within a row preserves the row-valid invariant.
    """
    result = copy.deepcopy(child)
    for r in range(9):
        if random.random() < mutation_rate:
            free = [c for c in range(9) if not fixed_mask[r][c]]
            if len(free) >= 2:
                c1, c2 = random.sample(free, 2)
                result[r][c1], result[r][c2] = result[r][c2], result[r][c1]
    return result


def select_elites(population, k=ELITE_K):
    return sorted(population, key=fitness)[:k]


def prune_stragglers(population, n=STRAGGLER_N):
    """Remove the n worst individuals (highest fitness) from the population."""
    worst = sorted(population, key=fitness, reverse=True)[:n]
    # Use id() comparison to avoid slow equality checks on nested lists
    worst_ids = {id(w) for w in worst}
    return [ind for ind in population if id(ind) not in worst_ids]


def compute_average_fitness(population):
    total = sum(fitness(ind) for ind in population)
    avg   = total / len(population)
    print(f"  avg fitness: {avg:.2f}")
    return avg


def plot_fitness_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker='o', color='blue')
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness Over Generations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------
# Genetic Algorithm — main loop (iterative, no recursion)
# -----------------------------------------------------------------------

def genetic_algorithm_runner(original_grid):
    fixed_mask = get_fixed_mask(original_grid)

    population       = generate_population(original_grid, fixed_mask, POPULATION_SIZE)
    avg_fitness_hist = []

    for generation in range(1, MAX_ITERATIONS + 1):
        print(f"Generation {generation}  |  population: {len(population)}")
        avg = compute_average_fitness(population)
        avg_fitness_hist.append(avg)

        # Check for a solution in the current population
        for ind in population:
            if fitness(ind) == 0:
                print(f"Solution found at generation {generation}!")
                plot_fitness_history(avg_fitness_hist)
                return ind

        # --- Build next generation ---
        next_gen = []

        # Produce offspring
        target_size = POPULATION_SIZE
        while len(next_gen) < target_size - ELITE_K:
            p1, p2  = tournament_select(population)
            child   = crossover(p1, p2)
            if random.random() < 0.4:
                child = mutate(child, fixed_mask)
            next_gen.append(child)

        # Elitism: carry forward the best individuals unchanged
        elites = select_elites(population, k=ELITE_K)
        next_gen.extend(elites)

        # Prune worst individuals once avg fitness is low enough to matter
        if avg < 30:
            next_gen = prune_stragglers(next_gen, n=STRAGGLER_N)

        # Inject fresh blood periodically to maintain diversity
        if generation % NEW_BLOOD_EVERY == 0:
            new_blood = generate_population(original_grid, fixed_mask, NEW_BLOOD_COUNT)
            next_gen.extend(new_blood)

        population = next_gen

    # Return best found even if no perfect solution was reached
    best = min(population, key=fitness)
    print(f"Max iterations reached. Best fitness: {fitness(best)}")
    plot_fitness_history(avg_fitness_hist)
    return best



sudoku_main()