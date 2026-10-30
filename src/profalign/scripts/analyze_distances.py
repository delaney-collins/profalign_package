# analyze_distances.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))
from distance_matrix import build_distance_matrix
from pipeline_utils import ScoreStore
import numpy as np

# Загрузить scores
scores = ScoreStore('results/work/distances.tsv')
scores.load()

# Построить матрицу расстояний
results = scores.to_task_results()
matrix, names = build_distance_matrix(
    results,
    normalization_mode=3,
    symmetrize_mode=1, 
    distance_mode=2,
    max_distance=12.0,
    min_coverage=0.667
)

# Статистика
n = len(names)
distances = []
for i in range(n):
    for j in range(i+1, n):
        distances.append(matrix[i,j])

distances = np.array(distances)
print(f"Профилей: {n}")
print(f"Пар: {len(distances)}")
print(f"\nРаспределение расстояний:")
print(f"  Min:    {distances.min():.3f}")
print(f"  Max:    {distances.max():.3f}")
print(f"  Mean:   {distances.mean():.3f}")
print(f"  Median: {np.median(distances):.3f}")
print(f"\nКвантили:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    print(f"  {int(q*100)}%: {np.quantile(distances, q):.3f}")

print(f"\nПары с distance < 1.2 (текущий порог):")
for i in range(n):
    for j in range(i+1, n):
        if matrix[i,j] < 1.2:
            print(f"  {names[i]} — {names[j]}: {matrix[i,j]:.3f}")

print(f"\nПары с distance < 2.0:")
for i in range(n):
    for j in range(i+1, n):
        if matrix[i,j] < 2.0:
            print(f"  {names[i]} — {names[j]}: {matrix[i,j]:.3f}")