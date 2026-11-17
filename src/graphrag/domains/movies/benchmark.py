from pathlib import Path

# The book's ch08 benchmark (template/kg-rag/data/benchmark_data.csv):
# ``question;cypher`` rows, where the Cypher computes the ground truth
# against the movies graph that ``MovieGraphSeeder`` loads.
MOVIES_BENCHMARK_PATH = Path(__file__).parent / "benchmark.csv"
