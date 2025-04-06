import psutil
import os
import re
from time import perf_counter

def format_time(seconds):
    """Format time in appropriate units"""
    ms = seconds * 1000
    if ms < 1:
        return f"{ms * 1000:.2f}µs"
    return f"{ms:.2f}ms"

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # in MB

def update_readme_benchmark(markdown_table):
    """Updates the benchmark section in README.md."""
    readme_path = "README.md"
    start_marker = "<!-- BENCHMARK_START -->"
    end_marker = "<!-- BENCHMARK_END -->"
    
    try:
        with open(readme_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {readme_path} not found.")
        return

    pattern = re.compile(f"({re.escape(start_marker)}).*?({re.escape(end_marker)})", re.DOTALL)
    replacement = f"{start_marker}\n{markdown_table}\n{end_marker}"
    
    new_content, num_replacements = pattern.subn(replacement, content)
    
    if num_replacements > 0:
        try:
            with open(readme_path, 'w') as f:
                f.write(new_content)
            print(f"Successfully updated benchmark table in {readme_path}")
        except IOError:
            print(f"Error: Could not write to {readme_path}.")
    else:
        print(f"Error: Benchmark markers ({start_marker}, {end_marker}) not found in {readme_path}.")

class BenchmarkRunner:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.results = {}
        self.start_time = None

    def start_timer(self):
        """Start timing an operation"""
        self.start_time = perf_counter()

    def stop_timer(self):
        """Stop timing and return elapsed time"""
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        elapsed = perf_counter() - self.start_time
        self.start_time = None
        return elapsed

    def add_result(self, method_name, result_dict):
        """Add a benchmark result"""
        self.results[method_name] = result_dict

    def print_results(self):
        """Print benchmark results in a formatted table"""
        print(f"\n{self.name} Benchmark Results")
        print("=" * 80)
        print(f"Description: {self.description}")
        print("-" * 80)
        
        # Get all metrics from results
        metrics = set()
        for result in self.results.values():
            metrics.update(result.keys())
        metrics = sorted(list(metrics))
        
        # Print header
        header = "Method".ljust(20)
        for metric in metrics:
            header += metric.ljust(15)
        print(header)
        print("-" * len(header))
        
        # Print results
        for method, result in self.results.items():
            line = method.ljust(20)
            for metric in metrics:
                value = result.get(metric, "N/A")
                if isinstance(value, float):
                    line += f"{value:.4f}".ljust(15)
                else:
                    line += str(value).ljust(15)
            print(line)
        print("=" * 80) 