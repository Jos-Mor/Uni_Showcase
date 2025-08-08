# K-Means Clustering with OpenMP

A parallel implementation of the K-Means clustering algorithm I built for my Concurrent Programming class. This was my first real dive into parallel programming, and I learned a lot about how to make code run faster using multiple threads.

## What This Project Does

K-Means is a clustering algorithm that groups data points into clusters. I implemented two versions:
- **Sequential version** - runs on a single thread (the baseline)
- **OpenMP version** - uses multiple threads to speed things up

The goal was to see how much faster I could make the algorithm by parallelizing the computationally expensive parts.

## Team Members
- **Guilherme Antunes** (nº 70231)
- **José Morgado** (nº 59457)  
- **Pedro Mascarenhas** (nº 63240)

## What I Learned

### OpenMP Basics
This was my introduction to OpenMP, which makes parallel programming in C way easier:
- Used `#pragma omp parallel for` to split loops across multiple threads
- Learned about race conditions
- Figured out how to use reduction operations to safely combine results from different threads
- Got experience with thread synchronization

### Performance Analysis
- Compared execution times between sequential and parallel versions
- Learned that not everything speeds up linearly with more threads
- Discovered that small datasets sometimes run slower in parallel due to overhead
- Used timing functions to measure actual performance improvements

## How It Works

The K-Means algorithm has two main computational steps that I parallelized:

1. **Distance Calculation**: For each data point, find the closest centroid
   - This is embarrassingly parallel - each thread can work on different data points
   - Used `#pragma omp parallel for` with reduction to count changes

2. **Centroid Updates**: Calculate new centroid positions based on assigned points
   - Trickier to parallelize because multiple threads might update the same centroid
   - Used reduction operations to safely accumulate sums across threads

## Building and Running

```bash
# Compile both versions
make all

# Run sequential version
./kmeans_seq input_file num_clusters max_iterations change_threshold movement_threshold output_file

# Run parallel version  
./kmeans_omp input_file num_clusters max_iterations change_threshold movement_threshold output_file
```

### Example
```bash
./kmeans_seq test_files/input2D.inp 5 100 0.01 0.001 output.txt
./kmeans_omp test_files/input2D.inp 5 100 0.01 0.001 output.txt
```

## Testing Framework

I wrote a Python script that automatically:
- Runs both versions with different parameter sets
- Verifies that both versions produce identical results (super important!)
- Measures and compares execution times
- ~~Generates graphs showing speedup~~ (matplotlib wasn't working on the cluster)

```bash
cd test_scripts
python test_script.py
```

The script uses SHA-256 hashing to verify that the parallel version produces exactly the same clustering results as the sequential version.

## Performance Results

From my testing, the OpenMP version typically shows:
- **2-3x speedup** on 4 cores for larger datasets
- **Diminishing returns** beyond 8 threads (probably due to memory bandwidth)
- **Overhead issues** with very small datasets where parallel version is actually slower

The speedup isn't perfect because:
- Some parts of the algorithm can't be parallelized
- Thread creation and synchronization have overhead
- Memory bandwidth becomes a bottleneck with too many threads

## What I'd Do Differently

Looking back, there are definitely things I could improve:
- Better load balancing (some threads finish way before others)
- More sophisticated memory access patterns
- Maybe try other parallel programming models like MPI for distributed systems
- Add more robust error handling

## Course Context

This was built for the **Concurrent Programming** course as part of learning:
- Parallel programming fundamentals
- OpenMP directives and best practices
- Performance analysis and optimization
- Debugging parallel code (which is way harder than sequential!)

It's not the most sophisticated parallel implementation out there, but it was a great learning experience and really helped me understand the challenges and benefits of parallel programming.

## Files Structure

```
src/
├── kmeans.c          # Sequential implementation
├── kmeans_omp.c      # OpenMP parallel implementation
├── Makefile          # Build configuration
test_files/           # Input datasets of various sizes
test_scripts/         # Automated testing and benchmarking
```
