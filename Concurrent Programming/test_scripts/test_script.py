import hashlib
import multiprocessing
import os
import re
import subprocess
import json

#import matplotlib.pyplot as plt
import os

'''

This python script allows to:
    1. Create several objects containing the kmeans params
        1.1 You can hardcode them on function populate_runs
        1.2 Or you can choose or create .json files with available params on directory param_files
            and use function populate_runs_from_file( json_path )
            
    2. For each object run the sequential and parallel version of KMeans
        2.1 Possible to set number of wanted threads to use as the second 
        parameter of run_kmeans_executables(runs, nrThreads), if it is not
        given it uses all threads available
        
    3. Verify correctness of parallel KMeans output compared to sequential version using a Hash function (SHA-256)
    
    4. Creates an .txt file comparing the computation times for each param object
    
    5. For each of those computation time comparisons, creates a bar graph comparing computation time
        of sequential vs parallel (omp) versions
        
    
To run don't forget:
    start python environment
    pip install matplotlib
    have compiled executables of kmeans and kmeans_omp and if needed change their
    path on KMEANS_OMP_EXE_PATH and KMEANS_SEQ_EXE_PATH

'''

KMEANS_SEQ_EXE_PATH = "../src/kmeans_seq"
KMEANS_OMP_EXE_PATH = "../src/kmeans_omp"
DEBUG = False

def check_environment():
    print("------------ ENVIRONMENT ------------")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {subprocess.run(['python3', '--version'], capture_output=True, text=True).stdout.strip()}")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    
    # Check if we're in the right directory
    expected_files = ["test_script.py", "param_files", "results"]
    for file in expected_files:
        if os.path.exists(file):
            print(f"Found: {file}")
        else:
            print(f"Missing: {file}")


def debug_executables():
    print("------------ EXECUTABLES ------------")
    
    executables = {
        "sequential": KMEANS_SEQ_EXE_PATH,
        "parallel": KMEANS_OMP_EXE_PATH
    }
    
    for name, path in executables.items():
        print(f"\n--- Checking {name} executable ---")
        print(f"Path: {path}")
        print(f"Absolute path: {os.path.abspath(path)}")
        
        if os.path.exists(path):
            print("File exists")
            
            if os.access(path, os.X_OK):
                print("File is executable")
            else:
                print("File is not executable")
                print("  Try: chmod +x", path)
                
            stat_info = os.stat(path)
            print(f"File size: {stat_info.st_size} bytes")
            print(f"File permissions: {oct(stat_info.st_mode)}")
            
        else:
            print("File does not exist")
            print("  Make sure to compile the executables first")

def debug_input_files():
    """Debug function to check input file status"""
    print("\n------------ INPUT FILES ------------")
    
    json_path = "param_files/all_params.json"
    if os.path.exists(json_path):
        print(f"JSON file exists: {json_path}")
        try:
            with open(json_path, 'r') as f:
                params = json.load(f)
            print(f"JSON file is valid, contains {len(params)} parameter sets")
            
            for i, param_set in enumerate(params):
                input_file = param_set.get("input_file", "")
                print(f"\n--- Parameter set {i+1} ---")
                print(f"Input file: {input_file}")
                print(f"Absolute path: {os.path.abspath(input_file)}")
                
                if os.path.exists(input_file):
                    print("Input file exists")
                    stat_info = os.stat(input_file)
                    print(f"File size: {stat_info.st_size} bytes")
                    
                    try:
                        with open(input_file, 'r') as f:
                            first_lines = [f.readline().strip() for _ in range(3)]
                        print("First 3 lines of file:")
                        for j, line in enumerate(first_lines, 1):
                            print(f"  Line {j}: {line}")
                    except Exception as e:
                        print(f"Error reading file: {e}")
                else:
                    print("Input file does not exist")
                    
        except json.JSONDecodeError as e:
            print(f"JSON file is invalid: {e}")
        except Exception as e:
            print(f"Error reading JSON file: {e}")
    else:
        print(f"JSON file does not exist: {json_path}")

def test_executable_with_simple_args():
    """Test running executables with minimal arguments to see what happens"""
    print("\n------------ EXECUTABLE EXECUTION ------------")
    
    executables = {
        "sequential": KMEANS_SEQ_EXE_PATH,
        "parallel": KMEANS_OMP_EXE_PATH
    }
    
    for name, exe_path in executables.items():
        print(f"\n--- Testing {name} executable ---")
        
        if not os.path.exists(exe_path) or not os.access(exe_path, os.X_OK):
            print(f"✗ Skipping {name} - not executable")
            continue
            
        try:
            result = subprocess.run([exe_path], capture_output=True, text=True, timeout=5)
            print(f"Exit code: {result.returncode}")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
        except subprocess.TimeoutExpired:
            print("✗ Executable timed out (might be waiting for input)")
        except Exception as e:
            print(f"✗ Error running executable: {e}")

def test_single_run():
    """Test a single run with detailed error capture"""
    print("\n=------------ SINGLE RUN ------------")
    
    # Try to load the first parameter set
    json_path = "param_files/all_params.json"
    try:
        with open(json_path, 'r') as f:
            param_sets = json.load(f)
        
        if not param_sets:
            print("✗ No parameter sets found")
            return
            
        params = param_sets[0]  # Use first parameter set
        print(f"Using parameter set: {params}")
        
        # Test sequential version
        exe = KMEANS_SEQ_EXE_PATH
        output_file = "results/debug_output.txt"
        
        # Make sure results directory exists
        os.makedirs("results", exist_ok=True)
        
        args = [
            exe,
            params["input_file"],
            str(params["num_clusters"]),
            str(params["max_iterations"]),
            str(params["change_threshold"]),
            str(params["movement_threshold"]),
            output_file
        ]
        
        print(f"Command to run: {' '.join(args)}")
        
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            print(f"Exit code: {result.returncode}")
            
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
                
            if result.returncode == 0:
                print("✓ Execution successful!")
                if os.path.exists(output_file):
                    print(f"✓ Output file created: {output_file}")
                else:
                    print("✗ No output file created")
            else:
                print(f"✗ Execution failed with exit code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            print("✗ Execution timed out")
        except Exception as e:
            print(f"✗ Error during execution: {e}")
            
    except Exception as e:
        print(f"✗ Error setting up test: {e}")

# Executes all param objects in param_sets with kmeans and kmeans_omp and saves results
def run_kmeans_executables(param_sets, nrThreads = -1):

    nrThreadsonCPU = multiprocessing.cpu_count()

    if(nrThreads > nrThreadsonCPU):
        print("Warning: Creating more threads than available")

    executables = {
        "sequential": KMEANS_SEQ_EXE_PATH,
        "parallel": KMEANS_OMP_EXE_PATH
    }

    env = os.environ.copy()
    if nrThreads > 0:
        env["OMP_NUM_THREADS"] = str(nrThreads)

    os.makedirs("results", exist_ok=True)

    times_file = "results/computation_times.txt"
    with open(times_file, "w") as tf:
        for i, params in enumerate(param_sets):
            tf.write(f"Run {i + 1}:\n")
            tf.flush()

            for mode, exe in executables.items():
                output_file = f"results/output_{mode}_{i+1}.txt"

                args = [
                    exe,
                    params["input_file"],
                    str(params["num_clusters"]),
                    str(params["max_iterations"]),
                    str(params["change_threshold"]),
                    str(params["movement_threshold"]),
                    output_file
                ]

                try:
                    print(f"Running {mode} K-means for parameter set {i+1}...")
                    result = subprocess.run(args, check=True, capture_output=True, text=True, env=env)
                    stdout = result.stdout
                    stderr = result.stderr

                    if stderr:
                        print(f"Warning: {mode} K-means stderr: {stderr}")

                    match = re.search(r"Computation:\s+([\d.]+)\s+seconds", stdout)
                    time_str = match.group(1) if match else "N/A"

                    tf.write(f"{mode.capitalize()}: {time_str} seconds\n")
                    tf.flush()

                    print(f"Output saved to {output_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Execution failed for {mode} K-means, param set {i+1}: {e}")
                    tf.write(f"{mode.capitalize()}: FAILED\n")
                    tf.flush()
                except Exception as e:
                    print(f"Error running {mode} K-means, param set {i+1}: {e}")
                    tf.write(f"{mode.capitalize()}: ERROR\n")
                    tf.flush()

            tf.write("\n")
            tf.flush()

# Populates param_objects with hardcoded param objects
def populate_runs():

    param_objects = [{
        "input_file": "../test_files/input2D.inp",
        "num_clusters": 10,
        "max_iterations": 100,
        "change_threshold": 0.01,
        "movement_threshold": 0.001
    }, {
        "input_file": "../test_files/input100D.inp",
        "num_clusters": 5,
        "max_iterations": 500,
        "change_threshold": 0.02,
        "movement_threshold": 0.005
    }, {
        "input_file": "../test_files/input100D.inp",
        "num_clusters": 10,
        "max_iterations": 1000,
        "change_threshold": 0.02,
        "movement_threshold": 0.005
    }]

    return param_objects

# Populates param_objects with hardcoded param objects
def populate_runs_from_file(json_path):
    with open(json_path, 'r') as f:
        param_objects = json.load(f)

    return param_objects

# Returns the hash value of a file using SHA-256
def hash_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# Verifies correctness of every parallel run of param_sets
def verify_integrity(param_sets):

    print("Verifying output correctness...\n")
    all_correct = True

    for i in range(len(param_sets)):
        seq_path = f"results/output_sequential_{i+1}.txt"
        par_path = f"results/output_parallel_{i+1}.txt"

        if not os.path.exists(seq_path) or not os.path.exists(par_path):
            print(f"Missing output file for parameter set {i+1}")
            all_correct = False
            continue

        seq_hash = hash_file(seq_path)
        par_hash = hash_file(par_path)

        if seq_hash == par_hash:
            print(f"Param Set {i+1}: Output is correct")
        else:
            print(f"Param Set {i+1}: Output is wrong.")
            all_correct = False

    return all_correct

# Creates a bar plot comparing the computation times written in the given times_file
def plot_times(times_file="results/computation_times.txt", output_pdf="results/comparison_plot_v1.pdf"):
    sequential_times = []
    parallel_times = []
    run_labels = []

    with open(times_file, "r") as f:
        lines = f.readlines()

    run_index = -1
    for line in lines:
        if line.startswith("Run"):
            run_index += 1
            run_labels.append(f"Run {run_index + 1}")
        elif line.startswith("Sequential:"):
            time_str = line.strip().split(":")[1].strip().split()[0]
            sequential_times.append(float(time_str) if time_str != "N/A" else None)
        elif line.startswith("Parallel:"):
            time_str = line.strip().split(":")[1].strip().split()[0]
            parallel_times.append(float(time_str) if time_str != "N/A" else None)

    x = range(len(run_labels))
    width = 0.35

    #plt.figure(figsize=(10, 6))
    #plt.bar([i - width/2 for i in x], sequential_times, width, label='Sequential', color='skyblue')
    #plt.bar([i + width/2 for i in x], parallel_times, width, label='Parallel', color='salmon')

    #plt.xlabel('Run')
    #plt.ylabel('Computation Time (seconds)')
    #plt.title('Sequential vs Parallel K-means Computation Time')
    #plt.xticks(x, run_labels)
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig(output_pdf)
    #plt.close()

    print(f"Bar chart saved as {output_pdf}")

# Deletes every file in results dir
def cleanse_results_dir():
    results_dir = "results"
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            file_path = os.path.join(results_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def main():
    check_environment()
    if DEBUG is True:
        debug_executables()
        debug_input_files()
        test_executable_with_simple_args()
        test_single_run()


    # Use this to run the hardcoded params objects
    #param_objects = populate_runs()

    # Use this to run the params objects from a specific .json files
    #change file to use as needed
    param_objects = populate_runs_from_file("param_files/5_8param.json")

    #This way uses all threads available on the running machin
    #run_kmeans_executables(param_objects)

    #The second parameter is the number of threads to use on OpenMP
    run_kmeans_executables(param_objects, 16)

    verify_integrity(param_objects)

    #can't use on cluster, matplotlib is not installed
    #plot_times()

if __name__ == '__main__':
    cleanse_results_dir()
    main()

