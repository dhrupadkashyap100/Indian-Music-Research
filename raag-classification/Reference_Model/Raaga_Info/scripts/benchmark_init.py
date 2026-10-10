import time
from swaragram_dataset import SwaragramDataset

def main():
    print("Benchmarking dataset initialization (Before)...")
    start_time = time.time()
    dataset = SwaragramDataset()
    duration = time.time() - start_time
    print(f"Initialization took {duration:.4f} seconds.")

if __name__ == '__main__':
    main()
