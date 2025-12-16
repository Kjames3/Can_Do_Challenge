import pstats

def analyze(filename="server.prof"):
    print(f"--- Profiling Results: {filename} ---")
    p = pstats.Stats(filename)
    p.strip_dirs()
    
    print("\n[ Top 20 by Cumulative Time (includes subcalls) ]")
    # This shows where the program spent most of its time generally (usually waiting in loops)
    p.sort_stats('cumulative').print_stats(20)
    
    print("\n[ Top 20 by Internal Time (CPU heavy functions) ]")
    # This shows where the CPU was actually burning cycles
    p.sort_stats('time').print_stats(20)

if __name__ == "__main__":
    analyze()
