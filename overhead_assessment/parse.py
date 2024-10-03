import re
import os
import numpy as np
import matplotlib.pyplot as plt

def extract_values(file_path):
    """Extract performance values (GFLOPS, Bandwidth) from a file."""
    
    # Initialize the result dictionary
    result = {
        "SP Peak GFLOPS": {},
        "L1 Bandwidth": {},
        "DRAM Bandwidth": {}
    }

    # Regular expressions for single-threaded and multi-threaded values
    regex_patterns = {
        "SP Peak GFLOPS": [re.compile(r'"SP Vector FMA Peak \(single-threaded\)"[\s\S]*?val":\s*([\d\.]+)'),
                           re.compile(r'"SP Vector FMA Peak"[\s\S]*?val":\s*([\d\.]+)')],
        "L1 Bandwidth": [re.compile(r'"L1 Bandwidth \(single-threaded\)"[\s\S]*?val":\s*([\d\.]+)'),
                         re.compile(r'"L1 Bandwidth"[\s\S]*?val":\s*([\d\.]+)')],
        "DRAM Bandwidth": [re.compile(r'"DRAM Bandwidth \(single-threaded\)"[\s\S]*?val":\s*([\d\.]+)'),
                           re.compile(r'"DRAM Bandwidth"[\s\S]*?val":\s*([\d\.]+)')]
    }

    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

        # Extract single-threaded and multi-threaded values
        for key, patterns in regex_patterns.items():
            result[key]["Single-threaded"] = float(patterns[0].search(content).group(1))
            result[key]["Multi-threaded"] = float(patterns[1].search(content).group(1))

    return result


def plot_data(model_type, emb_values, colors, folder_path, eps, marker, color_code):
    """Plot performance data for given models."""
    Ps, pis = [], []
    
    for i, emb in enumerate(emb_values):
        filename = f"{model_type}-{emb}.txt"
        path = os.path.join(folder_path, filename)
        result = extract_values(path)

        pi = result["SP Peak GFLOPS"]["Multi-threaded"]
        P = pi / result["DRAM Bandwidth"]["Multi-threaded"]

        plt.plot(P, pi, color=color_code[i], marker=marker, markersize=20)
        plt.text(P + eps, pi + eps, f"{model_type.upper()}-{emb}", fontsize=20, ha='left', va='bottom', color=color_code[i])
        
        Ps.append(P)
        pis.append(pi)
    
    return Ps, pis


def main():
    folder_path = './reports'  # Set the path to the folder containing the reports
    eps = 5e-5  # Small offset for text placement in the plot

    # RESOLVE model data
    ds = [768, 1024]
    emb_values_lars = [32, 64]
    comps = [32]
    Ps, pis = [], []

    for d in ds:
        for e in emb_values_lars:
            for c in comps:
                filename = f"lars-{e}-{d}-{c}.txt"
                path = os.path.join(folder_path, filename)
                result = extract_values(path)

                pi = result["SP Peak GFLOPS"]["Multi-threaded"]
                P = pi / result["DRAM Bandwidth"]["Multi-threaded"]

                plt.plot(P, pi, color='r', marker='*', markersize=20)
                plt.text(P + eps, pi + eps, f"RESOLVE-{e}-{d}-{c}", fontsize=20, ha='left', va='bottom', color='r')
                Ps.append(P)
                pis.append(pi)

    # MULT model data
    emb_values_mult = [32, 64]
    Ps_mult, pis_mult = plot_data("mult", emb_values_mult, ['b', 'g'], folder_path, eps, 'o', ['b', 'g'])

    # Combine and plot the final data
    Ps = np.array(Ps + Ps_mult)
    pis = np.array(pis + pis_mult)

    plt.ylim(0.999 * pis.min(), 1.001 * pis.max())
    plt.xlim(0.999 * Ps.min(), 1.001 * Ps.max())
    plt.xlabel(r'L1 bandwidth bound $\beta$ (FLOP/Byte)')
    plt.ylabel(r'Processor peak performance $\pi$ (GFLOPS)')
    plt.title('Single Thread Performance')
    plt.show()


if __name__ == "__main__":
    main()

