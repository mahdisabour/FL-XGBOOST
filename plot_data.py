import matplotlib.pyplot as plt
import numpy as np 

def get_specific_acc_list(path: str = "./results.txt"):
    result = {}
    n_client = 1
    with open(path, "r") as file:
        for line in file:
            if "xgboost" in line:
                n_client = int(line.split(".log")[0][-1])
                result.update({n_client: []})
            elif "****" in line:
                pass
            else:
                acc = float(line.split(",")[-1].strip())
                result[n_client].append(acc)
    return result


if __name__ == "__main__":
    result = get_specific_acc_list()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for n_client, data in result.items():
        plt.figure()  # Create a new figure
        plt.plot(data, color=colors[n_client % len(colors)], label=f'Number of clients: {n_client}')
        plt.title(f'Number of clients: {n_client}')
        plt.xlabel('round number')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./fig/number_of_clients_{n_client}.jpg")

    plt.figure()
    for n_clients, data in result.items():
        plt.plot(data, color=colors[(n_clients // 2) % len(colors)], label=f'Number of clients: {n_clients}')
    plt.title('Comparison')
    plt.xlabel('round number')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./fig/all.jpg")