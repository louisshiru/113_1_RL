import matplotlib.pyplot as plt
import re

def extract_losses_from_file(file_path):
    """
    從文件中提取 Training avg loss 數據並返回損失值列表。
    :param file_path: 日誌文件的路徑
    :return: 損失值列表
    """
    with open(file_path, 'r') as file:
        logs = file.read()
    
    # 使用正則表達式提取 Training avg loss
    loss_values = [float(value) for value in re.findall(r"Training avg loss\s+([\d.]+)", logs)]
    return loss_values

def plot_losses(file1, file2):
    """
    根據兩個文件的損失值繪製圖表（橘色和藍色線段）。
    :param file1: 第一個日誌文件的路徑
    :param file2: 第二個日誌文件的路徑
    """
    losses1 = extract_losses_from_file(file1)
    losses2 = extract_losses_from_file(file2)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(losses1) + 1), losses1, linestyle='-', linewidth=2, label='BC Losses', color='blue')
    plt.plot(range(1, len(losses2) + 1), losses2, linestyle='-', linewidth=2, label='MARWIL Losses', color='orange')
    plt.title("Training Average Loss Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("log.png")

if __name__ == "__main__":
    # 替換成你的文件路徑
    file1 = "bc.log"       # 替換為 BC 日誌文件的路徑
    file2 = "marwil.out"   # 替換為 MARWIL 日誌文件的路徑

    try:
        plot_losses(file1, file2)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
