from matplotlib import pyplot as plt
import random


def plot_line_length(line_length):
    print("We are now in data_visualization")
    print(line_length)
    print(line_length[0][0])
    print(line_length[0][1])
    print(".....")
    for line in range(len(line_length)):
        time = line_length[line][0]
        line_length_over_time = line_length[line][1]
        plt.step(time, line_length_over_time)
        plt.xlabel('Time in Seconds')
        plt.ylabel('LineLength')
        plt.title(f"Line Number {line+1}")
        plt.show()

def generate_random(input):
    return random.randint(input, 100)





if __name__ == "__name__":
    plot_line_length()

