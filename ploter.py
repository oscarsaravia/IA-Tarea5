# CODIGO UTILIZADO EXTRAIDO DE: https://github.com/python-engineer/snake-ai-pytorch
from IPython import display
import matplotlib.pyplot as plt

plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.title('Training...')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
