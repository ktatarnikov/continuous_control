from typing import Any, Callable, Dict, Optional, Sequence, Union

from plots import Plots
from train_runner import TrainRunner

if __name__ == '__main__':
    trainer = TrainRunner(20, "./reacher20/Reacher_Linux/Reacher.x86_64",
                          "./reacher20")
    scores = trainer.run()
    trainer.close()
    Plots().plot_scores(scores)
