import random
import matplotlib.pyplot as plt
import numpy as np
# import hydra

import wandb


# @hydra.main(config_path="configs/", config_name="defaults")
def main(cfg=None):
    wandb.init(project="wandb-test", entity="s183920")
    for _ in range(100):
        wandb.log({"test_metric": random.random()})


    # create a random image
    img = plt.imshow(np.random.rand(28, 28))
    wandb.log({"random_image": wandb.Image(img)})
    
if __name__ == "__main__":
    main()