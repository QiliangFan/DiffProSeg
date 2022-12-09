from pytorch_lightning.utilities.seed import seed_everything, reset_seed

class SeedContext:

    def __init__(self, seed=106):
        self.seed = seed

    def __enter__(self):
        seed_everything(self.seed)

    def __exit__(self):
        reset_seed()