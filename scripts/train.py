from biomol_surface_unsup.utils.config import load_experiment_config
from biomol_surface_unsup.training.trainer import Trainer

def main():
    cfg = load_experiment_config()
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()