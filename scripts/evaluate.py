from biomol_surface_unsup.utils.config import load_eval_config
from biomol_surface_unsup.training.trainer import Trainer

def main():
    cfg = load_eval_config()
    trainer = Trainer(cfg)
    trainer.evaluate()

if __name__ == "__main__":
    main()