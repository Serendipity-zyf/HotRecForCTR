"""
Main script for deep learning-based CTR prediction.
"""

import argparse

from utils import import_modules
from utils import ColorLogger
from utils import pretty_dict
from prepare import setup_components

logger = ColorLogger(name="Main")


def args_parser():
    parser = argparse.ArgumentParser(description="CTR Prediction with Deep Learning Models")
    parser.add_argument("--is_wandb", action="store_true", help="Enable wandb logging")

    group = parser.add_argument_group("wandb")
    group.add_argument("--wandb_entity", type=str, help="Wandb entity (required if --is_wandb is used)")
    group.add_argument("--wandb_project", type=str, default="CriteoForCTR", help="Wandb project name")

    args = parser.parse_args()

    if args.is_wandb and not args.wandb_entity:
        parser.error("--wandb_entity is required when using --is_wandb")

    return args


def main():

    args = args_parser()

    # Scan and Import modules with interactive selection
    selected = import_modules(interactive=True)

    # Set up all components and get model info
    components, project_info = setup_components(selected)

    # Show all available components
    logger.info("Summarization of Selected Components:")
    pretty_dict(selected, title="Selected")

    if args.is_wandb:
        import wandb

        wdb = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=project_info,
        )
    else:
        wdb = None

    trainer = components.pop("trainer")
    trainer.setup_wandb(wdb)
    trainer.train(**components)


if __name__ == "__main__":
    main()
