"""Train, validate or test any config: python main.py {fit,validate,test} --config <config.yaml>."""

from lightning.pytorch.cli import LightningCLI


def cli_main():
    """LightningCLI over every LightningModule and LightningDataModule named in the config."""
    LightningCLI(save_config_callback=None, subclass_mode_model=True)


if __name__ == "__main__":
    cli_main()
