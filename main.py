from lightning.pytorch.cli import LightningCLI

def cli_main():
    cli = LightningCLI(save_config_callback=None,
                       subclass_mode_model=True,)

if __name__ == '__main__':
    cli_main()