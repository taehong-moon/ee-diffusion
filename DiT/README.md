## Configuration
Please refer to the [DiT repository](https://github.com/your_repo_link) for setup instructions.

## Fine-tuning Setup
1. Create a `pretrained` folder and download the DiT checkpoint from the following link:
   - [DiT-XL-2-256x256.pt](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt)

## Fine-tuning and Sampling Commands
- Refer to the `commands` folder for training and sampling commands.
  - Currently, only `ddim` and `ddpm` sampling methods are included. Support for `dpm solver` will be added soon.
- The `train_ase.py` script contains a demo dropping schedule. To customize it further, modify the `drop_scheduler.py` (full code update is pending).
