## Locality-Aware Transformer for Multiclass Anomaly Detection

This repository implements a unified model (Locality-Aware Transformer) for multi-class anomaly detection in industrial images using the MVTec-AD dataset.

## 1. Installation and Setup

Before you begin, ensure you have Python and the necessary libraries installed. We recommend using a package manager like conda to create a virtual environment with the required dependencies.


**1.1 Downloading the Dataset:**

1. Download the MVTec-AD dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).
2. Unzip the downloaded file and move the extracted folders to create the following directory structure within your project:

```
data/
  MVTec-AD/
    mvtec_anomaly_detection/
    json_vis_decoder/
    train.json
    test.json
```

## 2. Training and Evaluation

**2.1 Training:**

1. Navigate to the experiment directory for MVTec-AD:

Bash

```
cd experiments/MVTec-AD/
```

2. Train the model using the following command:

Bash

```
sh train_torch.sh <NUM_GPUS> <GPU_IDS>
```

- Replace `<NUM_GPUS>` with the number of GPUs you want to use for training.
- Replace `<GPU_IDS>` with a comma-separated list of GPU IDs (e.g., `1,3,4`).

**2.2 Evaluation:**

1. Navigate to the experiment directory (same as training).
    
2. Evaluate the trained model using the following command:
    

Bash

```
sh eval_torch.sh <NUM_GPUS> <GPU_IDS>
```

- Arguments are the same as training.

**Note:** During evaluation, make sure to set the `config.saver.load_path` parameter in your configuration file to point to the saved checkpoint file.

## 3. Visualization of Reconstructed Features

**3.1 Training Decoders:**

1. Navigate to the experiment directory for training vis decoders:

Bash

```
cd experiments/train_vis_decoder/
```

2. Train a decoder for a specific class by running:

Bash

```
sh train_torch.sh <NUM_GPUS> <GPU_IDS> <CLASS_NAME>
```

- Replace placeholders with:
    - `<NUM_GPUS>`: Number of GPUs for training.
    - `<GPU_IDS>`: Comma-separated list of GPU IDs.
    - `<CLASS_NAME>`: The specific anomaly class (e.g., "screw").

**Important:** When using `torch.distributed.launch`, train one vis decoder for a single class at a time.

**3.2 Visualizing Reconstructed Features:**

1. Navigate to the experiment directory for visualization:

Bash

```
cd experiments/vis_recon/
```

2. Visualize reconstructed features for a specific class (currently supports single GPU only):

Bash

```
sh vis_recon_torch.sh <CLASS_NAME>
```

**Important:** When using `torch.distributed.launch`, visualize features for one class at a time.

## Acknowledgments

We acknowledge the valuable contributions of the authors behind the  [UniAD](https://github.com/zhiyuanyou/UniAD) and[ DAT ](https://github.com/zhengchen1999/DAT) projects. Our code heavily references these libraries.
