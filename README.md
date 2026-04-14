# scGPL: Single-cell Universal Gene Prediction Model

## Overview

scGPL (Single-cell Gene Prediction with Learned representations) is a multi-branch architecture model for single-cell transcription factor-target gene prediction. The model integrates Transformer encoders, subgraph graph neural networks (GNNs), and global graph neural networks to effectively learn complex relationships in gene regulatory networks.

## Key Features

- **Multi-branch Architecture**: Integrates Transformer, subgraph GNN, and global GNN for comprehensive feature extraction
- **Flexible Subgraph Extraction Strategies**: Supports BFS, importance-based priority search, and personalized PageRank
- **Multi-GPU Parallel Support**: Built-in multi-GPU concurrent execution for large-scale experiments
- **Multiple GNN Types**: Supports GAT, GCN, GraphSAGE, and GIN graph neural networks
- **Rich Hyperparameter Configuration**: Detailed model architecture and training parameter tuning

## Environment Requirements

- Python 3.8+
- PyTorch 2.4+
- CUDA (recommended for GPU acceleration)
- See `environment.yml` for complete dependency list

## Installation

### Option 1: Using Conda Environment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd scGPL

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate genelink
```

### Option 2: Manual Installation

```bash
# Install core dependencies
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn tqdm transformers
pip install torch-geometric  # For graph neural networks

# Install additional packages as needed
pip install matplotlib seaborn scanpy anndata
```

## Data Format

The project expects data in the following directory structure:

```
data/
├── Dataspilt/
│   ├── Specific/
│   │   ├── hESC/
│   │   │   ├── TFs_500/
│   │   │   │   ├── BL--ExpressionData.csv    # Gene expression data
│   │   │   │   ├── BL--network.csv           # Gene regulatory network
│   │   │   │   ├── TF.csv                    # Transcription factor list
│   │   │   │   ├── Target.csv                # Target gene list
│   │   │   │   ├── Label.csv                 # Ground truth labels
│   │   │   │   ├── Train_set.csv             # Training set
│   │   │   │   ├── Validation_set.csv        # Validation set
│   │   │   │   └── Test_set.csv              # Test set
```

## Quick Start

### Single Dataset Execution

Run the model on a single dataset with default parameters:

```bash
python scripts/demo.py --data_dir yourpath/data/Dataspilt/Specific/hESC/TFs_1000
```

### Batch Execution Across Multiple Datasets

Use the built-in batch execution script for large-scale experiments:

```bash
# Run with default settings (4 GPUs, all datasets, TFs_1000 configuration)
bash run.sh

# Specify GPUs
bash run.sh --gpus "0,1"

# Run specific network type only
bash run.sh --network_type Specific

# Specify TF configuration
bash run.sh --tf_config TFs_500

# Custom seeds for multiple experimental runs
bash run.sh --seeds "42,66,80,12,100"
```

## Parameter Configuration

### Model Architecture Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embed_size` | 1024 | Transformer embedding dimension |
| `--num_layers` | 4 | Number of Transformer layers |
| `--num_head` | 8 | Number of attention heads |

### GNN Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gnn_type` | gat | GNN type (gat/gcn/graphsage/gin) |
| `--gnn_hidden` | 128 | GNN hidden dimension |
| `--gnn_heads` | 4 | Number of attention heads for GAT |
| `--gnn_dropout` | 0.8 | GNN dropout rate |

### Subgraph Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_subgraph` | True | Enable subgraph GNN branch |
| `--subgraph_strategy` | priority | Subgraph extraction strategy (bfs/priority/ppr) |
| `--subgraph_gnn_layers` | 2 | Number of GNN layers for subgraph branch |
| `--subgraph_hops` | 2 | Number of hops for subgraph extraction |
| `--max_subgraph_size` | 100 | Maximum number of nodes per subgraph |

### Global Graph Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_globalgraph` | True | Enable global graph GNN branch |
| `--globalgraph_gnn_layers` | 2 | Number of GNN layers for global graph branch |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 256 | Batch size |
| `--lr` | 1e-5 | Learning rate |
| `--epochs` | 200 | Maximum number of epochs |
| `--weight_decay` | 1e-5 | Weight decay (L2 regularization) |
| `--patience` | 8 | Early stopping patience |
| `--seed` | 42 | Random seed |

### Learning Rate Scheduling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scheduler_flag` | True | Enable learning rate scheduler |
| `--step_size` | 10 | Step size for scheduler |
| `--gamma` | 0.999 | Multiplicative factor |

### Loss Functions

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--loss_type` | bce | bce/focal/weighted_bce/combined | Loss function type |

## Output Results

After model execution, the following outputs are generated:

1. **Training Logs**: Saved to `yourpath/result/logs/` directory
2. **Model Weights**: Saved to `yourpath/result/models/` directory
3. **Summary Results**: Appended to `yourpath/result/summary.csv`

Performance metrics include:
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- EPR (Early Precision at top-100)

## Usage Examples

### Basic Usage

```bash
# Run with default parameters
python scripts/demo.py

# Custom data directory
python scripts/demo.py --data_dir /path/to/your/data

# Modify model architecture
python scripts/demo.py --embed_size 512 --num_layers 2 --gnn_type gcn
```

### Large-scale Experiments

```bash
# Run on multiple GPUs across all datasets
bash run.sh --gpus "0,1,2,3"

# Ablation study: Transformer branch only
bash run.sh --use_subgraph False --use_globalgraph False

# Ablation study: Subgraph GNN branch only
bash run.sh --use_globalgraph False

# Compare different GNN types
bash run.sh --gnn_type gat
bash run.sh --gnn_type gcn
bash run.sh --gnn_type graphsage
```

### Hyperparameter Tuning

```bash
# Learning rate tuning
bash run.sh --lr 1e-4
bash run.sh --lr 5e-5
bash run.sh --lr 1e-5

# Batch size tuning
bash run.sh --batch_size 128
bash run.sh --batch_size 256
bash run.sh --batch_size 512

# Subgraph strategy comparison
bash run.sh --subgraph_strategy bfs
bash run.sh --subgraph_strategy priority
bash run.sh --subgraph_strategy ppr
```

## Important Notes

1. **GPU Memory**: The model requires significant GPU memory; GPUs with at least 8GB VRAM are recommended
2. **Data Paths**: Ensure data paths are correctly configured; default uses relative paths `yourpath/data/Dataspilt/...`
3. **Multi-GPU Execution**: Ensure CUDA environment is properly configured and specified GPU IDs exist
4. **Output Paths**: Ensure output directories exist and have write permissions
5. **Random Seeds**: Set fixed random seeds for reproducible results

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `batch_size`
   - Decrease `embed_size` or `gnn_hidden`
   - Disable some GNN branches

2. **Data path errors**
   - Check data directory structure
   - Verify file paths are correct

3. **Multi-GPU execution failures**
   - Check CUDA version
   - Verify GPU IDs are correct
   - Try single GPU execution for testing



## License

This project is licensed under the MIT License.
