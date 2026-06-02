# Multimodal Knowledge Graph Experiment Project

This project focuses on knowledge graph data export, multimodal feature construction, knowledge graph embedding (KGE), and link prediction experiments. The overall workflow is:

1. Export sharded graph data from a Neo4j graph database.
2. Enrich graph nodes with multimodal information such as text and images.
3. Train models with different text encoders, image encoders, graph neural networks, and KGE scoring functions.
4. Save experiment logs, model outputs, and evaluation metrics for comparing different configurations.

## Directory Structure

```text
.
|-- database_operations/          # Neo4j connection, statistics, export, and long-text crawling
|-- edge_type_kge/                # Multimodal KGE experiments focused on edge/relation types
|-- graphkge/                     # Multimodal GraphKGE experiments with graph encoders
|-- link_prediction_experiments/  # Main multimodal link prediction experiments
|-- log/                          # Saved experiment logs
`-- requirements.txt              # Python dependency list
```

## Folder Descriptions

### `database_operations/`

This folder contains scripts for interacting with the Neo4j graph database and exporting graph data into local files that can be used for training.

Main files:

- `config.py`: Neo4j connection configuration, including URI, username, password, and database name.
- `check_connection.py`: Checks whether Neo4j can be connected successfully.
- `count_nodes_edges.py`: Counts graph nodes, relationships, major node labels, and relationship types.
- `export_full_graph_sharded.py`: Exports the full Neo4j graph into sharded files. By default, it creates `full_graph_export/manifest.json` and `full_graph_export/chunks/`.
- `graphsql.py`: Utilities related to graph data loading or querying.
- `excute.py`: Database operation script.
- `wiki_text_crawler/`: Crawls long Wikipedia text based on node information from graph chunks.

Typical usage:

```bash
python database_operations/check_connection.py
python database_operations/count_nodes_edges.py
python database_operations/export_full_graph_sharded.py --output-dir full_graph_export
python database_operations/wiki_text_crawler/crawl_long_text_by_chunk.py
```

### `link_prediction_experiments/`

This is the main directory for link prediction experiments. It is used to train multimodal models that predict whether an edge exists between two graph nodes, or whether a candidate edge is plausible.

Main subfolders:

- `configs/`: Experiment configurations, such as BERT + ResNet, BERT + DINOv2, and ALBERT + InceptionResNetV2.
- `project/`: Core code for training, models, data loading, sampling, and evaluation.
- `scripts/`: Scripts for running experiments in batches.

Important modules in `project/`:

- `train.py`: Training entry point.
- `trainer.py`: Training loop, validation, checkpoint saving, and related logic.
- `config.py`: Loads and validates JSON configuration files.
- `metrics.py`: Evaluation metrics.
- `data/`: Sharded graph loading, neighbor sampling, text loading, image loading, and data splitting.
- `models/`: Text encoders, image encoders, GNNs, fusion modules, and link predictors.
- `utils/`: Logging, random seed setup, IO helpers, plotting, and other utilities.

Example:

```bash
cd link_prediction_experiments
python -m project.train --config configs/exp_bert_resnet50_pretrained.json
```

Common arguments:

- `--resume <checkpoint>`: Resume training from an existing checkpoint.
- `--build-index-only`: Build only the adjacency index, then exit.
- `--build-text-cache-only`: Build only the text cache, then exit.
- `--rebuild-index`: Force rebuilding the adjacency index.
- `--rebuild-split`: Force rebuilding the train/validation/test split.
- `--sample-chunks N`: Use only the first N graph chunks. This is useful for debugging.

### `edge_type_kge/`

This folder is used for multimodal KGE experiments related to edge types or relation types. Its structure is similar to `graphkge/`, but it focuses more on relation modeling and KGE scoring.

Main subfolders:

- `configs/`: Experiment configurations for TransE, ComplEx, and different text/image encoder combinations.
- `project/`: Training entry point, data loading, model definitions, scoring functions, evaluation, and utility code.

Main capabilities:

- Supports KGE scorers such as `TransE` and `ComplEx`.
- Supports text features, image features, and fused multimodal features.
- Supports relation-stratified train/validation/test splitting.
- Supports node text caches and image caches to reduce repeated encoding cost.

Example:

```bash
cd edge_type_kge
python -m project.train --config configs/transe_default.json
python -m project.train --config configs/complex_default.json
```

### `graphkge/`

This folder is used for multimodal GraphKGE experiments with graph structure encoders. Compared with `edge_type_kge/`, it places more emphasis on modeling neighborhood structure with graph neural networks.

Main subfolders:

- `configs/`: Experiment configurations for GraphSAGE, R-GCN, TransE, ComplEx, and different multimodal encoder combinations.
- `project/`: GraphKGE training, graph batching, data sampling, model code, and evaluation code.

Main capabilities:

- Supports fusion of text, image, and graph-structure information.
- Supports graph encoders such as GraphSAGE and R-GCN.
- Supports KGE scoring functions such as TransE and ComplEx.
- Supports neighbor sampling, graph batching, relation-stratified splitting, and caching.

Example:

```bash
cd graphkge
python -m project.train --config configs/transe_default.json
python -m project.train --config configs/complex_default.json
```

Common arguments:

- `--sample-chunks N`: Use only the first N chunks for quick testing.
- `--resume <checkpoint>`: Resume training from a checkpoint.
- `--force-rebuild-data`: Force rebuilding graph data caches.
- `--rebuild-split`: Force rebuilding the data split.
- `--prebuild-text-cache`: Compatibility option. The current code automatically prebuilds raw feature caches.

### `log/`

This folder stores logs from previous experiment runs. These logs are useful for checking training progress, error messages, and final results.

Main subfolders:

- `link_prediction_log/`: Logs for link prediction experiments.
- `pure_kge_log/`: Logs for pure KGE or non-graph-enhanced experiments.
- `graphkgelog/`: Logs for GraphKGE experiments.

Log filenames usually include the model combination, for example:

- `exp_bert_dinov2.log`
- `transe_roberta_resnet50_frozen.log`
- `complex_albert_inception_resnet_v2_rgcn_frozen.log`

## Environment Setup

The project uses Python and installs dependencies from `requirements.txt`.
Python 3.9 or newer is recommended. If you plan to run training on GPU, install
a CUDA-compatible PyTorch build that matches your local NVIDIA driver and CUDA
runtime.

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install dependencies

Install the default dependency list:

```bash
pip install -r requirements.txt
```

If you only work inside `link_prediction_experiments/`, you can also install
its local dependency list:

```bash
pip install -r link_prediction_experiments/requirements.txt
```

### 3. Check PyTorch and GPU availability

Run the following command to verify that PyTorch is installed correctly:

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

If `cuda: False` is printed but you expected GPU training, reinstall PyTorch
with the correct CUDA wheel for your machine before running large experiments.

### 4. Check Neo4j configuration

Database-related scripts need a running Neo4j instance. Update the connection
settings in `database_operations/config.py`, then verify the connection:

```bash
python database_operations/check_connection.py
```

### Dependency Groups

Dependencies are listed in `requirements.txt`. The main groups are:

- Neo4j database connection: `neo4j`
- Web requests and parsing: `requests`, `beautifulsoup4`, `lxml`
- Scientific computing: `numpy`, `scikit-learn`, `matplotlib`, `Pillow`
- Deep learning frameworks: `torch`, `torchvision`, `torch-geometric`
- Pretrained text and vision models: `transformers`, `timm`

## Data and Configuration

The `configs/*.json` files in each experiment directory usually define the following paths:

- `graph_dir`: Directory for sharded graph data.
- `text_dir`: Directory for node text data.
- `image_dir`: Directory for node image data.
- `cache_dir`: Cache directory.
- `runs_dir`: Training output directory.
- `local_model_root`: Local pretrained model directory.

Before running training, make sure the paths in the configuration file match the data locations on your machine. For quick checks, use a `debug_*.json` config or add the `--sample-chunks` argument.

## Outputs

After training starts, a new experiment directory is usually created under the configured `runs_dir`. The directory name typically includes a timestamp and the experiment name. Common outputs include:

- `config.json` or `config_resolved.json`: Configuration used for this run.
- `summary.json`: Final metric summary.
- Checkpoint files: Used for resuming training or later evaluation.
- Log output: Training stages, validation metrics, test metrics, and error messages.

## Notes

- `__pycache__/` folders are generated automatically by Python and do not affect the code logic.
- Large-scale training is recommended to run on GPU. Some configurations automatically detect CUDA.
- The first run may need to download or load pretrained text/image models. Preparing a local model directory in advance is recommended.
- If Neo4j connection fails, first check `database_operations/config.py`, environment variables, whether the database is running, and whether the Bolt port is available.
- If GPU memory is insufficient, reduce the batch size, image size, number of sampled neighbors, or use `--sample-chunks` for debugging first.
