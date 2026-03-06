## LLM Long-Context Characterization Framework (ISPASS 2026)
State Space Model and SSM-Transformer Hybrid model characterization on consumer GPU and edge devices for very large context

* This repository contains the source code for the performance characterization framework introduced in the paper, *"Characterizing State Space Model and Hybrid Language Model Performance with Long Context"*. 

* The framework provides comprehensive benchmarking and profiling tools to evaluate Transformers, State Space Models (SSMs), and Hybrid models (such as Qwen2.5, Mamba-2, and Falcon-H1).



## Key Features

* **Computational Performance Tracking**: Measures end-to-end inference metrics across generation stages, including Time to First Token (TTFT), Time per Output Token (TPOT), and overall throughput.
* **Detailed Memory Analysis**: Captures system-level peak GPU memory usage reserved during inference, alongside fine-grained operator-level memory footprints.
* **Operator-Level Profiling**: Generates latency breakdowns to identify execution bottlenecks, separating GEMM, non-GEMM, and novel SSM-specific operators.
* **Energy Consumption Metrics**: Calculates energy usage over time based on power draw statistics, which is critical for edge deployment evaluation.

## Repository Structure

```
src/
├── __init__.py
├── profiling/                  # Core PyTorch profiler engine
│   ├── __init__.py
│   ├── eval.py                 # Operator-level profiler (TTFT, TPOT, energy, shapes)
│   └── power_logger.py         # nvidia-smi power log parser
├── models/                     # Model loaders and profiling entry points
│   ├── __init__.py
│   └── profile_runner.py       # LMProfile, MambaProfile, per-model CLI functions
├── memory/                     # Memory footprint analysis
│   ├── __init__.py
│   ├── mem_footprint.py        # Prefill / decode memory measurement (PyTorch)
│   └── vllm_oom.py             # vLLM OOM boundary sweep
└── visualization/              # Figure generation from profiling CSVs
    ├── __init__.py
    └── gen_figure_data.py      # Operator breakdown plots and summary CSVs
```

## Environment Setup



## Running the Profiling Framework

All scripts are run from within the `src/` directory:

```bash
# Operator-level profiling (prefill / TTFT)
cd src
python -m models.profile_runner --model_name mamba2 --batch_size 1 --seq_len 1024 --device cuda

# Memory footprint sweep
python -m memory.mem_footprint

# vLLM OOM boundary sweep
python -m memory.vllm_oom

# Figure generation from existing CSVs
python -m visualization.gen_figure_data
```


## Citation

If you use this framework in your research, please cite:

```bibtex
@article{mitra2025characterizing,
  title={Characterizing state space model (ssm) and ssm-transformer hybrid language model performance with long context length},
  author={Mitra, Saptarshi and Karami, Rachid and Xu, Haocheng and Huang, Sitao and Kwon, Hyoukjun},
  journal={arXiv preprint arXiv:2507.12442},
  year={2025}
}
