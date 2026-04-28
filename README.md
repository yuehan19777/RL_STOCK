# WaveGraph-RAC

**WaveGraph-RAC = Wavelet Graph Risk-Aware Controller**  
中文：**小波图风险感知控制模型**

运行顺序：

```bash
cd C:\Users\16209\Desktop\WaveGraph-RAC
pip install numpy pandas pyyaml matplotlib scikit-learn scipy torch PyWavelets joblib
python run_main/precompute_wavelet_graphs.py --config config.yaml
python run_main/train_signal.py --config config.yaml
python run_main/backtest_eval.py --config config.yaml --mode fixed
python run_main/train_ppo_controller.py --config config.yaml
python run_main/backtest_eval.py --config config.yaml --mode ppo
```

如果 `data.dataset_npz` 不存在，可以先运行：

```bash
python run_main/build_dataset_from_qlib.py --config config.yaml
```

输出包括：`outputs/signal_model.joblib`、`outputs/wavelet_graph_cache.npz`、`outputs/ppo_controller.pt`、`outputs/backtest/*/report.json`、净值曲线、回撤曲线、现金权重和参数曲线。
