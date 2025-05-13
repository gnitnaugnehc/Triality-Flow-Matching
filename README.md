# Octonion-Triality Flow Matching on CIFAR-10

**Structure**  
- `algebra.py` : octonion multiplication  
- `layers.py`  : octonion-valued Conv / Linear  
- `transforms.py` : triality automorphisms  
- `model.py`   : `OctonionFlowField` definition  
- `data.py`    : CIFAR-10 loader & preprocessing  
- `train.py`   : training loop & CLI

**Quickstart**

```bash
pip install -r requirements.txt
bash scripts/run_training.sh
