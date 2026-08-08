# fed-har — Federated Human Activity Recognition

Federated learning on the UCI HAR dataset. Many simulated phones train one
shared activity-recognition model without pooling their raw sensor data.

## What this shows
- A centralised baseline (upper bound: one model sees all data).
- FedAvg implemented from scratch (McMahan et al., 2017).
- FedProx for the harder, realistic by-person data split (Li et al., 2020).
- The accuracy cost of keeping data private, and how FedProx claws some back.

## Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
## Status
Work in progress.

## References
Anguita, D., Ghio, A., Oneto, L., Parra, X. and Reyes-Ortiz, J.L. (2013) 'A public domain dataset for human activity recognition using smartphones', *ESANN 2013*.
Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A. and Smith, V. (2020) 'Federated optimization in heterogeneous networks', *Proceedings of Machine Learning and Systems*, 2, pp. 429–450.
McMahan, H.B., Moore, E., Ramage, D., Hampson, S. and Arcas, B.A. (2017) 'Communication-efficient learning of deep networks from decentralized data', *AISTATS*, pp. 1273–1282.
