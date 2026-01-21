# PPGFlowECG: Latent Rectified Flow with Cross-Modal Encoding for PPG-Guided ECG Generation and Cardiovascular Disease Detection

### Abstract: 
Electrocardiography (ECG) is the clinical gold standard for cardiovascular disease (CVD) assessment, yet continuous monitoring is constrained by the need for dedicated hardware and trained personnel. Photoplethysmography (PPG) is ubiquitous in wearable devices and readily scalable, but it lacks electrophysiological specificity, limiting diagnostic reliability. While generative methods aim to translate PPG into clinically useful ECG signals, existing approaches are limited by the misalignment of physiological semantics in generative models and the complexity of modeling in high-dimensional signals. To address these limitations, we propose PPGFlowECG, a two-stage framework that aligns PPG and ECG in a shared latent space using the CardioAlign Encoder and then synthesizes ECGs with latent rectified flow. We further provide a formal analysis of this coupling, showing that the CardioAlign Encoder is necessary to guarantee stable and semantically consistent ECG synthesis under our formulation. Extensive experiments on four datasets demonstrate improved synthesis fidelity and downstream diagnostic utility. These results indicate that PPGFlowECG supports scalable, wearable-first CVD screening when standard ECG acquisition is unavailable.

## ⚙️ PPGFlowECG Training
1. Build environment from requirements.txt
2. Train the appropriate weights using your own dataset.

### Stage 1: CardioAlign-Encoder:
```sh
python model/cardioalign_encoder/train.py --config config/cardioalign_encoder.yaml --save_dir results/cardioalign_encoder
```
### Stage 2: Latent Rectified Flow:
```sh
python main.py --train --config_file config/latent_rectified_flow.yaml --output baseline
```

### Evaluation:
During the evaluation process, for calculating the FID score, we used ECGFounder as the feature extractor; the weights for ECGFounder can be downloaded from the following link:
```sh
https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main
```

## 🚀 Quick Generation
You can use our pre-trained model weights for fast PPG to ECG.

### Our Model Checkpoints:
```sh
https://pan.baidu.com/s/1XGbouLx1tS5t63YbRfpl6A?pwd=hu68
```
```sh
https://huggingface.co/XiaochengFang/PPGFlowECG/tree/main
```

### Generation:
```sh
python main.py --config_file config/latent_rectified_flow.yaml --output baseline
```

## 📝 Citation:
If you find our work interesting and helpful, please consider giving our repo a star. Additionally, if you would like to cite our work, please use the following format:
```sh
@article{fang2025ppgflowecg,
  title={PPGFlowECG: Latent Rectified Flow with Cross-Modal Encoding for PPG-Guided ECG Generation and Cardiovascular Disease Detection},
  author={Fang, Xiaocheng and Jin, Jiarui and Wang, Haoyu and Liu, Che and Cai, Jieyi and Nie, Guangkun and Li, Jun and Li, Hongyan and Hong, Shenda},
  journal={arXiv preprint arXiv:2509.19774},
  year={2025}
}
```
