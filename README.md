# PPGFlowECG: Latent Rectified Flow with Cross-Modal Encoding for PPG-Guided ECG Generation and Cardiovascular Disease Detection

### Abstract: 
In clinical practice, electrocardiography (ECG) remains the gold standard for cardiac monitoring, providing crucial insights for diagnosing a wide range of cardiovascular diseases (CVDs). However, its reliance on specialized equipment and trained personnel limits feasibility for continuous routine monitoring. Photoplethysmography (PPG) offers accessible, continuous monitoring but lacks definitive electrophysiological information, preventing conclusive diagnosis. Generative models present a promising approach to translate PPG into clinically valuable ECG signals, yet current methods face substantial challenges, including the misalignment of physiological semantics in generative models and the complexity of modeling in high-dimensional signals. To this end, we propose PPGFlowECG, a two-stage framework that aligns PPG and ECG in a shared latent space via the CardioAlign Encoder and employs latent rectified flow to generate ECGs with high fidelity and interpretability. To the best of our knowledge, this is the first study to experiment on MCMED, a newly released clinical-grade dataset comprising over 10 million paired PPG–ECG samples with expert-labeled CVD annotations, demonstrating the effectiveness of our method for PPG-to-ECG translation and cardiovascular disease detection. Furthermore, cardiologist-led evaluations confirm that the synthesized ECGs achieve high fidelity and improve diagnostic reliability, underscoring our method’s potential for real-world cardiovascular screening.

<img src="./img/framework.png" width="800">

## ⚙️ PPGFlowECG Training
1. Build environment from requirements.txt
2. Modify the "train_dir" setting in config/cardioalign_encoder.yaml to point to your training dataset.
3. Load the model weights, or train the appropriate weights using your own dataset.
   ```sh
   https://pan.baidu.com/s/1XGbouLx1tS5t63YbRfpl6A?pwd=hu68
   ```
### Stage 1: CardioAlign-Encoder:
```sh
python model/cardioalign_encoder/train.py --config config/cardioalign_encoder.yaml --save_dir results/cardioalign_encoder
```
### Stage 2: Latent Rectified Flow:
```sh
python main.py --train --config_file config/latent_rectified_flow.yaml --output baseline
```

## 🚀 Quick Generation
```sh
python main.py --config_file config/latent_rectified_flow.yaml --output baseline
```
