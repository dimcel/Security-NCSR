### Analysis and Comparison of Intrusion Detection Models

#### Datasets

##### KDD Cup 99 Dataset
The KDD Cup 99 dataset is a classic benchmark used in the evaluation of network intrusion detection systems. It includes 21 different types of attacks and has been widely utilized to test various machine learning models. For our experiments, we used the 10% subset of the KDD Cup 99 dataset due to computational constraints. This subset still provides a comprehensive challenge for intrusion detection systems. This dataset is available for download from the [UCI Machine Learning Repository](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).

##### IoT Intrusion Dataset
The IoT Intrusion dataset is tailored to address the unique security challenges in Internet of Things (IoT) environments. It consists of two primary labels indicating whether an attack is present or not. This dataset reflects the characteristics and vulnerabilities specific to IoT devices and networks. The dataset can be found on platforms such as Kaggle or through specific research publications related to IoT security.

#### Methods and Models

For both datasets, we applied Min-Max scaling to standardize the data. We also identified the most influential features that impacted the model's performance. Understanding feature importance helps in focusing on the key attributes that contribute to intrusion detection, enhancing the model's efficiency and effectiveness.

We selected the Random Forest algorithm due to its balance of accuracy and computational efficiency. In the context of intrusion detection, it is often more critical to minimize false negatives (i.e., undetected attacks) than to avoid false positives. Random Forest provides rapid inference, making it ideal for real-time applications. While deep learning models might offer slightly higher accuracy, their slower inference times and higher computational costs make them less suitable for live production systems compared to Random Forest.

#### Results and Comparison

The performance of our models compared to state-of-the-art methods is summarized in the table below:

| Paper                                                                                           | Dataset                              | Model                        | Accuracy  |
|------------------------------------------------------------------------------------------------|--------------------------------------|------------------------------|-----------|
| [Yaras and Dener (2024) - "Hybrid Deep Learning Model for IoT Intrusion Detection"](https://doi.org/10.3390/electronics13061053)  | IoT Intrusion (CICIoT2023)           | Hybrid Deep Learning         | 99.96%    |
| Our KDD Cup 99 Model                                                                           | KDD Cup 99 (10% subset)              | Random Forest                | 99.896%   |
| Our IoT Intrusion Model                                                                        | IoT Intrusion                        | Random Forest                | 99.82%    |

### Conclusion
Our experiments with the 10% subset of the KDD Cup 99 dataset and the IoT Intrusion dataset demonstrate the effectiveness of the Random Forest classifier in detecting network intrusions with high accuracy. The preprocessing steps, including Min-Max scaling and feature importance analysis, were crucial in optimizing the model's performance. The choice of Random Forest ensures rapid and reliable intrusion detection, suitable for real-time deployment in various environments. The state-of-the-art research further validates the robustness of our approach and provides a benchmark for future improvements.
