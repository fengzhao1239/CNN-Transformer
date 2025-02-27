# A Hybrid CNN-Transformer Surrogate Model for the Multi-Objective Robust Optimization of Geological Carbon Sequestration

## Background
The optimization of well controls over time constitutes an essential step in the design of cost-effective and safe **geological carbon sequestration (GCS)** projects. However, the computational expense of these optimization problems, due to the extensive number of simulation evaluations, presents significant challenges for real-time decision-making.  

## Neural Network
In this paper, we propose a hybrid *CNN-Transformer* surrogate model to accelerate the well control optimization in GCS applications. The surrogate model encompasses a Convolution Neural Network (CNN) encoder to compress high-dimensional geological parameters, a Transformer processor to learn global patterns inherent in the well controls over time, and a CNN decoder to map the latent variables to the target solution variables.   
![Neural Network](https://github.com/fengzhao1239/CNN-Transformer/blob/main/assets/neural%20network.jpg)

## Numerical experiment
The surrogate model is trained to predict the spatiotemporal evolution of CO2 saturation and pressure within 3D heterogeneous permeability fields under dynamic CO2 injection rates. Results demonstrate that the surrogate model exhibits satisfactory performance in the context of prediction accuracy, computation efficiency, data scalability, and out-of-distribution generalizability.   
![Numerical Model](https://github.com/fengzhao1239/CNN-Transformer/blob/main/assets/numerical%20model.jpg)  

![Testset Performance](https://github.com/fengzhao1239/CNN-Transformer/blob/main/assets/last%20time%20step.jpg)

## Optimization
The surrogate model is further integrated with Multi-Objective Robust Optimization (MORO). Pareto optimal well controls are determined based on Non-dominated Sorting-based Genetic Algorithm II (NSGA-II), which maximize the storage efficiency and minimize the induced over-pressurization across an ensemble of uncertain geological realizations. The surrogate-based MORO reduces computational time by 99.99% compared to simulation-based optimization. The proposed workflow not only highlights the feasibility of applying the CNN-Transformer model for complex subsurface flow systems but also provides a practical solution for real-time decision-making in GCS projects.  
![Workflow](https://github.com/fengzhao1239/CNN-Transformer/blob/main/assets/workflow.jpg)

![Pareto](https://github.com/fengzhao1239/CNN-Transformer/blob/main/assets/pareto.jpg)  

## Directories in this repo:

- Model: the CNN-Transformer neural network
- TensorProcessing: the Dataset
- Train: the training script
- Optimization: the NSGA-2 optimization script
- checkpoints: the trained models
- dataset: dataset_link.txt


Refer to published paper on *Advances in Water Resources*:  https://doi.org/10.1016/j.advwatres.2025.104897
