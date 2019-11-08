# Penalty Method for Deep Bilevel Optimization

In this reporsitory we present the code used to generate the results for all the experiments in the paper titled <br><b> "Penalty Method for Inversion-Free Deep Bilevel Optimization"</b>

## Abstract
Bilevel optimization appears in many important machine learning problems, including hyperparmeter tuning, data denoising,  meta-learning, and data poisoning. Different from simultaneous or multi-objective optimization, continuous bilevel optimization requires computing the inverse Hessian of the cost function to get the exact descent direction even for first-order methods. In this paper, we propose an inversion-free method using penalty function to efficiently solve large bilevel problems. We prove convergence of the penalty-based method under mild conditions and show that it computes the exact hypergradients asymptotically. Our method has smaller time and space complexity than forward- and reverse-mode differentiation, which allows us to solve large problems involving deep neural networks with up to 3.8M  upper-level and 1.4M lower-level variables. We apply our algorithm to denoising label noise with MNIST/CIFAR10/SVHN datasets, few-shot learning with Omniglot/Mini-ImageNet datasets, and training-data  poisoning with MNIST/ImageNet datasets. Our method outperforms or is comparable to the previous results in all of these tasks.

## Structure of the Repository
The codes for each experiment are specified in different folders of this repository. After installing the necessary packages (as mentioned below) create a folder structure similar to this repository. The codes for different settings and datasets for a particular experiment are included in their folders along with their dependencies. For some of the codes the datasets used are from Keras and so will be automatically available once Keras is installed. For others separate links to the places where we obtained the dataset are included in the experiment description section below. Specific instruction for pre-processing the data are also included below along with the code needed to do that pre-processing. Since the same code can be used to test different experimental settings, we have identified the line numbers where you can make changes and run the codes to compare results in the Tables of the paper. 

## Libraries and packages used
1. Tensorflow
2. Keras
3. Cleverhans
4. H5py
5. Scipy
6. Numpy

## Synthetic examples
Run [test_synthetic.py](synthetic_examples/test_synthetic.py) after changing the settings of the test (e.g., number of iterations, or the location to store results.)

## Data denoising using importance learning experiments
### MNIST Experiments:

#### Small Scale:
	
Obtain data using keras
	
Run [test_bilevel_importance_learning_mnist.py](data_denoising/mnist_experiments/small_scale/test_bilevel_importance_learning_mnist.py)  with appropriate noise level specified on line 60.

#### Large Scale:

Obtain data using keras
   
Run [test_bilevel_importance_learning_mnist.py](data_denoising/mnist_experiments/large_scale/Penalty/test_bilevel_importance_learning_mnist.py) with appropriate noise level specified on line 62. 

### CIFAR10 Experiments:

Obtain data using keras
	
Run [test_bilevel_importance_learning_cifar10.py](data_denoising/cifar10_experiments/Penalty/test_bilevel_importance_learning_cifar10.py) with appropriate noise level specified on line 68. 

### SVHN Experiments:

Obtain data from http://ufldl.stanford.edu/housenumbers/
	
Split data into 72257 digits for training, 1000 digits for validation, 26032 digits for testing using [pre_process_svhn_data.py](data_denoising/svhn_experiments/Penalty/pre_process_svhn_data.py)
	
Run [test_bilevel_importance_learning_svhn.py](data_denoising/svhn_experiments/Penalty/test_bilevel_importance_learning_svhn.py) with appropriate noise level specified on line 60. 

## Few-shot learning experiments

Obtain Omniglot and Mini-Imagenet datasets from the following Github page of the paper Meta-Learning for Semi-Supervised Few-Shot Classification https://github.com/renmengye/few-shot-ssl-public

### Omniglot Experiments:

Run [test_bilevel_few_shot_learning_omniglot.py] (few_shot_learning/omniglot_experiments/Penalty/test_bilevel_few_shot_learning_omniglot.py) by setting N and K on lines 59 & 60 for N-way K-shot classification 

### Mini-Imagenet Experiments:

Run [test_bilevel_few_shot_learning_miniimagenet.py](few_shot_learning/mini-imagenet_experiments/Penalty/test_bilevel_few_shot_learning_miniimagenet.py) by setting N and K on lines 37 & 38 for N-way K-shot classification 

## Data poisoning experiments

### Data Augmentation Attacks

#### Untargeted attack:

Obtain data using keras
	
Run [test_bilevel_poisoning_untargeted.py](data_poisoning/data_augmentation_attacks/untargeted_attacks/Penalty/test_bilevel_poisoning_untargeted.py) by specifying number of poisoned points to add on line 10
		
#### Targeted attack:

Obtain data using keras

Run [test_bilevel_poisoning_targeted.py](data_poisoning/data_augmentation_attacks/targeted_attacks/Penalty/test_bilevel_poisoning_targeted.py) by specifying number of poisoned points to add on line 10
	
### Clean label attacks

Download the dogfish dataset from https://worksheets.codalab.org/bundles/0x550cd344825049bdbb865b887381823c/ and store them in dogfish_dataset named directory

Run [extract_inception_features.py](data_poisoning/clean_label_attacks/extract_inception_features.py) in the directory dogfish_dataset to extract 2048 dimensional features for all the images 

Run [test_bilevel_clean_label_attack.py](data_poisoning/clean_label_attacks/test_bilevel_clean_label_attack.py) from outside dogfish_dataset 


### Citing
If you use this package please cite
<pre>
<code>
Add citation here
</code>
</pre>
