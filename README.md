## Penalty Method for Deep Bilevel Optimization

#### An efficient method for solving bilevel optimization problems appearing in the field of machine learning, specifically for data denoising by importance learning, few-shot learning and training-data poisoning.

This is the reporsitory for the experiments of the paper <b>"[Penalty Method for Inversion-Free Deep Bilevel Optimization](https://arxiv.org/pdf/1911.03432.pdf)"</b>
<hr>

### Abstract
Bilevel optimizations are at the center of several important machine learning problems such as hyperparameter tuning, data denoising, few-shot learning, data poisoning. Different from simultaneous or multi-objective optimization, obtaining the exact descent direction for continuous bilevel optimization requires computing the inverse of the hessian of the lower-level cost function, even for first order methods. In this paper, we propose a new method for solving bilevel optimization, using the penalty function, which avoids computing the inverse of the hessian. We prove convergence of the method under mild conditions and show that it computes the exact hypergradient asymptotically. Small space and time complexity of our method allows us to solve large-scale bilevel optimization problems involving deep neural networks with up to 3.8M upper-level and 1.4M lower-level variables. We present results of our method for data denoising on MNIST/CIFAR10/SVHN datasets, for few-shot learning on Omniglot/Mini-Imagenet datasets and for training-data poisoning on MNIST/Imagenet datasets. In all experiments, our method outperforms or is comparable to previously proposed methods both in terms of accuracy and run-time.
<hr>

### Structure of the Repository
The codes for each experiment are specified in different folders of this repository. After installing the necessary packages (as mentioned below) create a folder structure similar to this repository. The codes for different settings and datasets for a particular experiment are included in their folders along with their dependencies. For some of the codes the datasets used are from Keras and so will be automatically available once Keras is installed. For others separate links to the places where we obtained the dataset are included in the experiment description section below. Specific instruction for pre-processing the data are also included below along with the code needed to do that pre-processing. Since the same code can be used to test different experimental settings, we have identified the line numbers where you can make changes and run the codes to compare results in the Tables of the paper. 
<hr>

### Libraries and packages used
1. Tensorflow 1.15
2. Keras
3. Cleverhans
4. H5py
5. Scipy
6. Numpy
<hr>

### Experiments
<hr>
Below we provide the link to the files that need to be run for replicating the results of Penalty as reported in the paper. We have also included all the codes for running the ApproxGrad comparisons in the corresponding directories as well. 

### Synthetic examples
Run [test_synthetic.py](applications/synthetic_examples/test_synthetic.py) after changing the settings of the test (e.g., number of iterations, or the location to store results.)
<hr>

### Data denoising using importance learning experiments
#### MNIST Experiments:

##### Small Scale:
Run [test_bilevel_importance_learning_mnist.py](applications/data_denoising/mnist_experiments/small_scale/test_bilevel_importance_learning_mnist.py)  with appropriate noise level specified on line 62.

##### Large Scale:   
Run [test_bilevel_importance_learning_mnist.py](applications/data_denoising/mnist_experiments/large_scale/Penalty/test_bilevel_importance_learning_mnist.py) with appropriate noise level specified on line 64. 

#### CIFAR10 Experiments:	
Run [test_bilevel_importance_learning_cifar10.py](applications/data_denoising/cifar10_experiments/Penalty/test_bilevel_importance_learning_cifar10.py) with appropriate noise level specified on line 70. 

#### SVHN Experiments:
Obtain data from [here](http://ufldl.stanford.edu/housenumbers/)
	
Split data into 72257 digits for training, 1000 digits for validation, 26032 digits for testing using [pre_process_svhn_data.py](applications/data_denoising/svhn_experiments/Penalty/pre_process_svhn_data.py)
	
Run [test_bilevel_importance_learning_svhn.py](applications/data_denoising/svhn_experiments/Penalty/test_bilevel_importance_learning_svhn.py) with appropriate noise level specified on line 62. 
<hr>

### Few-shot learning experiments
Obtain Omniglot and Mini-Imagenet datasets from the [Github](https://github.com/renmengye/few-shot-ssl-public) page of the paper Meta-Learning for Semi-Supervised Few-Shot Classification 

#### Omniglot Experiments:
Run [test_bilevel_few_shot_learning_omniglot.py](applications/few_shot_learning/omniglot_experiments/Penalty/test_bilevel_few_shot_learning_omniglot.py) by setting N and K on lines 59 & 60 for N-way K-shot classification 

#### Mini-Imagenet Experiments:
Run [test_bilevel_few_shot_learning_miniimagenet.py](applications/few_shot_learning/mini-imagenet_experiments/Penalty/test_bilevel_few_shot_learning_miniimagenet.py) by setting N and K on lines 37 & 38 for N-way K-shot classification 
<hr>

### Data poisoning experiments

#### Data Augmentation Attacks

##### Untargeted attack:	
Run [test_bilevel_poisoning_untargeted.py](applications/data_poisoning/data_augmentation_attacks/untargeted_attacks/Penalty/test_bilevel_poisoning_untargeted.py) by specifying number of poisoned points to add on line 15
		
##### Targeted attack:
Run [test_bilevel_poisoning_targeted.py](applications/data_poisoning/data_augmentation_attacks/targeted_attacks/Penalty/test_bilevel_poisoning_targeted.py) by specifying number of poisoned points to add on line 36
	
#### Clean label attacks
Download the dogfish dataset from [here](https://worksheets.codalab.org/bundles/0x550cd344825049bdbb865b887381823c/) and store them in dogfish_dataset named directory

Run [extract_inception_features.py](applications/data_poisoning/clean_label_attacks/dogfish_dataset/extract_inception_features.py) to extract 2048 dimensional features for all the images 

Run [test_bilevel_clean_label_attack.py](applications/data_poisoning/clean_label_attacks/test_bilevel_clean_label_attack.py) from outside dogfish_dataset 
<hr>

#### Citing
If you use this package please cite
<pre>
<code>
@misc{mehra2019penalty,
    title={Penalty Method for Inversion-Free Deep Bilevel Optimization},
    author={Akshay Mehra and Jihun Hamm},
    year={2019},
    eprint={1911.03432},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
</code>
</pre>
