# Penalty Method for Bilevel Optimization

In this reporsitory we present the code used to generate all the results in the paper titled "Penalty Method for Inversion-Free Deep Bilevel Optimization"

### Abstract
Bilevel optimization appears in many important machine learning problems, including hyperparmeter tuning, data denoising,  meta-learning, and data poisoning. Different from simultaneous or multi-objective optimization, continuous bilevel optimization requires computing the inverse Hessian of the cost function to get the exact descent direction even for first-order methods. In this paper, we propose an inversion-free method using penalty function to efficiently solve large bilevel problems. We prove convergence of the penalty-based method under mild conditions and show that it computes the exact hypergradients asymptotically. Our method has smaller time and space complexity than forward- and reverse-mode differentiation, which allows us to solve large problems involving deep neural networks with up to 3.8M  upper-level and 1.4M lower-level variables. We apply our algorithm to denoising label noise with MNIST/CIFAR10/SVHN datasets, few-shot learning with Omniglot/Mini-ImageNet datasets, and training-data  poisoning with MNIST/ImageNet datasets. Our method outperforms or is comparable to the previous results in all of these tasks.

### Structure of the Repository
The codes for each experiment are specified in different folders of this repository. After installing the necessary packages (as mentioned below) create a folder structure similar to this repository. The codes for different settings and datasets for a particular experiment are included in their folders along with their dependencies. For some of the codes the datasets used are from Keras and so will be automatically available once Keras is installed. For others separate links to the places where we obtained the dataset are included in the experiment description section below. Specific instruction for pre-processing the data are also included below along with the code needed to do that pre-processing. Since the same code can be used to test different experimental settings, we have identified the line numbers where you can make changes and run the codes to compare results in the Tables of the paper. 

### Libraries and packages used
1. Tensorflow
2. Keras
3. Cleverhans
4. H5py
5. Scipy
6. Numpy

### Synthetic examples
Run test_synthetic.py (located at synthetic_examples/ ) after changing the settings of the test (e.g., number of iterations, or the location to store results.)

### Data denoising using importance learning experiments
#### 1. MNIST Experiments:

##### a. Small Scale:
	
Obtain data using keras
	
Run test_bilevel_importance_learning_mnist.py (located at data_denoising/mnist_experiments/small_scale/)  with appropriate noise level specified on line 60.

##### b. Large Scale:

Obtain data using keras
   
Run test_bilevel_importance_learning_mnist.py (located at data_denoising/mnist_experiments/large_scale/Penalty/) with appropriate noise level specified on line 62. 

#### 2. CIFAR10 Experiments:

Obtain data using keras
	
Run test_bilevel_importance_learning_cifar10.py (located at data_denoising/cifar10_experiments/Penalty/) with appropriate noise level specified on line 68. 

#### 3. SVHN Experiments:

Obtain data from http://ufldl.stanford.edu/housenumbers/
	
Split data into 72257 digits for training, 1000 digits for validation, 26032 digits for testing using pre_process_svhn_data.py (located at data_denoising/svhn_experiments/Penalty/)
	
Run test_bilevel_importance_learning_svhn.py (located at data_denoising/svhn_experiments/Penalty/) with appropriate noise level specified on line 60. 

### Citing
If you use this package please cite
<pre>
<code>
Add citation here
</code>
</pre>
