# Differentiable Microscopy for Content and Task Aware Compressive Imaging

## Reproducing results

We encourage to use [conda package manager](https://docs.conda.io/en/latest/) for reproducing the results.

1. Clone the repository
    ```
    git clone https://github.com/qwertPaperSubmission/CompressiveDabbaMu.git
    cd CompressiveDabbaMu
    ```

2. Create environment and install packages
    ```
    conda create -n differentiableMicroscopy python=3.6
    source activate differentiableMicroscopy
    bash requirements.sh
    ```

3. Create PatchMNIST dataset

    ```
    python create_patchMNIST_dataset.py 
    ```

4. Run experiments: Train and Evaluate

    1. Content-aware experiments (Proposed method on x4096 compression)
        ```
        python run.py --data_dir "./datasets/mnistgrid_mnistsize(32)_imgsize(640)" --exp_type 'contentaware' --dataset_name mnistdigits_grid2patch
        ```
    2. Task-aware experiments (Proposed method on x4096 compression)
        ```
        python run.py --data_dir "./datasets/mnistgrid_mnistsize(32)_imgsize(640)" --exp_type 'segmentation' --dataset_name mnistdigits_grid2patch
        ```
        

* Change the parameters including upsampling block, Ht initialization type, Ht preprocessing type (eg: Frequency domain optimization), Noise parameters, PSFs, scale factor, number of excitation patterns, photon count, Learning rate for Ht training, etc
    1. Find the keyword for the parameter in defaults.py
    2. Add the keyword to the 'exps' dictionary inside the 'run.py'
    3. Give the value for the added key as a list of parameters that needed to be experimented
        eg: 'MODEL.MODEL_A.lambda_scale_factor': ['2', '4', '5]
    4. Run the 'run.py'
    
    
## Experiments on new datasets

1. Create get_dataset class (follow the examples in 'modules/data_utils.py')
2. Add function (with the prefered name) to obtain train, validation, test sets for the datasets (follow the examples in 'modules/datasets.py')
3. Add new dataset to 'exps' dictionary in 'run.py' using the key 'DATASET.name'
4. Run 'run.py'
