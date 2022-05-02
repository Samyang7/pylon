#!/usr/bin/env python
# coding: utf-8

# # Distributed Training of Mask-RCNN in Amazon SageMaker using S3
# 
# This notebook is a step-by-step tutorial on distributed training of [Mask R-CNN](https://arxiv.org/abs/1703.06870) implemented in [TensorFlow](https://www.tensorflow.org/) framework. Mask R-CNN is also referred to as heavy weight object detection model and it is part of [MLPerf](https://www.mlperf.org/training-results-0-6/).
# 
# Concretely, we will describe the steps for training [TensorPack Faster-RCNN/Mask-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) and [AWS Samples Mask R-CNN](https://github.com/aws-samples/mask-rcnn-tensorflow) in [Amazon SageMaker](https://aws.amazon.com/sagemaker/) using [Amazon S3](https://aws.amazon.com/s3/) as data source.
# 
# The outline of steps is as follows:
# 
# 1. Stage COCO 2017 dataset in [Amazon S3](https://aws.amazon.com/s3/)
# 2. Build SageMaker training image and push it to [Amazon ECR](https://aws.amazon.com/ecr/)
# 3. Configure data input channels
# 4. Configure hyper-prarameters
# 5. Define training metrics
# 6. Define training job and start training
# 
# Before we get started, let us initialize two python variables ```aws_region``` and ```s3_bucket``` that we will use throughout the notebook. The ```s3_bucket``` must be located in the region of this notebook instance.

# In[25]:


import boto3

session = boto3.session.Session()
aws_region = session.region_name
s3_bucket  = 'mask-cnn-sam-3'


try:
    s3_client = boto3.client('s3')
    response = s3_client.get_bucket_location(Bucket=s3_bucket)
    print(f"Bucket region: {response['LocationConstraint']}")
except:
    print(f"Access Error: Check if '{s3_bucket}' S3 bucket is in '{aws_region}' region")


# ## Stage COCO 2017 dataset in Amazon S3
# 
# We use [COCO 2017 dataset](http://cocodataset.org/#home) for training. We download COCO 2017 training and validation dataset to this notebook instance, extract the files from the dataset archives, and upload the extracted files to your Amazon [S3 bucket](https://docs.aws.amazon.com/en_pv/AmazonS3/latest/gsg/CreatingABucket.html) with the prefix ```mask-rcnn/sagemaker/input/train```. The ```prepare-s3-bucket.sh``` script executes this step.
# 

# In[2]:


get_ipython().system('cat ./prepare-s3-bucket.sh')


#  Using your *Amazon S3 bucket* as argument, run the cell below. If you have already uploaded COCO 2017 dataset to your Amazon S3 bucket *in this AWS region*, you may skip this step. The expected time to execute this step is 20 minutes.

# In[3]:


get_ipython().run_cell_magic('time', '', "!./prepare-s3-bucket.sh 'mask-cnn-sam-3'")


# ## Build and push SageMaker training images
# 
# For this step, the [IAM Role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) attached to this notebook instance needs full access to Amazon ECR service. If you created this notebook instance using the ```./stack-sm.sh``` script in this repository, the IAM Role attached to this notebook instance is already setup with full access to ECR service. 
# 
# Below, we have a choice of two different implementations:
# 
# 1. [TensorPack Faster-RCNN/Mask-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) implementation supports a maximum per-GPU batch size of 1, and does not support mixed precision. It can be used with mainstream TensorFlow releases.
# 
# 2. [AWS Samples Mask R-CNN](https://github.com/aws-samples/mask-rcnn-tensorflow) is an optimized implementation that supports a maximum batch size of 4 and supports mixed precision. This implementation uses custom TensorFlow ops. The required custom TensorFlow ops are available in [AWS Deep Learning Container](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) images in ```tensorflow-training``` repository with image tag ```1.15.2-gpu-py36-cu100-ubuntu18.04```, or later.
# 
# It is recommended that you build and push both SageMaker training images and use either image for training later.
# 

# ### TensorPack Faster-RCNN/Mask-RCNN
# 
# Use ```./container-script-mode/build_tools/build_and_push.sh``` script to build and push the TensorPack Faster-RCNN/Mask-RCNN  training image to Amazon ECR. 

# In[4]:


get_ipython().system('cat ./container-script-mode/build_tools/build_and_push.sh')


# Using your *AWS region* as argument, run the cell below.

# In[5]:


get_ipython().run_cell_magic('time', '', '! ./container-script-mode/build_tools/build_and_push.sh {aws_region}')


# Set ```tensorpack_image``` below to Amazon ECR URI of the image you pushed above.

# In[28]:


tensorpack_image = '122783958501.dkr.ecr.us-east-1.amazonaws.com/mask-rcnn-tensorpack-sagemaker-script-mode:tf1.15-tpdb541e8'


# ### AWS Samples Mask R-CNN
# Use ```./container-optimized-script-mode/build_tools/build_and_push.sh``` script to build and push the AWS Samples Mask R-CNN training image to Amazon ECR.

# In[7]:


get_ipython().system('cat ./container-optimized-script-mode/build_tools/build_and_push.sh')


# Using your *AWS region* as argument, run the cell below.

# In[8]:


get_ipython().run_cell_magic('time', '', '! ./container-optimized-script-mode/build_tools/build_and_push.sh {aws_region}')


#  Set ```aws_samples_image``` below to Amazon ECR URI of the image you pushed above.

# In[27]:


aws_samples_image = '122783958501.dkr.ecr.us-east-1.amazonaws.com/mask-rcnn-tensorflow-sagemaker-script-mode:tf1.15-99dda64'


# ## SageMaker Initialization 
# 
# First we upgrade SageMaker to 2.3.0 API. If your notebook is already using latest Sagemaker 2.x API, you may skip the next cell.

# In[29]:


get_ipython().system(' pip install --upgrade pip')
get_ipython().system(' pip install sagemaker')


# We have staged the data and we have built and pushed the training docker image to Amazon ECR. Now we are ready to start using Amazon SageMaker.
# 

# In[30]:


get_ipython().run_cell_magic('time', '', 'import sagemaker\nfrom sagemaker import get_execution_role\nfrom sagemaker.tensorflow.estimator import TensorFlow\n\nrole = (\n    get_execution_role()\n)  # provide a pre-existing role ARN as an alternative to creating a new role\nprint(f"SageMaker Execution Role:{role}")\n\nclient = boto3.client("sts")\naccount = client.get_caller_identity()["Account"]\nprint(f"AWS account:{account}")')


# Next, we set ```training_image``` to the Amazon ECR image URI you saved in a previous step. 

# In[31]:


training_image = aws_samples_image 
print(f'Training image: {training_image}')


# ## Define SageMaker Data Channels
# In this step, we define SageMaker *train* data channel. 

# In[33]:


from sagemaker.inputs import TrainingInput

prefix = "mask-rcnn/sagemaker"  # prefix in your S3 bucket

s3train = f"s3://{s3_bucket}/{prefix}/input/train"
train_input = TrainingInput(
    s3_data=s3train, distribution="FullyReplicated", s3_data_type="S3Prefix", input_mode="File"
)

data_channels = {"train": train_input}


# Next, we define the model output location in S3 bucket.

# In[34]:


s3_output_location = f"s3://{s3_bucket}/{prefix}/output"


# ## Configure Hyper-parameters
# Next, we define the hyper-parameters. 
# 
# Note, some hyper-parameters are different between the two implementations. The batch size per GPU in TensorPack Faster-RCNN/Mask-RCNN is fixed at 1, but is configurable in AWS Samples Mask-RCNN. The learning rate schedule is specified in units of steps in TensorPack Faster-RCNN/Mask-RCNN, but in epochs in AWS Samples Mask-RCNN.
# 
# The detault learning rate schedule values shown below correspond to training for a total of 24 epochs, at 120,000 images per epoch.
# 
# <table align='left'>
#     <caption>TensorPack Faster-RCNN/Mask-RCNN  Hyper-parameters</caption>
#     <tr>
#     <th style="text-align:center">Hyper-parameter</th>
#     <th style="text-align:center">Description</th>
#     <th style="text-align:center">Default</th>
#     </tr>
#     <tr>
#         <td style="text-align:center">mode_fpn</td>
#         <td style="text-align:left">Flag to indicate use of Feature Pyramid Network (FPN) in the Mask R-CNN model backbone</td>
#         <td style="text-align:center">"True"</td>
#     </tr>
#      <tr>
#         <td style="text-align:center">mode_mask</td>
#         <td style="text-align:left">A value of "False" means Faster-RCNN model, "True" means Mask R-CNN moodel</td>
#         <td style="text-align:center">"True"</td>
#     </tr>
#      <tr>
#         <td style="text-align:center">eval_period</td>
#         <td style="text-align:left">Number of epochs period for evaluation during training</td>
#         <td style="text-align:center">1</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">lr_schedule</td>
#         <td style="text-align:left">Learning rate schedule in training steps</td>
#         <td style="text-align:center">'[240000, 320000, 360000]'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">batch_norm</td>
#         <td style="text-align:left">Batch normalization option ('FreezeBN', 'SyncBN', 'GN', 'None') </td>
#         <td style="text-align:center">'FreezeBN'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">images_per_epoch</td>
#         <td style="text-align:left">Images per epoch </td>
#         <td style="text-align:center">120000</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">data_train</td>
#         <td style="text-align:left">Training data under data directory</td>
#         <td style="text-align:center">'coco_train2017'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">data_val</td>
#         <td style="text-align:left">Validation data under data directory</td>
#         <td style="text-align:center">'coco_val2017'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">resnet_arch</td>
#         <td style="text-align:left">Must be 'resnet50' or 'resnet101'</td>
#         <td style="text-align:center">'resnet50'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">backbone_weights</td>
#         <td style="text-align:left">ResNet backbone weights</td>
#         <td style="text-align:center">'ImageNet-R50-AlignPadding.npz'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">load_model</td>
#         <td style="text-align:left">Pre-trained model to load</td>
#         <td style="text-align:center"></td>
#     </tr>
#     <tr>
#         <td style="text-align:center">config:</td>
#         <td style="text-align:left">Any hyperparamter prefixed with <b>config:</b> is set as a model config parameter</td>
#         <td style="text-align:center"></td>
#     </tr>
# </table>
# 
#     
# <table align='left'>
#     <caption>AWS Samples Mask-RCNN  Hyper-parameters</caption>
#     <tr>
#     <th style="text-align:center">Hyper-parameter</th>
#     <th style="text-align:center">Description</th>
#     <th style="text-align:center">Default</th>
#     </tr>
#     <tr>
#         <td style="text-align:center">mode_fpn</td>
#         <td style="text-align:left">Flag to indicate use of Feature Pyramid Network (FPN) in the Mask R-CNN model backbone</td>
#         <td style="text-align:center">"True"</td>
#     </tr>
#      <tr>
#         <td style="text-align:center">mode_mask</td>
#         <td style="text-align:left">A value of "False" means Faster-RCNN model, "True" means Mask R-CNN moodel</td>
#         <td style="text-align:center">"True"</td>
#     </tr>
#      <tr>
#         <td style="text-align:center">eval_period</td>
#         <td style="text-align:left">Number of epochs period for evaluation during training</td>
#         <td style="text-align:center">1</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">lr_epoch_schedule</td>
#         <td style="text-align:left">Learning rate schedule in epochs</td>
#         <td style="text-align:center">'[(16, 0.1), (20, 0.01), (24, None)]'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">batch_size_per_gpu</td>
#         <td style="text-align:left">Batch size per gpu ( Minimum 1, Maximum 4)</td>
#         <td style="text-align:center">4</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">batch_norm</td>
#         <td style="text-align:left">Batch normalization option ('FreezeBN', 'SyncBN', 'GN', 'None') </td>
#         <td style="text-align:center">'FreezeBN'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">images_per_epoch</td>
#         <td style="text-align:left">Images per epoch </td>
#         <td style="text-align:center">120000</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">data_train</td>
#         <td style="text-align:left">Training data under data directory</td>
#         <td style="text-align:center">'train2017'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">data_val</td>
#         <td style="text-align:left">Validation data under data directory</td>
#         <td style="text-align:center">'val2017'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">resnet_arch</td>
#         <td style="text-align:left">Must be 'resnet50' or 'resnet101'</td>
#         <td style="text-align:center">'resnet50'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">backbone_weights</td>
#         <td style="text-align:left">ResNet backbone weights</td>
#         <td style="text-align:center">'ImageNet-R50-AlignPadding.npz'</td>
#     </tr>
#     <tr>
#         <td style="text-align:center">load_model</td>
#         <td style="text-align:left">Pre-trained model to load</td>
#         <td style="text-align:center"></td>
#     </tr>
#     <tr>
#         <td style="text-align:center">config:</td>
#         <td style="text-align:left">Any hyperparamter prefixed with <b>config:</b> is set as a model config parameter</td>
#         <td style="text-align:center"></td>
#     </tr>
# </table>

# In[35]:


hyperparameters = {
    "mode_fpn": "True",
    "mode_mask": "True",
    "eval_period": 1,
    "batch_norm": "FreezeBN",
}


# ## Define Training Metrics
# Next, we define the regular expressions that SageMaker uses to extract algorithm metrics from training logs and send them to [AWS CloudWatch metrics](https://docs.aws.amazon.com/en_pv/AmazonCloudWatch/latest/monitoring/working_with_metrics.html). These algorithm metrics are visualized in SageMaker console.

# In[36]:


metric_definitions = [
    {"Name": "fastrcnn_losses/box_loss", "Regex": ".*fastrcnn_losses/box_loss:\\s*(\\S+).*"},
    {"Name": "fastrcnn_losses/label_loss", "Regex": ".*fastrcnn_losses/label_loss:\\s*(\\S+).*"},
    {
        "Name": "fastrcnn_losses/label_metrics/accuracy",
        "Regex": ".*fastrcnn_losses/label_metrics/accuracy:\\s*(\\S+).*",
    },
    {
        "Name": "fastrcnn_losses/label_metrics/false_negative",
        "Regex": ".*fastrcnn_losses/label_metrics/false_negative:\\s*(\\S+).*",
    },
    {
        "Name": "fastrcnn_losses/label_metrics/fg_accuracy",
        "Regex": ".*fastrcnn_losses/label_metrics/fg_accuracy:\\s*(\\S+).*",
    },
    {
        "Name": "fastrcnn_losses/num_fg_label",
        "Regex": ".*fastrcnn_losses/num_fg_label:\\s*(\\S+).*",
    },
    {"Name": "maskrcnn_loss/accuracy", "Regex": ".*maskrcnn_loss/accuracy:\\s*(\\S+).*"},
    {
        "Name": "maskrcnn_loss/fg_pixel_ratio",
        "Regex": ".*maskrcnn_loss/fg_pixel_ratio:\\s*(\\S+).*",
    },
    {"Name": "maskrcnn_loss/maskrcnn_loss", "Regex": ".*maskrcnn_loss/maskrcnn_loss:\\s*(\\S+).*"},
    {"Name": "maskrcnn_loss/pos_accuracy", "Regex": ".*maskrcnn_loss/pos_accuracy:\\s*(\\S+).*"},
    {"Name": "mAP(bbox)/IoU=0.5", "Regex": ".*mAP\\(bbox\\)/IoU=0\\.5:\\s*(\\S+).*"},
    {"Name": "mAP(bbox)/IoU=0.5:0.95", "Regex": ".*mAP\\(bbox\\)/IoU=0\\.5:0\\.95:\\s*(\\S+).*"},
    {"Name": "mAP(bbox)/IoU=0.75", "Regex": ".*mAP\\(bbox\\)/IoU=0\\.75:\\s*(\\S+).*"},
    {"Name": "mAP(bbox)/large", "Regex": ".*mAP\\(bbox\\)/large:\\s*(\\S+).*"},
    {"Name": "mAP(bbox)/medium", "Regex": ".*mAP\\(bbox\\)/medium:\\s*(\\S+).*"},
    {"Name": "mAP(bbox)/small", "Regex": ".*mAP\\(bbox\\)/small:\\s*(\\S+).*"},
    {"Name": "mAP(segm)/IoU=0.5", "Regex": ".*mAP\\(segm\\)/IoU=0\\.5:\\s*(\\S+).*"},
    {"Name": "mAP(segm)/IoU=0.5:0.95", "Regex": ".*mAP\\(segm\\)/IoU=0\\.5:0\\.95:\\s*(\\S+).*"},
    {"Name": "mAP(segm)/IoU=0.75", "Regex": ".*mAP\\(segm\\)/IoU=0\\.75:\\s*(\\S+).*"},
    {"Name": "mAP(segm)/large", "Regex": ".*mAP\\(segm\\)/large:\\s*(\\S+).*"},
    {"Name": "mAP(segm)/medium", "Regex": ".*mAP\\(segm\\)/medium:\\s*(\\S+).*"},
    {"Name": "mAP(segm)/small", "Regex": ".*mAP\\(segm\\)/small:\\s*(\\S+).*"},
]


# ## Define SageMaker Training Job
# 
# Next, we use SageMaker [Estimator](https://sagemaker.readthedocs.io/en/stable/estimators.html) API to define a SageMaker Training Job. 

# In[37]:


script = "aws-mask-rcnn.py"
print(script)


# ### Select distribution mode
# 
# We use Message Passing Interface (MPI) to distribute the training job across multiple hosts. The ```custom_mpi_options``` below is only used by [AWS Samples Mask R-CNN](https://github.com/aws-samples/mask-rcnn-tensorflow) model, and can be safely commented out for [TensorPack Faster-RCNN/Mask-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) model.

# In[38]:


mpi_distribution = {"mpi": {"enabled": True, "custom_mpi_options": "-x TENSORPACK_FP16=1 "}}


# ### Define SageMaker Tensorflow Estimator
# We recommned using 32 GPUs, so we set ```instance_count=4``` and ```instance_type='ml.p3.16xlarge'```, because there are 8 Tesla V100 GPUs per ```ml.p3.16xlarge``` instance. We recommend using 100 GB [Amazon EBS](https://aws.amazon.com/ebs/) storage volume with each training instance, so we set ```volume_size = 100```. 
# 
# We run the training job in your private VPC, so we need to set the ```subnets``` and ```security_group_ids``` prior to running the cell below. You may specify multiple subnet ids in the ```subnets``` list. The subnets included in the ```sunbets``` list must be part of the output of  ```./stack-sm.sh``` CloudFormation stack script used to create this notebook instance. Specify only one security group id in ```security_group_ids``` list. The security group id must be part of the output of  ```./stack-sm.sh``` script.
# 
# For ```instance_type``` below, you have the option to use ```ml.p3.16xlarge``` with 16 GB per-GPU memory and 25 Gbs network interconnectivity, or ```ml.p3dn.24xlarge``` with 32 GB per-GPU memory and 100 Gbs network interconnectivity. The ```ml.p3dn.24xlarge``` instance type offers significantly better performance than ```ml.p3.16xlarge``` for Mask R-CNN distributed TensorFlow training.
# 
# We use MPI to distribute the training job across multiple hosts.

# In[39]:


# Give Amazon SageMaker Training Jobs Access to FileSystem Resources in Your Amazon VPC.
security_group_ids = ['sg-0361ef549bfeec802'] 
subnets = [ 'subnet-0cd529de27570c8b1']
sagemaker_session = sagemaker.session.Session(boto_session=session)

mask_rcnn_estimator = TensorFlow(image_uri=training_image,
                                role=role, 
                                py_version='py3',
                                instance_count=4, 
                                instance_type='ml.p3.16xlarge',
                                distribution=mpi_distribution,
                                entry_point=script,
                                volume_size = 100,
                                max_run = 400000,
                                output_path=s3_output_location,
                                sagemaker_session=sagemaker_session, 
                                hyperparameters = hyperparameters,
                                metric_definitions = metric_definitions,
                                subnets=subnets,
                                security_group_ids=security_group_ids)


# Finally, we launch the SageMaker training job. See ```Training Jobs``` in SageMaker console to monitor the training job. 

# In[40]:


import time

job_name = f"mask-rcnn-s3-script-mode{int(time.time())}"
print(f"Launching Training Job: {job_name}")

# set wait=True below if you want to print logs in cell output
mask_rcnn_estimator.fit(inputs=data_channels, job_name=job_name, logs="All", wait=False)


# In[ ]:




