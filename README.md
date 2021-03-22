# Wav2Vec2.0 pretraining docker

pretraining wav2vec docker for sagemaker.
This docker is written with the assumption that it will be run by aws sagemaker.

# Required resources

Unlabeled data (audios without transcriptions) of your own language is required.
A good amount of unlabeled audios (e.g. 500 hours) will significantly reduce the amount of labeled data needed, and also boost up the model performance. Youtube/Podcast is a great place to collect the data for your own language. Prepare an s3 bucket with the audio data in it.

# Install instruction

## Wandb setup

1. Set WANDB_API_KEY in line 72 of `Dockerfile`.
2. And set wandb project name of `wandb_project` in `wav2vec2_base_librispeech.yaml`

## Upload docker to your ECS

Before upload docker, you have to setup aws cli.
Please check here: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html

After you install aws cli, run `aws configure`.
The region of the ecs where the docker will be uploaded must be the same region as the bucket where the dataset will be prepared.

You can upload docker to run `build_and_push.sh`.
The first parameter of the shell script is the docker image name.

```shell
sh build_and_push.sh wav2vec2-pretrain
```

## Define IAM role

```python
from sagemaker import get_execution_role

role = get_execution_role()
```

## Dataset

For example, we will have an s3 bucket with the following structure. There is no specification for naming the folders or wav files.

```txt
s3_backet
└── data
     ├── xxxx.wav
     ├── xxxx.wav
     ....
```

Define the path of the s3 bucket you prepared.

```python
data_location = 's3://{backetname}/{audio_foldername}'
```

## Create the session

```python
import sagemaker as sage
from time import gmtime, strftime

sess = sage.Session()
sess.default_bucket()
```

## Create an estimator and fit the model

```python
import boto3

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/wav2vec2-pretrain:latest'.format(account, region)

# https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html
model = sage.estimator.Estimator(image,
                       role, 1, 'ml.p3.16xlarge',
                       volume_size=1000,
                       output_path="s3://{}/output".format(sess.default_bucket()),
                       checkpoint_s3_uri="s3://{}/checkpoints".format(sess.default_bucket()),
                       checkpoint_local_path="/opt/ml/checkpoints/",
                       use_spot_instances=True,
                       max_run=320000,
                       max_wait=400000,
                       sagemaker_session=sess)
```

Run train!

```python
model.fit(data_location)
```

## Reference:

Paper: wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations: https://arxiv.org/abs/2006.11477 \
Source code: https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md

## License

MIT
