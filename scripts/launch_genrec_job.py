#!/usr/bin/env python3
"""
SageMaker GenRec Job Launcher
Uploads source code to S3 and triggers GenRec ranker training on AWS GPU instance.
"""

import boto3
import time
import os
import sys
import tarfile

def upload_code(s3, bucket_name, job_name):
    prefix = f"code/{job_name}"
    s3.upload_file("scripts/train_genrec.py", bucket_name, f"{prefix}/train_genrec.py")
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                local_path = os.path.join(root, file)
                s3_key = f"{prefix}/{local_path}"
                s3.upload_file(local_path, bucket_name, s3_key)
    return f"s3://{bucket_name}/{prefix}/"

def main():
    profile = os.environ.get("AWS_PROFILE", "tutor-deploy")
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    use_spot = os.environ.get("USE_SPOT", "false").lower() == "true"
    model_name = os.environ.get("MODEL_NAME", "gpt2") # Can set to meta-llama/Llama-3.1-8B or Qwen/Qwen2.5-0.5B
    
    session = boto3.Session(profile_name=profile, region_name=region)
    sagemaker = session.client("sagemaker")
    cfn = session.client("cloudformation")
    s3 = session.client("s3")

    print("🔍 Fetching CDK Stack Outputs (S3 Bucket & SageMaker IAM Role)...")
    try:
        response = cfn.describe_stacks(StackName="GenrecStack")
        outputs = {item["OutputKey"]: item["OutputValue"] for item in response["Stacks"][0]["Outputs"]}
        bucket_name = outputs["DataBucketName"]
        role_arn = outputs["SageMakerRoleArn"]
    except Exception as e:
        print(f"❌ Error fetching GenrecStack outputs: {e}")
        print("Please ensure you have run 'npx cdk deploy' first!")
        sys.exit(1)

    print(f"✅ S3 Bucket: {bucket_name}")
    print(f"✅ SageMaker Role: {role_arn}")

    import re
    safe_name = re.sub(r'[^a-zA-Z0-9-]', '-', model_name.lower()).strip('-')
    job_name = f"genrec-train-{safe_name}-{int(time.time())}"

    # 1. Upload source code to S3 prefix
    code_s3_uri = upload_code(s3, bucket_name, job_name)
    print(f"📦 Uploaded source code -> {code_s3_uri}")

    # 2. Configure Job Parameters
    mode_str = "SPOT" if use_spot else "ON-DEMAND"
    resume_from_job = os.environ.get("RESUME_FROM_JOB", None)
    if resume_from_job:
        checkpoint_s3_uri = f"s3://{bucket_name}/checkpoints/{resume_from_job}/"
        print(f"🔄 Resuming from existing S3 checkpoint: {checkpoint_s3_uri}")
    else:
        checkpoint_s3_uri = f"s3://{bucket_name}/checkpoints/{job_name}/"
    output_s3_uri = f"s3://{bucket_name}/output/{job_name}/"

    instance_type = os.environ.get("INSTANCE_TYPE", "ml.g4dn.xlarge")

    # PyTorch 2.0 GPU Deep Learning Container
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2"

    print("\n" + "=" * 60)
    print(f"🚀 LAUNCHING SAGEMAKER GENREC TRAINING JOB ({mode_str})")
    print(f"🤖 Backbone Model: {model_name}")
    print(f"💰 Managed Spot Pricing: {use_spot}")
    print(f"💾 Checkpoint Location: {checkpoint_s3_uri}")
    print(f"💻 Instance Type: {instance_type}")
    print("=" * 60 + "\n")

    resource_config = {
        "InstanceType": instance_type,
        "InstanceCount": 1,
        "VolumeSizeInGB": 50
    }

    stopping_condition = {"MaxRuntimeInSeconds": 10800} # 3 hours max
    if use_spot:
        stopping_condition["MaxWaitTimeInSeconds"] = 7200

    job_kwargs = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "ContainerArguments": [
                "python3", "-u", "/opt/ml/input/data/code/train_genrec.py",
                "--model-name", model_name,
                "--epochs", "2",
                "--batch-size", "16"
            ]
        },
        "RoleArn": role_arn,
        "OutputDataConfig": {"S3OutputPath": output_s3_uri},
        "ResourceConfig": resource_config,
        "StoppingCondition": stopping_condition,
        "EnableManagedSpotTraining": use_spot,
        "CheckpointConfig": {
            "S3Uri": checkpoint_s3_uri,
            "LocalPath": "/opt/ml/checkpoints"
        },
        "InputDataConfig": [
            {
                "ChannelName": "code",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": code_s3_uri,
                        "S3DataDistributionType": "FullyReplicated"
                    }
                }
            }
        ],
        "Tags": [
            {"Key": "Project", "Value": "genrec"},
            {"Key": "Env", "Value": "dev"}
        ]
    }

    response = sagemaker.create_training_job(**job_kwargs)

    print(f"✅ Job submitted successfully! ARN: {response['TrainingJobArn']}")
    print("📋 Monitoring progress (Ctrl+C to stop monitoring, job will run safely on AWS)...\n")

    # 4. Stream Logs & Wait for Completion
    status = "InProgress"
    try:
        while status in ["InProgress", "Stopping"]:
            job_info = sagemaker.describe_training_job(TrainingJobName=job_name)
            status = job_info["TrainingJobStatus"]
            secondary_status = job_info.get("SecondaryStatus", "")
            print(f"STATUS: {status} | Details: {secondary_status}")

            if status in ["Completed", "Failed", "Stopped"]:
                break
            time.sleep(15)
    except KeyboardInterrupt:
        print("\n⚠️ Monitoring interrupted by user. The job continues running safely on SageMaker.")
        print(f"To check status later: aws sagemaker describe-training-job --training-job-name {job_name} --profile {profile}")
        return

    print("\n" + "=" * 60)
    if status == "Completed":
        billable_time = job_info.get("BillableTimeInSeconds", 0)
        training_time = job_info.get("TrainingTimeInSeconds", 0)
        print("🎉 GENREC TRAINING COMPLETED SUCCESSFULLY!")
        print(f"⏱️ Actual Training Duration: {training_time} seconds")
        print(f"💰 Billable Time: {billable_time} seconds")
        print(f"💵 Total Cost: ~${round(billable_time * 1.40 / 3600, 2)}")
    else:
        print(f"❌ Job ended with status: {status}")
        print(f"Failure Reason: {job_info.get('FailureReason', 'Unknown')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
