#!/usr/bin/env python3
import boto3
import time
import os
import sys

def main():
    profile = os.environ.get("AWS_PROFILE", "tutor-deploy")
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    use_spot = os.environ.get("USE_SPOT", "false").lower() == "true"
    
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

    # 1. Upload training script to S3
    script_s3_key = "scripts/train_test.py"
    print(f"📤 Uploading local scripts/train_test.py -> s3://{bucket_name}/{script_s3_key}...")
    s3.upload_file("scripts/train_test.py", bucket_name, script_s3_key)

    # 2. Configure Job Parameters
    mode_str = "SPOT" if use_spot else "ON-DEMAND"
    job_name = f"genrec-test-gpu-{mode_str.lower()}-{int(time.time())}"
    checkpoint_s3_uri = f"s3://{bucket_name}/checkpoints/{job_name}/"
    output_s3_uri = f"s3://{bucket_name}/output/{job_name}/"

    # PyTorch 2.0 GPU Deep Learning Container in ap-south-1
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2"

    print("\n" + "=" * 60)
    print(f"🚀 LAUNCHING SAGEMAKER GPU TRAINING TEST JOB ({mode_str}): {job_name}")
    print(f"💰 Managed Spot Pricing: {use_spot}")
    print(f"💾 Checkpoint S3 Location: {checkpoint_s3_uri}")
    print(f"⏱️ Hard Timeout Limit: 600 seconds (10 mins)")
    print(f"💻 Instance Type: ml.g5.2xlarge (1x NVIDIA A10G GPU 24GB VRAM)")
    print(f"💵 Est. Cost (~3 mins): ~${'0.05' if use_spot else '0.07'}")
    print("=" * 60 + "\n")

    resource_config = {
        "InstanceType": "ml.g5.2xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 30
    }

    stopping_condition = {"MaxRuntimeInSeconds": 600}
    if use_spot:
        stopping_condition["MaxWaitTimeInSeconds"] = 1200

    job_kwargs = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "ContainerArguments": ["python3", "-u", "/opt/ml/input/data/code/train_test.py"]
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
                        "S3Uri": f"s3://{bucket_name}/{script_s3_key}",
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
    print("📋 Monitoring progress...\n")

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
        print("🎉 JOB COMPLETED SUCCESSFULLY!")
        print(f"⏱️ Actual Training Duration: {training_time} seconds")
        print(f"💰 Billable Time: {billable_time} seconds")
        print(f"💵 Est. Total Cost: ~${round(billable_time * 1.40 / 3600, 3)}")
    else:
        print(f"❌ Job ended with status: {status}")
        print(f"Failure Reason: {job_info.get('FailureReason', 'Unknown')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
