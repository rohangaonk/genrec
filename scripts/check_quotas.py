import boto3
import os

def check_quotas():
    profile = os.environ.get("AWS_PROFILE", "tutor-deploy")
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    session = boto3.Session(profile_name=profile, region_name=region)
    sq = session.client("service-quotas")

    print(f"🔎 Checking SageMaker Training Job Quotas in {region}...")
    paginator = sq.get_paginator("list_service_quotas")
    
    found = []
    for page in paginator.paginate(ServiceCode="sagemaker"):
        for quota in page["Quotas"]:
            name = quota["QuotaName"]
            val = quota["Value"]
            if "training job usage" in name.lower() and val > 0:
                found.append((name, val))
                print(f"  ✅ [ALLOWED] {name}: {val} instances")

    if not found:
        print("  ⚠️ No non-zero SageMaker training quotas found in this region.")

if __name__ == "__main__":
    check_quotas()
