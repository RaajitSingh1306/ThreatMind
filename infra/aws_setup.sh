#!/usr/bin/env bash
# ThreatMind AWS infrastructure bootstrap.
# Creates: S3 bucket, ECR repository, EC2 instance (t3.medium).
# Requires: AWS CLI configured with appropriate IAM permissions.

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
BUCKET_NAME="${S3_BUCKET:-threatmind-$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)}"
ECR_REPO="threatmind-api"
EC2_INSTANCE_TYPE="t3.medium"
EC2_AMI="ami-0c55b159cbfafe1f0"  # Amazon Linux 2 in us-east-1, update for other regions
EC2_KEY_NAME="${EC2_KEY_NAME:-threatmind-key}"

echo "=== ThreatMind AWS Bootstrap ==="
echo "Region:  $REGION"
echo "S3:      $BUCKET_NAME"
echo "ECR:     $ECR_REPO"
echo ""

# ── S3 Bucket ──────────────────────────────────────────────────────────────────
echo "→ Creating S3 bucket..."
if [ "$REGION" == "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION"
else
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION"
fi

aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled

aws s3api put-public-access-block \
    --bucket "$BUCKET_NAME" \
    --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

echo "✓ S3 bucket created: s3://$BUCKET_NAME"

# ── ECR Repository ─────────────────────────────────────────────────────────────
echo "→ Creating ECR repository..."
aws ecr create-repository \
    --repository-name "$ECR_REPO" \
    --region "$REGION" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    2>/dev/null || echo "ECR repo already exists"

ECR_URI=$(aws ecr describe-repositories \
    --repository-names "$ECR_REPO" \
    --region "$REGION" \
    --query 'repositories[0].repositoryUri' \
    --output text)
echo "✓ ECR repository: $ECR_URI"

# ── EC2 Instance ───────────────────────────────────────────────────────────────
echo "→ Creating EC2 key pair..."
aws ec2 create-key-pair \
    --key-name "$EC2_KEY_NAME" \
    --region "$REGION" \
    --query 'KeyMaterial' \
    --output text > "${EC2_KEY_NAME}.pem" 2>/dev/null || echo "Key pair already exists"
chmod 400 "${EC2_KEY_NAME}.pem" 2>/dev/null || true

echo "→ Creating security group..."
SG_ID=$(aws ec2 create-security-group \
    --group-name threatmind-sg \
    --description "ThreatMind API security group" \
    --region "$REGION" \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=threatmind-sg" \
        --region "$REGION" \
        --query 'SecurityGroups[0].GroupId' \
        --output text)

# Allow SSH and API port
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr 0.0.0.0/0 \
    --region "$REGION" 2>/dev/null || true
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0 \
    --region "$REGION" 2>/dev/null || true

echo "→ Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$EC2_AMI" \
    --instance-type "$EC2_INSTANCE_TYPE" \
    --key-name "$EC2_KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --region "$REGION" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=threatmind-api}]" \
    --user-data '#!/bin/bash
        yum update -y
        amazon-linux-extras install docker -y
        systemctl start docker
        systemctl enable docker
        usermod -aG docker ec2-user
        yum install -y awscli' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "→ Waiting for EC2 to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

EC2_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=== Bootstrap Complete ==="
echo "S3 Bucket:       s3://$BUCKET_NAME"
echo "ECR Registry:    $ECR_URI"
echo "EC2 Instance ID: $INSTANCE_ID"
echo "EC2 IP:          $EC2_IP"
echo "SSH:             ssh -i ${EC2_KEY_NAME}.pem ec2-user@$EC2_IP"
echo ""
echo "Add these to your GitHub Secrets:"
echo "  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
echo "  AWS_REGION=$REGION"
echo "  S3_BUCKET=$BUCKET_NAME"
echo "  ECR_REGISTRY=$(echo $ECR_URI | cut -d'/' -f1)"
echo "  EC2_HOST=$EC2_IP"
echo "  EC2_SSH_KEY=<contents of ${EC2_KEY_NAME}.pem>"
