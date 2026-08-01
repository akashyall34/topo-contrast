#!/usr/bin/env bash
# Launches a spot GPU instance to run the pilot or full pretraining, with an
# auto-shutdown safeguard so a crashed/finished job doesn't bill overnight.
#
# Fill in AMI_ID / INSTANCE_TYPE / KEY_NAME / SECURITY_GROUP for your AWS
# account before use — left as placeholders since these are account-specific.
set -euo pipefail

AMI_ID="${AMI_ID:?set AMI_ID (Deep Learning AMI with CUDA)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
KEY_NAME="${KEY_NAME:?set KEY_NAME}"
SECURITY_GROUP="${SECURITY_GROUP:?set SECURITY_GROUP}"
RUN_SCRIPT="${1:-scripts/run_pilot.sh}"

USER_DATA=$(cat <<EOF
#!/bin/bash
cd /home/ubuntu
git clone <your-repo-url> tubular-topo-contrastive
cd tubular-topo-contrastive
conda env create -f environment.yml
source activate tubular-topo
bash ${RUN_SCRIPT} 2>&1 | tee run.log

# Auto-shutdown after the job finishes (or fails) so a spot instance never
# idles on your bill.
sudo shutdown -h now
EOF
)

aws ec2 request-spot-instances \
    --instance-count 1 \
    --launch-specification "{
        \"ImageId\": \"${AMI_ID}\",
        \"InstanceType\": \"${INSTANCE_TYPE}\",
        \"KeyName\": \"${KEY_NAME}\",
        \"SecurityGroups\": [\"${SECURITY_GROUP}\"],
        \"UserData\": \"$(echo "$USER_DATA" | base64)\",
        \"InstanceInitiatedShutdownBehavior\": \"terminate\"
    }"

echo "Spot request submitted. Instance will self-terminate after ${RUN_SCRIPT} completes."
