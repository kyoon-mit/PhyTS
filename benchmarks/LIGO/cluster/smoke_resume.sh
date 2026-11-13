#!/bin/bash
# Resume test for train_ligo_cfg.sh: 1 epoch, then raise max_epochs to 2 and run again (should resume).
cd /home/barmstrong/TimeSeriesPhysics
/lustre/home/barmstrong/phyts/cluster/train_ligo_cfg.sh configs/LIGO/v2/_smoke_resume_linoss.yaml && echo "PASS run1" || echo "FAIL run1"
ls -la /fast/barmstrong/LIGO_2048Hz_v2_smoke/ckpt
sed -i 's/max_epochs: 1$/max_epochs: 2/' configs/LIGO/v2/_smoke_resume_linoss.yaml
/lustre/home/barmstrong/phyts/cluster/train_ligo_cfg.sh configs/LIGO/v2/_smoke_resume_linoss.yaml && echo "PASS run2" || echo "FAIL run2"
