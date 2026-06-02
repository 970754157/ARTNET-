#!/usr/bin/env bash
nohup python -m project.train --config configs/exp_bert_resnet50_pretrained.json > logs/exp_bert_resnet50_pretrained.log 2>&1
nohup python -m project.train --config configs/exp_bert_resnet18_pretrained.json > logs/exp_bert_resnet18_pretrained.log 2>&1
nohup python -m project.train --config configs/exp_bert_resnet18_scratch.json > logs/exp_bert_resnet18_scratch.log 2>&1
nohup python -m project.train --config configs/exp_bert_vgg19.json > logs/exp_bert_vgg19.log 2>&1
nohup python -m project.train --config configs/exp_bert_dinov2.json > logs/exp_bert_dinov2.log 2>&1
nohup python -m project.train --config configs/exp_roberta_resnet50.json > logs/exp_roberta_resnet50.log 2>&1
nohup python -m project.train --config configs/exp_bert_fasterrcnn.json > logs/exp_bert_fasterrcnn.log 2>&1
nohup python -m project.train --config configs/exp_albert_inceptionresnetv2.json > logs/exp_albert_inceptionresnetv2.log 2>&1

