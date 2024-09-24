#!/bin/bash

# 実行したいコマンドをnohupでバックグラウンド実行
nohup /usr/bin/python3 qiskit_entropy/vn_ent_qubits10.py > output.log 2>&1 &

# 実行後のメッセージ（オプション）
echo "your_command is running in the background, log output is redirected to output.log"
