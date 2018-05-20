#!/bin/sh

kaggle competitions download -c coupon-purchase-prediction
mkdir datasets
cp -r ~/.kaggle/competitions/coupon-purchase-prediction/* datasets/
cd datasets
unzip '*.zip'
