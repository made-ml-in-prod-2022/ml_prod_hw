# ML prod hw2
---
__Docker:__
To get docker image from Dockerhub:
```
docker pull iqgroper/ml_prod_hw:latest
```
and to start it
```
docker run --rm -p 8000:8000 iqgroper/ml_prod_hw
```
***
To build docker image locally 
```
docker build -t hw2:optim .
```
to start it 
```
docker run --rm -p 8000:8000 hw2:optim
```
***
__Usage:__
***
Hit method ```/health``` to check whether service is ready to call ```/predict```
***
After starting docker container with online inference model, start script requesting the service with
```
python req.py
```
__Data__
Features must be in this exact order:
* age
* sex
* cp
* trestbps
* chol
* fbs
* restecg
* thalach
* exang
* oldpeak
* slope
* ca
* thal

In case any incorrect data sent, you will get __Error 400__
