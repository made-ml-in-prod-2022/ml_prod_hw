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
***
__Docker image optimizations experience:__
* Добавил несколько файлов в ```.dockerignore``` - особо не выиграл, было 2.7GB стало 2.69GB
Когда же выбрал другую начальную среду (вместо ```python:3.9``` ```python:3.6-slim-stretch```) образ стал весить 2.17GB
* стоило перетащить команду ```COPY . /project``` вниз, а 
```
COPY requirements.txt /project
RUN pip install --no-cache-dir -r requirements.txt
```
наверх в Dockerfile и теперь его пересборка вместо утомляющих 180 секунд стала занимать 1.4 секунды