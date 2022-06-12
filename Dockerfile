FROM python:3.6-slim-stretch

RUN mkdir -p /project && mkdir /project/model

WORKDIR /project

COPY requirements.txt /project
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT 8000
ENV PATH_TO_MODEL model/model.pkl

COPY . /project

CMD ["python", "fast.py"]
