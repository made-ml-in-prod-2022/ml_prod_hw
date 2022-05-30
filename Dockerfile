FROM python:3.9

RUN mkdir -p /project

WORKDIR /project

COPY requirements.txt /project
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT 8000
ENV PATH_TO_MODEL model/model.pkl

COPY . /project

CMD ["python", "fast.py"]
