FROM python:3.11-slim
EXPOSE 8088
RUN mkdir /home/backend
ADD . /home/backend
RUN apt-get update
RUN apt-get install python3-pip -y
RUN pip install --no-cache-dir -r /home/backend/requirements.txt
RUN pip install uvicorn
WORKDIR /home/backend
CMD ["uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "8088"]