FROM python:3.11-slim
RUN mkdir /home/bot
RUN apt-get update
RUN apt-get install python3-pip -y
ADD . /home/bot
RUN pip install --no-cache-dir -r /home/bot/requirements.txt
WORKDIR /home/bot
CMD ["python3", "tg_bot.py"]