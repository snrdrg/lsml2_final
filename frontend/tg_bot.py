import requests as rq
import json
import base64
import asyncio
from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import ContentType
from aiogram.types import BotCommand
from aiogram.types import ReplyKeyboardRemove, ReplyKeyboardMarkup
from aiogram.types import KeyboardButton as kb
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from multiprocessing import Process
from functools import wraps, partial

print('Loading API keys')
with open('api_key.json', 'r') as f:
    API_TOKEN = json.load(f)["key"]

print('Loading configurations')
with open('bot_messages.json', 'r', encoding='utf-8') as f:
    bot_messages = json.load(f)
print('Complete')

WELCOME_STR = bot_messages['welcome']
HELP_STR = bot_messages['help']#'Go help yoursef'

 
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

async def setup_bot_commands(dp):
    await dp.bot.set_my_commands([
        BotCommand(command="/start", description="Start bot"),
        BotCommand(command="/help", description="Show help")
    ])

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply(WELCOME_STR)


@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    await message.reply(HELP_STR)
   

@dp.message_handler()
async def echo(message: types.Message):
    photos = message.photo
    for photo in photos:
        await photo.download()
        url = "http://backend:8088/model"
        reqHeader = {'Content-Type': 'application/json'}
        json_data = {"question": base64.b64encode(photo.file).decode("utf8")}
        try:
            resp = rq.post(url, headers=reqHeader, json=json_data, verify=False)
            if resp.status_code == 200:
                result = json.loads(json.loads(resp.content.decode('utf-8')))['result']
            else:
                print('Service unavailable. Status code:', resp.status_code)
                result = 'Не могу обработать ваш запрос. Модель не доступна, попробуйте позже. 1 ' + str(resp.status_code)         
        except rq.exceptions.ConnectionError as e:
            print('Failed to connect to the server:', e)
            result = 'Не могу обработать ваш запрос. Модель не доступна, попробуйте позже. 2 ' + str(e)
        except Exception as e:
            print('An error occurred:', e)
            result = 'Не могу обработать эти данные'

    await message.reply(result)


if __name__ == '__main__':
    print('Starting bot')
    executor.start_polling(dp, skip_updates=True)
