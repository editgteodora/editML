from flask import Flask
from flask import jsonify
from threading import Thread
from queue import Queue

import datetime
import requests
import json
import time


last_recognized_id = ""
writeID = ""
writeHealthStatus = ""
last_recognized_health_status=""
last_recognized_time=datetime.datetime.now()
id_ip=""
health_status_ip=""
locked = False

app = Flask(__name__)

def start_server_thread():

  app = Flask(__name__)

  @app.route('/', methods = ['GET'])
  def server():
    if(locked == False):
      response = jsonify(ID=writeID, HealthStatus=writeHealthStatus,DateTime=last_recognized_time)
      response.headers.add('Access-Control-Allow-Origin', '*')
      return response
    
  if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4200)

server_thread = Thread(target=start_server_thread)
server_thread.start()

def detect_synced_changes():
    read_id = requests.get(id_ip)
    read_health_status = requests.get(health_status_ip)
    queue_pairs = Queue()
    
    while (True):
      locked = True
      if(last_recognized_id != read_id["ID"]):
        queue_pairs.put((read_id["ID"],""))
        last_recognized_id=read_id["ID"]
      if(last_recognized_health_status != read_health_status["HealthStatus"]):
        write_item=queue_pairs.get()
        last_recognized_health_status = read_health_status["HealthStatus"]
        write_item[1] = last_recognized_health_status
        read_id= write_item[0]
        read_health_status = write_item[1]
        last_recognized_time = datetime.datetime.now()
        locked = False
      time.sleep(0.5)


       
