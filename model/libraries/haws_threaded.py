from libraries.haws import *
from libraries.utils import *
import threading
import time
    

class ThreadedJSONWebsocketClient(JSONWebSocketClient, threading.Thread):
  def __init__(self, name, uri):
    threading.Thread.__init__(self)
    super(ThreadedJSONWebsocketClient, self).__init__(name, uri)
    self.active = False
    self.counter = 0
    self.time_limit = 5
    header("Thread %s created with %i seconds of runtime" % (self.name,self.time_limit), 0)
  def shutdown(self):
    self.active = False
    self.close()
  def set_limit(self, limit):
    self.time_limit = limit
  def run(self):
    c = time.time()
    header("Starting " + self.name, 1)
    self.active = True
    try:
      while self.active:
        if (time.time() - c) > self.time_limit:
          break
        self.counter = self.counter + 1
        self.listen()
        # time.sleep(0.5)
      runtime = time.time() - c
      header("Exiting %s: %i calls - %2.2f seconds of runtime - %2.2f packet rate"%(self.name, self.counter, runtime, self.counter/runtime), 1)
    except:
      header("Some issue with %s"%(self.name), -1)
      raise
      return 0

  
