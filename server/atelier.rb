# MULTICAST IMPLEMENTATION
# 5/3/2021
# DO NOT USE LOCALHOST, BIND TO LOCAL IP

require_relative "websocket-config-atelier"

Mongo::Logger.logger.level = Logger::WARN


STORAGE_COLLECTION = "XXXX"
SENDER_NAME = "XXXX"


class HapticCaptionSocketServer
  def initialize()
  end
end

def handle_api(ws, msg)
  command = msg["api"]["command"]
  params = msg["api"]["params"]
  print("\tAPI: #{command} #{params}\n")

  case command
    when "RECORD_START"
      $sensor_data = []

    
    when "RECORD_END"     
      # PROCESS $sensor_data TO SCHEMA 
      if $sensor_data.length == 0
        print("No sensor data recorded :(")
        print("Params", params)
      else
        data = $sensor_data.map{|x| x["data"]}.flatten
        time = $sensor_data[0]["time"]

        resp = {
          "event": "matread",
          "time": time, 
          "data": data,
          "params": params
        }.transform_keys(&:to_s)

        print "\t\tSaved #{$sensor_data.length} sensor readings to mongo...\n"
        $client[STORAGE_COLLECTION].insert_one(resp)
        $routes.unicast("Collected sensor data", ws, resp) # Send to subscribers
      end

    else
      $routes.multicast("Unhandled API calls", ws, msg)
    end
end

def handle_events(ws, msg)
  event = msg["event"]
  if event != "read-pressure"
    print("\tEVENT: #{event}\n")
  end
  case event 
    when "read-pressure"
      $sensor_data.push(msg)
      # puts("\t  Samples collected: #{$sensor_data.length}")
      $routes.unicast("Pressure data forwarded to", ws, msg)
    else                
      $routes.multicast("Unhandled events forward to", ws, msg)
    end
end

$pa = HapticCaptionSocketServer.new()
$sensor_data = []
start_server()

