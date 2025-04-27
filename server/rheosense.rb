#####################################
# # MULTICAST IMPLEMENTATION
# # Demo 2024
# # 14/06/2024
# # DO NOT USE LOCALHOST, BIND TO LOCAL IP


require_relative "websocket-config"
require 'eventmachine'
require 'em-websocket'
require 'json'
require 'mongo'

STORAGE_COLLECTION = "XXXX"
SENDER_NAME = "XXXX"

class ProgrammableAir
  attr_accessor :pressure_on

  def initialize
    @pressure_on = true
  end

  def turn_on(ws)
    @pressure_on = true
    print("\tPA ON\n")
    trigger = { api: { command: "PRESSURE_ON" } }
    $routes.multicast("Server initiated PA command", ws, trigger)
  end

  def turn_off(ws)
    @pressure_on = false
    print("\tPA OFF\n")
    trigger = { api: { command: "PRESSURE_OFF" } }
    $routes.multicast("Server initiated PA command", ws, trigger)
  end
end

def handle_api(ws, msg)
  command = msg["api"]["command"]
  params = msg["api"]["params"]
  print("\tAPI: #{command} #{params}\n")

  case command
  when "VISCOSENSE"
    $sensor_data = []
    print(command, $sensor_data.length);
  when "SENSE"
    $pa.turn_on(ws)
  when "RECORD_START"
    $sensor_data = []
    print(command, $sensor_data.length);
  when "RECORD_END_AND_MODEL"
    resp = process_sensor_data(params, ws, "matsense")
  
    # resp[:data] = pulses_append(params["pulses"], $sensor_data)
    send_model(params, ws, "matsense", resp)
    
    $pa.turn_off(ws)
    $sensor_data = []
  when "MODEL_TEST"
    resp = {}
    resp[:data] = pulses_append(params["pulses"], $sensor_data)
    send_model(params, ws, "matsense", resp)
  when "RECORD_END"
    resp = process_sensor_data(params, ws, "matread")
    $sensor_data = []
    print(command, $sensor_data.length);
  when "CHARLIE"
    print("Test")
  else
    $routes.multicast("Unhandled API calls", ws, msg)
  end
end

def pulses_append(pulses, current_sensor_data)
  last_two_pulses_data = $client[STORAGE_COLLECTION]
                          .find
                          .sort({ time: -1 })
                          .limit(2)
                          .map { |record| record["data"] }
                          .flatten

  current_pulse_data = current_sensor_data.map { |x| x["data"] }.flatten
  last_two_pulses_data + current_pulse_data
end

def send_model(params, ws, event_type, response)
  print("Sending to model", ws.name)
  $routes.unicast("Collected sensor data", ws, response)
end

def process_sensor_data(params, ws, event_type)
  if $sensor_data.empty?
    print("No sensor data recorded :(\n")
    print("Params: #{params}\n")
    return {}
  else
    print("Params: #{params}\n")

    data = $sensor_data.map { |x| x["data"] }.flatten
    time = $sensor_data[0]["time"]

    resp = {
      event: event_type,
      time: time,
      data: data,
      params: params
    }.transform_keys(&:to_s)

    print "\t\tSaved #{$sensor_data.length} sensor readings to MongoDB...\n"
    $client[STORAGE_COLLECTION].insert_one(resp)
    return resp
  end
end

def handle_events(ws, msg)
  event = msg["event"]
  if event != "read-pressure"
    print("\tEVENT: #{event}\n")
  end

  case event
  when "read-pressure"
    if $pa.pressure_on
      $sensor_data.push(msg)
      #print("\t  Samples collected: #{$sensor_data.length}\n")
      $routes.unicast("Pressure data forwarded to", ws, msg)
    end
  else
    $routes.multicast("Unhandled events forward to", ws, msg)
  end
end

$pa = ProgrammableAir.new
$sensor_data = []
start_server()





