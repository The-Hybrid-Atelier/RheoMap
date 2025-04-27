# MULTICAST IMPLEMENTATION
# 5/3/2021

require 'em-websocket'
require 'json'
require 'optparse'
require 'socket'
require 'mongo'

STORAGE_COLLECTION = "XXXX"
# DO NOT USE LOCALHOST, BIND TO LOCAL IP
client = Mongo::Client.new([ 'XXXX' ], 
   user: 'XXXX',
   password: 'XXXX',
   database: 'XXXX',
   auth_source: 'XXX',
   auth_mech: :scram
  )

Mongo::Logger.logger.level = Logger::WARN
options = {b: "0.0.0.0", p: 3031, v: false, ssl: false, cert: nil, key: nil}

OptionParser.new do |opts|
  opts.banner = "Usage: example.rb [options]"
  opts.on("-d", "--[no-]debug", "Run debug") do |d|
    options[:d] = d
  end
  opts.on("-v", "--[no-]verbose", "Run verbosely") do |v|
    options[:verbose] = v
  end
  opts.on("-bIP", "--ip=IP", "Bind to IP address") do |v|
    options[:b] = v
  end
  opts.on("-pPORT", "--p=PORT", "Bind to specific port") do |v|
    options[:p] = v
  end
  opts.on("--ssl", "Enable SSL") do |v|
    options[:ssl] = true
  end
  opts.on("--cert=FILE", "Path to SSL certificate file") do |v|
    options[:cert] = v
  end
  opts.on("--key=FILE", "Path to SSL key file") do |v|
    options[:key] = v
  end
end.parse!

p options


module JSONWebsocket
  attr_accessor :name
  attr_accessor :subscriptions
  @subscriptions = {}

  def jsend(msg, sender)
    header = {}
    header["sender"] = sender
    msg = header.merge(msg)
    self.send(msg.to_json)
    
    #print("Server >> #{self.name} #{msg}\n")
  end
  def remote_ip
    if self.get_peername
      self.get_peername[2,6].unpack('nC4')[1..4].join('.')
    else
      return nil
    end
  end
end

module EventMachine
  module WebSocket
    class Connection < EventMachine::Connection
      include JSONWebsocket
    end
  end
end

def handle_special_services(ws, msg)
  if msg["event"] == "greeting"
    ws.name = msg["name"]
  elsif msg["store"]
    collection = msg["store"]
    # result = client[collection].insert_one(msg)
    print "Mongo << "+ msg + "\n"
  elsif msg["subscribe"]
    sender = msg["subscribe"]
    service = msg["service"]
    print "\t#{ws.name} subscribing to #{sender}'s #{service}:\n"
    if not ws.subscriptions[sender]
      ws.subscriptions[sender] = []
    end
    ws.subscriptions[sender] << service
    print "\t\t#{ws.subscriptions.to_s}\n"
  elsif msg["unsubscribe"]
    sender = msg["unsubscribe"]
    service = msg["service"]
    print "\t#{ws.name} unsubscribing from #{sender}'s #{service}:\n"
    
    if ws.subscriptions[sender]
      ws.subscriptions[sender].delete_if{|item| item == service}
    else
      print "\tNot subscribed to #{service} from #{sender}"
    end
    print "\t\t#{ws.subscriptions.to_s}\n"
  elsif msg.key?("api") and msg["api"]["command"] == "SERVER_STATE"
    msg = {}
    msg["data"] = websockets.map { |sid, ms| ms.name}
    ws.jsend(msg, "socket-server")
  end
end

EventMachine.run do
  @channel = EM::Channel.new
  
  # START JSON WEBSOCKET SERVER
  websockets = {} # Keeps track of all socket connections
  sensor_data = []

  websocket_options = {
    host: options[:b],
    port: options[:p],
    debug: options[:d]
  }

  if options[:ssl]
    websocket_options[:secure] = true
    websocket_options[:tls_options] = {
      private_key_file: options[:key],
      cert_chain_file: options[:cert]
    }
  end
  
  EM::WebSocket.start(websocket_options) do |ws|
    
    ws.onopen do |handshake|
      ws.name = ws.remote_ip()
      ws.subscriptions = {}
      websockets[ws.signature] = ws
      
      data = {}
      data["sid"] = ws.signature
      data["ip"] = ws.name
      msg = {event: "connection_opened", data: data}
      ws.jsend(msg, "socket-server")
      ws.jsend({"api": {"command": "PUMP_OFF"}}, "socket-server")
      print "#{websockets.length} Devices Connected\n"
      
      # MULTICAST ONLY JSON-FORMATTED MESSAGES
      ws.onmessage do |msg, data|
        begin 
          msg = JSON.parse(msg)
          msg["timestamp"] = Time.now.to_f
          # STORE IN MONGODB IS STORE COLLECTION IS PRESENT
          
          # DON'T ECHO MESSAGE
          multicast = websockets.reject { |sid, ms| [ws.signature].include? sid }
          subcast = websockets.select { |sid, ms| ms.subscriptions["*"] }
          unicast = websockets.select { |sid, ms| ms.subscriptions[ws.name] and ms.subscriptions[ws.name].include? msg["event"] }
          
          if options[:verbose]
            if not msg["api"] 
              print "\n#{ws.name} >> #{msg}\n  MC: #{multicast.length}, *: #{subcast.length}, UC: #{unicast.length}\n"
            else
              print("-")
            end
          end

          handle_special_services(ws, msg)
          
          #print("#{msg}\n")

          # EVENT HANDLING
          if msg["event"]
            event = msg["event"]
            #print(event)
            case event 
              when "read-pressure"
		#readings = msg["data"].split.map(&:to_i)
                #msg["data"] = readings
		sensor_data.push(msg)
		multicast.each{ |sid, ms| ms.jsend(msg, ws.name)}                
	        print "."
	      when "matmodel"
                print "MATMODEL Name: #{ws.name}\n"
                #unicast.each{ |sid, ms| ms.jsend(msg, ws.name)}
              else                
                print "??? Event: #{msg}\n"
              end
          end
          

          # API HANDLING
          if  msg["api"]

            command = msg["api"]["command"]
            print("API command #{command}\n")

            case command
              when "VISCOSENSE"
                print("STARTED RECORDING")
                sensor_data = []
                trigger = {"api": {"command": "PRESSURE_ON"}}
                multicast.each{ |sid, ms| ms.jsend(trigger, ws.name)} 
              
              when "RECORD_START"
                print("STARTED RECORDING")
                sensor_data = []
                trigger = {"api": {"command": "PRESSURE_ON"}}
                multicast.each{ |sid, ms| ms.jsend(trigger, ws.name)} 
              
              when "RECORD_END"
                print("ENDED RECORDING")
                print "\t\tSaving pressure reading #{sensor_data.length}..\n"
                # PROCESS SENSOR_DATA TO SCHEMA 

                if sensor_data.length == 0
                  print("No sensor data recorded :(")
                  print("Params", msg["api"]["params"])
                else
                  data = sensor_data.map{|x| x["data"]}.flatten
                  time = sensor_data[0]["time"]

                  sample = {
                    "time": time, 
                    "data": data,
                    "params": msg["api"]["params"]
                  }

                  # print(sample)
                  print "\t\tSaved to mongo...\n"
                  client[STORAGE_COLLECTION].insert_one(sample)
                  sample["event"] = "matread"
		              print "#{ms.name} trying to send #{sample}\n"
                  multicast.each{ |sid, ms| ms.jsend(sample, ws.name)}
                end

                
                # client[STORAGE_COLLECTION].insert_one(msg)
                trigger = {"api": {"command": "PRESSURE_OFF"}}
                multicast.each{ |sid, ms| ms.jsend(trigger, ws.name)} 
                print "\t\tSaved to mongo...\n"
              else
                print "\t\tMC: Sent to #{multicast.length} devices..\n"
                multicast.each{ |sid, ms| ms.jsend(msg, ws.name)} 
                # client["flowstudy_2022"].insert_one(msg) 
              end
          end

          if (subcast.length + unicast.length) > 0
            print "\t\tUC: Sent to #{subcast.length + unicast.length} devices..\n"
            print unicast.map{|sid, ms|ms.name}
            print subcast.map{|sid, ms|ms.name}

            subcast.each { |sid, ms| ms.jsend(msg, ws.name)} 
            unicast.each { |sid, ms| ms.jsend(msg, ws.name)}
          end
        rescue StandardError => bang
           print "Invalid JSON message received #{msg} : #{bang}.\n" 
           ws.jsend({"error": "Invalid JSON message received #{msg}"}, "socket-server")
        end    
      end

      ws.onclose do
        data = {}
        data["sid"] = ws.signature
        data["ip"] = ws.name
        msg = {event: "connection_closed", data: data}
        websockets.delete(ws.signature)
        print "#{websockets.length} Devices Connected\n"
        websockets.each do |sid, ms|
          ms.jsend(msg, "socket-server")
        end  
      end

    end
  end

  puts "Server: Started at ws#{options[:ssl] ? 's' : ''}://#{options[:b]}:#{options[:p]}"
  print "#{websockets.length} Devices Connected\n"
end


