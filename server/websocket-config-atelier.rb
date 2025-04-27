# websocket_config.rb

require 'em-websocket'
require 'json'
require 'optparse'
require 'socket'
require 'mongo'


Mongo::Logger.logger.level = Logger::WARN
$options = {b: "0.0.0.0", p: 3047, v: false, ssl: false, cert: nil, key: nil}

OptionParser.new do |opts|
  opts.banner = "Usage: example.rb [options]"
  opts.on("-d", "--[no-]debug", "Run debug") do |d|
    $options[:d] = d
  end
  opts.on("-v", "--[no-]verbose", "Run verbosely") do |v|
    $options[:verbose] = v
  end
  opts.on("-bIP", "--ip=IP", "Bind to IP address") do |v|
    $options[:b] = v
  end
  opts.on("-pPORT", "--p=PORT", "Bind to specific port") do |v|
    $options[:p] = v
  end
  opts.on("--ssl", "Enable SSL") do |v|
    $options[:ssl] = true
  end
  opts.on("--cert=FILE", "Path to SSL certificate file") do |v|
    $options[:cert] = v
  end
  opts.on("--key=FILE", "Path to SSL key file") do |v|
    $options[:key] = v
  end
end.parse!

p $options

class Routes
  attr_accessor :websockets

  def initialize()
    @websockets = {}
    @multicast = {}
    @subcast = {}
    @unicast = {}
  end

  def add(ws)
    ws.routes = self  # Set the routes instance to the WebSocket connection
    @websockets[ws.signature] = ws
    ws.name = ws.remote_ip()
    ws.subscriptions = {}

    data = {}
    data["sid"] = ws.signature
    data["ip"] = ws.name
    msg = {event: "connection_opened", data: data}
    self.multicast("Opening forward to", ws, msg)
    puts("Opened - #{msg}")
    print()
  end



  def delete(signature)
    ws = @websockets[signature]
    data = {}
    data["sid"] = ws.signature
    data["ip"] = ws.name
    @websockets.delete(signature)
    data = {}
    msg = {event: "connection_closed", data: data}
    self.multicast("Closing forwarded to", ws,  msg)
    # puts("Closed - #{msg}")
    # print()
  end

  def print()  
  	puts "  #{@websockets.length} Devices Connected - #{@websockets.map{ |sid, ms| ms.name}}\n"
    # puts "   Routes >> MC: #{@multicast.length}, *: #{@subcast.length}, UC: #{@unicast.length}\n"
  end

  def server_state()
    msg = {}
    msg["data"] = self.websockets.map { |sid, ms| ms.name }
    multicast("Server state update", nil, msg)  # Use nil or a placeholder for ws
  end

    # Send to all connected
  def multicast(prefix="", ws=nil, msg)
    @multicast = @websockets.reject { |sid, ms| ws && [ws.signature].include?(sid) }
    return if @multicast.empty?

    @multicast.each { |sid, ms| ms.jsend(ws ? ws.name : "server", msg) }
    puts "\tMC: #{prefix} sent to #{@multicast.map { |sid, ms| ms.name }} devices..\n"
  end  
  # Send to those with blanket * subscriptions
  def subcast(prefix="", ws, msg)
  	@subcast = @websockets.select { |sid, ms| ms.subscriptions["*"] }
    if @subcast.empty? then return end
    @subcast.each{ |sid, ms| ms.jsend(ws.name, msg)}
    puts "\tSC: #{prefix} sent to #{@subcast.map{|sid, ms| ms.name}} devices..\n"
  end
  
  # Send to only those subscribed to specific sender/event
  def unicast(prefix="", ws, msg)
  	# puts "UNI MSG #{msg}\n ---#{msg["event"]}---\n"
  	# @websockets.map do |sid, ms| 
  	# 	puts "#{ms.name} subscriptions -- looking for #{ws.name}:#{msg["event"]}\n"
  	# 	puts "#{ms.subscriptions}\n"
  	# end

  	@unicast = @websockets.select { |sid, ms| ms.subscriptions[ws.name] and ms.subscriptions[ws.name].include? msg["event"] }
  	if @unicast.empty? then return end
    @unicast.each{ |sid, ms| ms.jsend(ws.name, msg)}
    puts "\tUC: #{prefix} sent to #{@unicast.map{ |sid, ms| ms.name}} devices..\n"
  end

end

module JSONWebsocket
  attr_accessor :name, :subscriptions, :api_handler, :event_handler, :routes

  def post_init
    @subscriptions = {}
    print "#{websockets.length} Devices Connected\n"    
  end

  def jsend(sender, msg)
    header = {}
    header["sender"] = sender
    msg = header.merge(msg)
    self.send(msg.to_json)
    #puts("Server >> #{self.name} #{msg}\n")
  end

  def jsend_api(command, params)
    header = {}
    header["sender"] = sender
    header["api"] = {command: command, params: params}
    self.send(header.to_json)
    #puts("Server >> #{self.name} #{msg}\n")
  end

  def remote_ip
    if self.get_peername
      self.get_peername[2,6].unpack('nC4')[1..4].join('.')
    else
      return nil
    end
  end

  def subscribe(msg)
    sender = msg["subscribe"]
    service = msg["service"]

    puts "\tSUBSCRIBE: #{self.name} subscribing to #{sender}'s #{service}:\n"
    
    if not self.subscriptions[sender]
      self.subscriptions[sender] = []
    end
    self.subscriptions[sender] << service
    puts "\t\t#{self.subscriptions.to_s}\n"
  end

  def unsubscribe(msg)
    sender = msg["unsubscribe"]
    service = msg["service"]
    puts "\tUNSUBSCRIBE: #{self.name} unsubscribing from #{sender}'s #{service}:\n"
    
    if self.subscriptions[sender]
      self.subscriptions[sender].delete_if{|item| item == service}
    else
      puts "\tNot subscribed to #{service} from #{sender}"
    end
    puts "\t\t#{self.subscriptions.to_s}\n"
  end

  def store(msg)
    collection = msg["store"]
    result = $client[collection].insert_one(msg)
    puts "\tSTORE: Mongo << #{msg}\n"
  end

  def set_name(msg)
    puts "\tGREET: #{msg["name"]}\n"
    self.name = msg["name"]
  end

  def handle_special_services(msg)
    case
      when msg["event"] == "greeting"
        self.set_name(msg)
      when msg["store"]
        self.store(msg)
      when msg["subscribe"]
        self.subscribe(msg)
      when msg["unsubscribe"]
        self.unsubscribe(msg)
      when msg.key?("api") && msg["api"]["command"] == "SERVER_STATE"
        self.routes.server_state()
      else
        # No special service detected
        return msg
    end
    return nil
  end

end

module EventMachine
  module WebSocket
    class Connection < EventMachine::Connection
      include JSONWebsocket
    end
  end
end

$client = Mongo::Client.new([ '138.68.230.2:27017' ], 
   user: 'hybridatelier',
   password: 'ERB281282',
   database: 'slurpie',
   auth_source: 'admin',
   auth_mech: :scram
  )

def start_server
  EventMachine.run do
    @channel = EM::Channel.new
    $routes = Routes.new()

    websocket_options = {
      host: $options[:b],
      port: $options[:p],
      debug: $options[:d]
    }

    if $options[:ssl]
      websocket_options[:secure] = true
      websocket_options[:tls_options] = {
        private_key_file: $options[:key],
        cert_chain_file: $options[:cert]
      }
    end

    puts "Server: Started at ws#{$options[:ssl] ? 's' : ''}://#{$options[:b]}:#{$options[:p]}"


    EM::WebSocket.start(websocket_options) do |ws|
      
      ws.onopen do |handshake|

        $routes.add(ws)
      end
      ws.onmessage do |msg| 

        begin 
          msg = JSON.parse(msg)
          msg["timestamp"] = Time.now.to_f
          msg = ws.handle_special_services(msg)

          if not msg.nil? then 
            if msg["api"]
              handle_api(ws, msg)
            elsif msg["event"]
              handle_events(ws, msg)
            else
              puts("\t#{msg} needs to be processed") 
            end
          end

          

        rescue StandardError => bang
          puts "\tInvalid JSON message received #{msg} : #{bang}.\n" 
          ws.jsend("server", {"error": "Invalid JSON message received #{msg}"})
	  #puts bang.backtrace.join("\n")  # Print the stack trace to the console
          #raise
        end    
      end

      ws.onclose do 
        $routes.delete(ws.signature)
      end

    end
  end
end
