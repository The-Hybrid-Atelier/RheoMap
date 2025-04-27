require 'websocket-client-simple'
require 'json'

# Ensure this IP address and port match those of your WebSocket server
ws = WebSocket::Client::Simple.connect 'ws://XXX.XXX.X.XX:XXXX'

ws.on :open do
  puts 'Connected to server'

  # Send RECORD_END_AND_MODEL command
  command = {
    api: {
      command: 'RECORD_END_AND_MODEL',
      params: {
        material: 'Water',
        pulses: 2,
        name: 'Water',
        color: 'WTR',
        abbv: '#95D8EB'
      }
    }
  }

  puts "Sending command: #{command.to_json}"
  ws.send(command.to_json)
end

ws.on :message do |msg|
  puts "Received message: #{msg.data}"
end

ws.on :close do |e|
  puts "Connection closed: #{e.inspect}"
end

ws.on :error do |e|
  puts "Error: #{e.message}"
end

# Keep the script running for a while to allow for message exchange
sleep 10
ws.close
