let _ = require('underscore');

function makeSocket(name, recordingEvent, setRecordings, setScreen){
  const wsURL = "ws://XXX.XXX.X.XX:XXXX";
  console.log("Connecting to ", wsURL)
  const socket = new WebSocket(wsURL);
  socket.name = name;
  socket.IDCOUNT = 100;
  socket.onmessage = function(result){
    const obj = JSON.parse(result.data)

    console.log(obj)
    if(obj && obj.event === recordingEvent){
      console.log("NEW RECORDING", obj)
      // obj.data.name = "R"+socket.IDCOUNT
      // obj.data.id = socket.IDCOUNT;
      // socket.IDCOUNT++;
      setRecordings(recordings => [...recordings, obj])
      setScreen("pulse")
    }
  }
  socket.onopen = function(event){
    this.greet();
    this.subscribe("haws-server", recordingEvent)
  }
  socket.onclose = function(event){
    console.log(event)
  }
  socket.onerror = function(event){
    console.error(event)
  }
  socket.greet = function(){
    socket.jsend({name: socket.name, event: 'greeting'})
  }
  socket.subscribe = function(sender, service){
    console.log("SUBSCRIBING", {subscribe: sender, service: service})
    this.jsend({subscribe: sender, service: service})
  }
    
  socket.unsubscribe = function(sender, service){
    this.jsend({unsubscribe: sender, service: service})
  }

  socket.jsend = function(msg){
    if(socket.readyState == 1){
      this.send(JSON.stringify(msg));
      console.log(">>", msg);
    }
    else{
      console.log("SIM >>", msg);
    }
  }
  socket.pulse = function(){
    // socket.jsend({"api":{"command":"VISCOSENSE", "params": {"material": "test"}}})
    console.log("sense")
    socket.jsend({"api":{"command":"SENSE", "params": {"material": "test"}}})
  }
  socket.start_recording = function(){
    console.log("start_recording")
    socket.jsend({"api":{"command":"RECORD_START"}})
  }
  socket.end_recording = function(name, color, abbv, pulses){
    let msg = {"api":
      {command:"RECORD_END",
        params:{
            material: name,
            pulses: pulses,
            name: name,
            color: color,
            abbv: abbv
          }
      }
    }
    console.log("end_recording")
    socket.jsend(msg);
  }
  socket.end_recording_and_model = function(name, color, abbv, pulses){
    let msg = {"api":
      {command:"RECORD_END_AND_MODEL",
        params:{
            material: name,
            pulses: pulses,
            name: name,
            color: color,
            abbv: abbv
          }
      }
    }
    console.log("end_recording_and_model")
    socket.jsend(msg);
  }
  socket.timeouts = []
  socket.sense = function(name, color, abbv, pulses){
    console.log("pulses:", pulses)
    const PULSE_TIME = 3000;
    const START_PULSE_GAP = 1000;
    const END_GAP  = 100;
    setScreen("sensing")
    console.log("SENSING", name, "x", pulses);
    let scope = socket

   
    // For each pulse
    _.each(_.range(pulses), function(pulse_id){
      var star_pulse_time =  pulse_id * (PULSE_TIME + START_PULSE_GAP + END_GAP)

      scope.timeouts.push(setTimeout(()=> scope.start_recording(), star_pulse_time));
      console.log("timeline","start", star_pulse_time)

      scope.timeouts.push(setTimeout(scope.pulse, star_pulse_time +  START_PULSE_GAP));
      console.log("timeline","sense", star_pulse_time + START_PULSE_GAP)

      var end_pulse_time = star_pulse_time + START_PULSE_GAP + PULSE_TIME
      
      if(pulse_id != pulses-1){
        scope.timeouts.push(setTimeout(()=>scope.end_recording(name, color, abbv, pulse_id), end_pulse_time)); 
        console.log("timeline","record end", end_pulse_time)
      }
      else{
        scope.timeouts.push(setTimeout(()=>scope.end_recording_and_model(name, color, abbv, pulse_id), end_pulse_time)); 
        console.log("timeline","record end abd model",  end_pulse_time)
      }
      console.log("Condiction: ", pulse_id, pulses)
    })

    
  }
  socket.emergency_stop = function(){
    _.each(socket.timeouts, function(el){ 
      console.log(" Process", el, "terminated");
      clearTimeout(el);
    });
    socket.timeouts = [];
    setScreen("sense")
  }
  socket.clear_out = function(){
    socket.jsend({api:
      {command:"BLOW"}
    });
  }
  return socket
}

export {makeSocket};
