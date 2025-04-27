import { useState } from 'react';
import { Input, Label, Button, Image, Icon} from 'semantic-ui-react';
import {Row, Column, Pill} from '../Components.js';
import { Loader, Dimmer } from 'semantic-ui-react'

const RheoSensingView = function({socket, setScreen}){
   return (
    <>
        <Column className="sense-view fh">
          <Row>
            <Dimmer active inverted>
              <Loader size='massive'>Sensing</Loader>
            </Dimmer>
          </Row>
          <Row>
            
            <Button className="stop" onClick={socket.emergency_stop}>
              EMERGENCY STOP
            </Button>
          </Row>
        </Column>
    </>
    );
}
const RheoSenseView = function({socket, setScreen}){
  const [name, setName] = useState("Water")
  const [abbv, setAbbv] = useState("WTR")
  const [pulses, setPulses] = useState(3)
  const [color, setColor] = useState("#95D8EB")

  const handleNameChange = (event)=>{setName(event.target.value)}
  const handleAbbvChange = (event)=>{setAbbv(event.target.value)}
  const handlePulseChange = (event)=>{setPulses(parseInt(event.target.value))}
  const handleColorChange = (event)=>{setColor(event.target.value)}
  const sense = ()=> {
    console.log(socket)
    socket.sense(name, abbv, color, pulses);
  }
  
  return (
    <>
        <Column className="sense-view between">
          <Row>
            <Column>
              <Label> Label </Label>
              <Input className="hero" type='text' value={name} onChange={handleNameChange}>
                  <input type='text'/>
              </Input>
            </Column>
          </Row>
          <Row>
            <Column>
            <Label> ABBV </Label>
            <Input type='text' value={abbv} onChange={handleAbbvChange}>
                <input type='text'/>
            </Input>
            </Column>
          </Row>
          <Row>
            <Column>
            <Label> PULSES </Label>
            <Input type='number' value={pulses} onChange={handlePulseChange}>
                <input type='number'/>
            </Input>
            </Column>
          </Row>
          

          <Row className="wrap">
            <Row>
            <Button onClick={socket.clear_out}>
              CLEAR
            </Button>
            <Button onClick={sense}>
              SENSE
            </Button>
            </Row>
            <Button className="stop" onClick={socket.emergency_stop}>
              EMERGENCY STOP
            </Button>
          </Row>
        </Column>
    </>
  );
}

export {RheoSenseView, RheoSensingView};