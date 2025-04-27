import './App.css';
import 'semantic-ui-css/semantic.min.css';
import {RheoSenseView, RheoPulseView, RheoLibraryView, RheoSensingView} from './Rheosense/RheoInterface.js'
import Ruler from './javascripts/ruler.js';
import { useState, useEffect } from 'react';
import {makeSocket} from "./Socket.js"
import {Row, Column, Pill} from './Components.js';
import {Button, Icon} from 'semantic-ui-react';
import {View} from './View.js';
import {referencesLibrary, recordingsLibrary} from "./Rheosense/RheoResults"

let _ = require('underscore');
const START_SCREEN = "pulse"; // sense| pulse | library
const START_VIEW = "pca"; // profile | pca
const MODELING_EVENT = "matmodel";


function loadFrom(x, active){
    let profiles = _.map(x, function(profile){
      profile.active = active;
      profile.pulseview = active;
      return profile;
    })
    return profiles;
}


function App() {
  const [socket, setSocket] = useState(null);
  const [appScreen, setScreen] = useState(START_SCREEN);
  const [profiles, setProfiles] = useState([])


  useEffect(()=>{
        setSocket(makeSocket("rheosense-controller", MODELING_EVENT, setProfiles, setScreen))
        setProfiles(recordings => [...loadFrom(recordingsLibrary, true), ...loadFrom(referencesLibrary, false)]);
    }, [])
  return (
    <div className="App">
      {appScreen == "sense" &&
        <View icon="cancel" goToScreen="pulse" screen={appScreen} setScreen={setScreen} padding="paddedtb">
          <RheoSenseView socket={socket} screen={appScreen} ></RheoSenseView>
        </View>
      }
      {appScreen == "sensing" &&
        <View icon="cancel" goToScreen="sense" screen={appScreen} setScreen={setScreen} padding="paddedtb">
          <RheoSensingView socket={socket} screen={appScreen} ></RheoSensingView>
        </View>
      }
      {appScreen == "pulse" &&
        <View icon="plus" goToScreen="sense" screen={appScreen} setScreen={setScreen} padding="paddedtb">
          <RheoPulseView socket={socket} screen={appScreen} startView={START_VIEW} setScreen={setScreen} profiles={profiles} setProfiles={setProfiles}></RheoPulseView>
        </View> 
      }
      {appScreen == "library" &&
        <View icon="back" goToScreen="pulse" screen={appScreen} setScreen={setScreen} padding="paddedtb">
          <RheoLibraryView socket={socket} screen={appScreen} setScreen={setScreen} profiles={profiles} setProfiles={setProfiles}></RheoLibraryView>
        </View> 
      }
      

    </div>
  );
}

export default App;
