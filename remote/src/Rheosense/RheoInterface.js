import { useState, useEffect } from 'react';
import { Input, Label, Button, Image, Icon} from 'semantic-ui-react';
import {Row, Column, Pill} from '../Components.js';
import {RheoSenseView, RheoSensingView} from "./RheoSenseView.js"
import {RheoPulseView} from "./RheoPulseView.js"
import {RheoLibraryView} from "./RheoLibrary.js"

let _ = require('underscore');

const ViewPicker = function({view, setView, exportSVG}){

  return (
    <div id="view-picker">
      <Button.Group className="small">
        <Button className={(view == "profile" ? 'selected' : '')}  primary icon onClick={()=>setView("profile")}>
          <img src="/profile.png"/>
        </Button>
        <Button className={(view == "pca" ? 'selected' : '')} primary icon onClick={()=>setView("pca")}>
          <img src="/pca.png"/>
        </Button>
        {false &&
          <Button icon onClick={()=>setView("origami")}>
            <Icon name='star' />
          </Button>
        }
        { true &&
          <Button icon onClick={exportSVG}>
            <Icon name='download'  />
          </Button>
        }
      </Button.Group> 
    </div>
  );
}

const RheoGallery = function({socket, title, filter, profiles, profileClass, setProfiles, palette, screen}){

  const [subProfiles, setSubProfiles] = useState([])

  const handleRemove = function(idx){
    setProfiles(profiles => {
      return profiles.filter((_, i) => i !== idx)
    })
  }
  const handleUpdate = (id)=>{
    // console.log(id, profiles, filter)
    const currentItemIndex = profiles.findIndex((item) => item.id === id);
    let update = {active: !profiles[currentItemIndex].active}
    if(filter == "reference"){
      update["pulseview"] = !profiles[currentItemIndex].pulseview
    }
    const updatedItem = {...profiles[currentItemIndex], ...update};

    const newProfiles = [...profiles];
    newProfiles[currentItemIndex] = updatedItem;
    setProfiles(newProfiles);
  }
  
  useEffect(()=>{
    var filteredProfiles = profiles;

    if(profileClass){
      filteredProfiles = _.filter(profiles, function(el, i){
        return profileClass.indexOf(el.name) >= 0
      })
    }

    if(filter){
      filteredProfiles = filteredProfiles.filter((profile, idx) => (profile.type === filter || profile.pulseview || profile.active))
    }
    else{
      filteredProfiles = filteredProfiles.filter((profile, idx) => (profile.type === "recording" || profile.pulseview))
    }
    filteredProfiles = filteredProfiles.sort((a,b)=> { return a.id - b.id})
    setSubProfiles(filteredProfiles)
  }, [profiles, screen])
 

  return (
    <>
        { title && <h2> {title} </h2>}
        <Row className="profiles paddedlr">
          {subProfiles.map((profile, idx) => {
              return (
                <Pill key={profile.id} onClick={() => handleUpdate(profile.id)} profile={profile} setProfiles={setProfiles} color={palette?palette[profile.id%palette.length]: null}>
                </Pill>
              )
            })}
        </Row>
    </>
  )
}




export {RheoSenseView, RheoPulseView, RheoSensingView, RheoLibraryView, ViewPicker, RheoGallery};

