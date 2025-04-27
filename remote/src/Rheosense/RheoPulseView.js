import React, { useState, useEffect, useRef} from 'react';
import {makeProfile, makePlot, makeBarPlot, makeLinePlot, plotProfile, makeBarProfile, resizePlot} from "./RheoPlot.js"
import {RheoGallery, ViewPicker, RheoWebSocket} from "./RheoInterface.js"
import {results, colors, pulses} from "./RheoResults.js"
import {setupPaper, exportSVG} from "./Paper.js"
import {View} from '../View.js';
import {palette} from './Palettes.js'
import {Row, Column} from '../Components.js'

let FileSaver = require('file-saver');
let _ = require('underscore');
let paper = require('paper');


function loadResult(material, pca, color){
  // .slice(0, 5)
  let res = {active: true, name: material.toUpperCase(), pca: pca, color: color}
  return res
}

function getActiveProfiles(profiles){
  return _.filter(profiles, function(el, i){
    return el.pulseview && el.active;
  })
}
const capitalize = word => word.charAt(0).toUpperCase()+ word.slice(1)
function Prediction({profile}){
  return (
    <Row id="lda" className="paddedtb">
      {`Prediction: ${capitalize(profile.data.lda.category)} (${(profile.data.lda.confidence*100).toFixed(2)}%)`}
    </Row>
  )
}


function pca_distance(m1, m2){
  var a = m2[0] - m1[0]
  var b = m2[1] - m1[1]
  var c = m2[2] - m1[2]
  var d = m2[3] - m1[3]

  var distance = Math.sqrt(a * a + b * b + c * c + d * d);
  return distance
}
function Comparator({profiles}){
  const [diff, setDiff] = useState("")
  useEffect(()=>{
    if(profiles.length > 0){
      var d = pca_distance(profiles[0]["data"]["pca"], profiles[1]["data"]["pca"])
      setDiff(1 / (1 + d))
    }
    
  }, [profiles])
  return (
    <Row id="lda" className="paddedtb">
      {`Comparison: ${(diff * 100).toFixed(2)}% similar`}
    </Row>
  )
}

function RheoPulseView({ socket, screen, startView, setScreen, profiles, setProfiles, children }) {
    const [view, setView] = useState(startView)
    const [paperReady, setPaperReady] = useState(false)

    const canvasRef = useRef(null);  
    useEffect(() => {
      setupPaper(canvasRef)
      setPaperReady(true)
    }, [])

    useEffect(()=>{
      console.log("PROFILES", profiles)
      if(paperReady && paper.project){
        paper.project.clear()
        if(view === "pca"){
            makeBarPlot();  
            _.each(profiles, function(el, i){ if(el.active && el.pulseview){makeBarProfile(el.data, palette[el.id%palette.length]); }})
            // paper.view.zoom = 1
            // console.log("PCA", profiles)
            resizePlot();
          }else if(view === "profile"){
            makeLinePlot()
            _.each(profiles, function(el, i){ if(el.active && el.pulseview){plotProfile(el.data, palette[el.id%palette.length]); }})
            paper.view.zoom = 1
          }
      }
    }, [paperReady, view, profiles])
  
    return (
      <Column className="fh">
        
        <div id="canvas-wrapper">
          <ViewPicker view={view} setView={setView} exportSVG={exportSVG}></ViewPicker>
          <canvas ref={canvasRef} resize="true" width="100%" height="100%"></canvas>
          
            {getActiveProfiles(profiles).length == 1 && 
              <Prediction profile={getActiveProfiles(profiles)[0]}>
              </Prediction>
            }
            {getActiveProfiles(profiles).length == 2 && 
              <Comparator profiles={getActiveProfiles(profiles)}>
              </Comparator>
            }
          
          <View icon="library" goToScreen="library" setScreen={setScreen} padding="">
            <RheoGallery socket={socket} profiles={profiles} setProfiles={setProfiles} palette={palette}></RheoGallery>
          </View> 
        </div>
      </Column>
    );
}

export {RheoPulseView};