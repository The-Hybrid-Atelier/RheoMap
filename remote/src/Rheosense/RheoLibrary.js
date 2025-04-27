import React, { useState, useEffect} from 'react';
import {Row, Column, Pill} from '../Components.js';
import {RheoGallery} from "./RheoInterface.js"
import {cornsyrupLibrary, ecologyLibrary, claySlipLibrary, timeShiftLibrary, stressTestingLibrary} from "./RheoResults.js"




const RheoLibraryView = function({socket, screen, profiles, setProfiles}){
    
    return (
     <>
        <Column className="library-view">
            <RheoGallery screen={screen} title="Reference Materials" filter="reference" socket={socket} profiles={profiles} profileClass={ecologyLibrary} setProfiles={setProfiles}></RheoGallery>
            <RheoGallery screen={screen} title="Concentration" filter="reference" socket={socket} profiles={profiles} profileClass={cornsyrupLibrary} setProfiles={setProfiles}></RheoGallery>
            <RheoGallery screen={screen} title="Time Shift" filter="reference" socket={socket} profiles={profiles} profileClass={timeShiftLibrary} setProfiles={setProfiles}></RheoGallery>
            <RheoGallery screen={screen} title="Clay" filter="reference" socket={socket} profiles={profiles} profileClass={claySlipLibrary} setProfiles={setProfiles}></RheoGallery>
            <RheoGallery screen={screen} title="Stress Tests" filter="reference" socket={socket} profiles={profiles} profileClass={stressTestingLibrary} setProfiles={setProfiles}></RheoGallery>
        </Column>
     </>
     );	

}
export {RheoLibraryView};