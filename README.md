# RheoMap (CHI 2025)

### Mapping inks, gels, pastes, and slurries within a rheological embedding space using retraction-extrusion pressure sensor vectors

![RheoMap](https://raw.githubusercontent.com/The-Hybrid-Atelier/RheoMap/main/image/demo2.png)

This repository is the official implementation of the CHI2025 paper *[RheoMap](https://dl.acm.org/doi/10.1145/3706598.3713835)*.

## Authors

- [Hoang "Charlie" Vuong](https://hybridatelier.uta.edu/members/159-charlie-vuong)
- [Cesar Torres](http://hybridatelier.uta.edu/members/1-cesar-torres)

## Folder Structure

- hardware - Arduino code and hookup instructions for pairing a Adafruit Feather WiFi to the Programmable Air.
- model - a python script that loads a sci-learn python model; the script generates a server that listens for data and sends out model inferences.
- remote - a React app that facilitates communication between the hardware and model; it presents results from the RheoPulse routine back to the user.
- server - a ruby script that runs a Websocket server

## Contributing

The material available through this repository is open-source under the MIT License.
We welcome contributions of community-made designs! You can either submit a pull request via Github or send us a link to your Instructables, Thingiverse, or design files to /Anonymized for Submission/

## Remote

Navigate to:
`cd remote`

Install dependencies:
`npm install`

Run the web server:
`npm start`

## Model (Python)

Requires python >= 3.0
`python model.py`

### Emulate a READ or MODEL event

Requires python >= 3.0
`python server-emulator.py`
